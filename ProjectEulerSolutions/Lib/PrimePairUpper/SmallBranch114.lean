import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates342 : List Nat := [2339, 2423, 2843, 2963, 3137, 3221, 3533, 3701, 5189, 5261, 5297, 5927, 5987, 6311, 6701, 7229, 7253, 7907, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates342_checked : candidates edge 5 0 2333 tail343 best.bound = rootCandidates342 := by
  unfold candidates
  have row : edge 2333 = PrimePairCache.smallRow2333.contains := by funext q; rfl
  rw [row]
  decide

def innerTail342_19 : List Nat := []
theorem innerScan342_19 : scan edge (search edge 3) 4 [2333] 2333 innerTail342_19 best = best := rfl

def innerTail342_18 : List Nat := 8387 :: innerTail342_19
def innerCandidates342_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates342_18_checked : candidates edge 4 2333 8387 innerTail342_19 best.bound = innerCandidates342_18 := by
  unfold candidates
  have row : edge 8387 = SearchData.row8387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan342_18 : scan edge (search edge 3) 4 [2333] 2333 innerTail342_18 best = best := by
  rw [innerTail342_18, scan, if_neg (by decide), innerCandidates342_18_checked]
  change scan edge (search edge 3) 4 [2333] 2333 innerTail342_19
    (if 3 ≤ innerCandidates342_18.length then search edge 3 [2333, 8387] innerCandidates342_18 10720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan342_19

def innerTail342_17 : List Nat := 7907 :: innerTail342_18
def innerCandidates342_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates342_17_checked : candidates edge 4 2333 7907 innerTail342_18 best.bound = innerCandidates342_17 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan342_17 : scan edge (search edge 3) 4 [2333] 2333 innerTail342_17 best = best := by
  rw [innerTail342_17, scan, if_neg (by decide), innerCandidates342_17_checked]
  change scan edge (search edge 3) 4 [2333] 2333 innerTail342_18
    (if 3 ≤ innerCandidates342_17.length then search edge 3 [2333, 7907] innerCandidates342_17 10240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan342_18

def innerTail342_16 : List Nat := 7253 :: innerTail342_17
def innerCandidates342_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates342_16_checked : candidates edge 4 2333 7253 innerTail342_17 best.bound = innerCandidates342_16 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan342_16 : scan edge (search edge 3) 4 [2333] 2333 innerTail342_16 best = best := by
  rw [innerTail342_16, scan, if_neg (by decide), innerCandidates342_16_checked]
  change scan edge (search edge 3) 4 [2333] 2333 innerTail342_17
    (if 3 ≤ innerCandidates342_16.length then search edge 3 [2333, 7253] innerCandidates342_16 9586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan342_17

def innerTail342_15 : List Nat := 7229 :: innerTail342_16
def innerCandidates342_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates342_15_checked : candidates edge 4 2333 7229 innerTail342_16 best.bound = innerCandidates342_15 := by
  unfold candidates
  have row : edge 7229 = SearchData.row7229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan342_15 : scan edge (search edge 3) 4 [2333] 2333 innerTail342_15 best = best := by
  rw [innerTail342_15, scan, if_neg (by decide), innerCandidates342_15_checked]
  change scan edge (search edge 3) 4 [2333] 2333 innerTail342_16
    (if 3 ≤ innerCandidates342_15.length then search edge 3 [2333, 7229] innerCandidates342_15 9562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan342_16

def innerTail342_14 : List Nat := 6701 :: innerTail342_15
def innerCandidates342_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates342_14_checked : candidates edge 4 2333 6701 innerTail342_15 best.bound = innerCandidates342_14 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan342_14 : scan edge (search edge 3) 4 [2333] 2333 innerTail342_14 best = best := by
  rw [innerTail342_14, scan, if_neg (by decide), innerCandidates342_14_checked]
  change scan edge (search edge 3) 4 [2333] 2333 innerTail342_15
    (if 3 ≤ innerCandidates342_14.length then search edge 3 [2333, 6701] innerCandidates342_14 9034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan342_15

def innerTail342_13 : List Nat := 6311 :: innerTail342_14
def innerCandidates342_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates342_13_checked : candidates edge 4 2333 6311 innerTail342_14 best.bound = innerCandidates342_13 := by
  unfold candidates
  have row : edge 6311 = SearchData.row6311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan342_13 : scan edge (search edge 3) 4 [2333] 2333 innerTail342_13 best = best := by
  rw [innerTail342_13, scan, if_neg (by decide), innerCandidates342_13_checked]
  change scan edge (search edge 3) 4 [2333] 2333 innerTail342_14
    (if 3 ≤ innerCandidates342_13.length then search edge 3 [2333, 6311] innerCandidates342_13 8644 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan342_14

def innerTail342_12 : List Nat := 5987 :: innerTail342_13
def innerCandidates342_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates342_12_checked : candidates edge 4 2333 5987 innerTail342_13 best.bound = innerCandidates342_12 := by
  unfold candidates
  have row : edge 5987 = SearchData.row5987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan342_12 : scan edge (search edge 3) 4 [2333] 2333 innerTail342_12 best = best := by
  rw [innerTail342_12, scan, if_neg (by decide), innerCandidates342_12_checked]
  change scan edge (search edge 3) 4 [2333] 2333 innerTail342_13
    (if 3 ≤ innerCandidates342_12.length then search edge 3 [2333, 5987] innerCandidates342_12 8320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan342_13

def innerTail342_11 : List Nat := 5927 :: innerTail342_12
def innerCandidates342_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates342_11_checked : candidates edge 4 2333 5927 innerTail342_12 best.bound = innerCandidates342_11 := by
  unfold candidates
  have row : edge 5927 = SearchData.row5927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan342_11 : scan edge (search edge 3) 4 [2333] 2333 innerTail342_11 best = best := by
  rw [innerTail342_11, scan, if_neg (by decide), innerCandidates342_11_checked]
  change scan edge (search edge 3) 4 [2333] 2333 innerTail342_12
    (if 3 ≤ innerCandidates342_11.length then search edge 3 [2333, 5927] innerCandidates342_11 8260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan342_12

def innerTail342_10 : List Nat := 5297 :: innerTail342_11
def innerCandidates342_10 : List Nat := [6701]
set_option maxRecDepth 4096 in
theorem innerCandidates342_10_checked : candidates edge 4 2333 5297 innerTail342_11 best.bound = innerCandidates342_10 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan342_10 : scan edge (search edge 3) 4 [2333] 2333 innerTail342_10 best = best := by
  rw [innerTail342_10, scan, if_neg (by decide), innerCandidates342_10_checked]
  change scan edge (search edge 3) 4 [2333] 2333 innerTail342_11
    (if 3 ≤ innerCandidates342_10.length then search edge 3 [2333, 5297] innerCandidates342_10 7630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan342_11

def innerTail342_9 : List Nat := 5261 :: innerTail342_10
def innerCandidates342_9 : List Nat := [5987, 8387]
set_option maxRecDepth 4096 in
theorem innerCandidates342_9_checked : candidates edge 4 2333 5261 innerTail342_10 best.bound = innerCandidates342_9 := by
  unfold candidates
  have row : edge 5261 = SearchData.row5261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan342_9 : scan edge (search edge 3) 4 [2333] 2333 innerTail342_9 best = best := by
  rw [innerTail342_9, scan, if_neg (by decide), innerCandidates342_9_checked]
  change scan edge (search edge 3) 4 [2333] 2333 innerTail342_10
    (if 3 ≤ innerCandidates342_9.length then search edge 3 [2333, 5261] innerCandidates342_9 7594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan342_10

def innerTail342_8 : List Nat := 5189 :: innerTail342_9
def innerCandidates342_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates342_8_checked : candidates edge 4 2333 5189 innerTail342_9 best.bound = innerCandidates342_8 := by
  unfold candidates
  have row : edge 5189 = SearchData.row5189.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan342_8 : scan edge (search edge 3) 4 [2333] 2333 innerTail342_8 best = best := by
  rw [innerTail342_8, scan, if_neg (by decide), innerCandidates342_8_checked]
  change scan edge (search edge 3) 4 [2333] 2333 innerTail342_9
    (if 3 ≤ innerCandidates342_8.length then search edge 3 [2333, 5189] innerCandidates342_8 7522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan342_9

def innerTail342_7 : List Nat := 3701 :: innerTail342_8
def innerCandidates342_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates342_7_checked : candidates edge 4 2333 3701 innerTail342_8 best.bound = innerCandidates342_7 := by
  unfold candidates
  have row : edge 3701 = SearchData.row3701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan342_7 : scan edge (search edge 3) 4 [2333] 2333 innerTail342_7 best = best := by
  rw [innerTail342_7, scan, if_neg (by decide), innerCandidates342_7_checked]
  change scan edge (search edge 3) 4 [2333] 2333 innerTail342_8
    (if 3 ≤ innerCandidates342_7.length then search edge 3 [2333, 3701] innerCandidates342_7 6034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan342_8

def innerTail342_6 : List Nat := 3533 :: innerTail342_7
def innerCandidates342_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates342_6_checked : candidates edge 4 2333 3533 innerTail342_7 best.bound = innerCandidates342_6 := by
  unfold candidates
  have row : edge 3533 = SearchData.row3533.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan342_6 : scan edge (search edge 3) 4 [2333] 2333 innerTail342_6 best = best := by
  rw [innerTail342_6, scan, if_neg (by decide), innerCandidates342_6_checked]
  change scan edge (search edge 3) 4 [2333] 2333 innerTail342_7
    (if 3 ≤ innerCandidates342_6.length then search edge 3 [2333, 3533] innerCandidates342_6 5866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan342_7

def innerTail342_5 : List Nat := 3221 :: innerTail342_6
def innerCandidates342_5 : List Nat := [6311]
set_option maxRecDepth 4096 in
theorem innerCandidates342_5_checked : candidates edge 4 2333 3221 innerTail342_6 best.bound = innerCandidates342_5 := by
  unfold candidates
  have row : edge 3221 = SearchData.row3221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan342_5 : scan edge (search edge 3) 4 [2333] 2333 innerTail342_5 best = best := by
  rw [innerTail342_5, scan, if_neg (by decide), innerCandidates342_5_checked]
  change scan edge (search edge 3) 4 [2333] 2333 innerTail342_6
    (if 3 ≤ innerCandidates342_5.length then search edge 3 [2333, 3221] innerCandidates342_5 5554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan342_6

def innerTail342_4 : List Nat := 3137 :: innerTail342_5
def innerCandidates342_4 : List Nat := [5189, 6311, 8387]
set_option maxRecDepth 4096 in
theorem innerCandidates342_4_checked : candidates edge 4 2333 3137 innerTail342_5 best.bound = innerCandidates342_4 := by
  unfold candidates
  have row : edge 3137 = SearchData.row3137.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild342_4 : search edge 3 [2333, 3137] innerCandidates342_4 5470 best = best := by decide

theorem innerScan342_4 : scan edge (search edge 3) 4 [2333] 2333 innerTail342_4 best = best := by
  rw [innerTail342_4, scan, if_neg (by decide), innerCandidates342_4_checked]
  change scan edge (search edge 3) 4 [2333] 2333 innerTail342_5
    (if 3 ≤ innerCandidates342_4.length then search edge 3 [2333, 3137] innerCandidates342_4 5470 best else best) = best
  rw [if_pos (by decide), innerChild342_4]
  exact innerScan342_5

def innerTail342_3 : List Nat := 2963 :: innerTail342_4
def innerCandidates342_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates342_3_checked : candidates edge 4 2333 2963 innerTail342_4 best.bound = innerCandidates342_3 := by
  unfold candidates
  have row : edge 2963 = SearchData.row2963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan342_3 : scan edge (search edge 3) 4 [2333] 2333 innerTail342_3 best = best := by
  rw [innerTail342_3, scan, if_neg (by decide), innerCandidates342_3_checked]
  change scan edge (search edge 3) 4 [2333] 2333 innerTail342_4
    (if 3 ≤ innerCandidates342_3.length then search edge 3 [2333, 2963] innerCandidates342_3 5296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan342_4

def innerTail342_2 : List Nat := 2843 :: innerTail342_3
def innerCandidates342_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates342_2_checked : candidates edge 4 2333 2843 innerTail342_3 best.bound = innerCandidates342_2 := by
  unfold candidates
  have row : edge 2843 = SearchData.row2843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan342_2 : scan edge (search edge 3) 4 [2333] 2333 innerTail342_2 best = best := by
  rw [innerTail342_2, scan, if_neg (by decide), innerCandidates342_2_checked]
  change scan edge (search edge 3) 4 [2333] 2333 innerTail342_3
    (if 3 ≤ innerCandidates342_2.length then search edge 3 [2333, 2843] innerCandidates342_2 5176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan342_3

def innerTail342_1 : List Nat := 2423 :: innerTail342_2
def innerCandidates342_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates342_1_checked : candidates edge 4 2333 2423 innerTail342_2 best.bound = innerCandidates342_1 := by
  unfold candidates
  have row : edge 2423 = PrimePairCache.smallRow2423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan342_1 : scan edge (search edge 3) 4 [2333] 2333 innerTail342_1 best = best := by
  rw [innerTail342_1, scan, if_neg (by decide), innerCandidates342_1_checked]
  change scan edge (search edge 3) 4 [2333] 2333 innerTail342_2
    (if 3 ≤ innerCandidates342_1.length then search edge 3 [2333, 2423] innerCandidates342_1 4756 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan342_2

def innerTail342_0 : List Nat := 2339 :: innerTail342_1
def innerCandidates342_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates342_0_checked : candidates edge 4 2333 2339 innerTail342_1 best.bound = innerCandidates342_0 := by
  unfold candidates
  have row : edge 2339 = PrimePairCache.smallRow2339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan342_0 : scan edge (search edge 3) 4 [2333] 2333 innerTail342_0 best = best := by
  rw [innerTail342_0, scan, if_neg (by decide), innerCandidates342_0_checked]
  change scan edge (search edge 3) 4 [2333] 2333 innerTail342_1
    (if 3 ≤ innerCandidates342_0.length then search edge 3 [2333, 2339] innerCandidates342_0 4672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan342_1

theorem rootBranch342 : search edge 4 [2333] rootCandidates342 2333 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2333] 2333 innerTail342_0 best = best
  exact innerScan342_0

def rootCandidates341 : List Nat := [2647, 3061, 3313, 3529, 3847, 3943, 4549, 5179, 6199, 6469, 6619, 6871, 6883, 7639, 8287]
set_option maxRecDepth 4096 in
theorem rootCandidates341_checked : candidates edge 5 0 2311 tail342 best.bound = rootCandidates341 := by
  unfold candidates
  have row : edge 2311 = PrimePairCache.smallRow2311.contains := by funext q; rfl
  rw [row]
  decide

def innerTail341_15 : List Nat := []
theorem innerScan341_15 : scan edge (search edge 3) 4 [2311] 2311 innerTail341_15 best = best := rfl

def innerTail341_14 : List Nat := 8287 :: innerTail341_15
def innerCandidates341_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates341_14_checked : candidates edge 4 2311 8287 innerTail341_15 best.bound = innerCandidates341_14 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan341_14 : scan edge (search edge 3) 4 [2311] 2311 innerTail341_14 best = best := by
  rw [innerTail341_14, scan, if_neg (by decide), innerCandidates341_14_checked]
  change scan edge (search edge 3) 4 [2311] 2311 innerTail341_15
    (if 3 ≤ innerCandidates341_14.length then search edge 3 [2311, 8287] innerCandidates341_14 10598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan341_15

def innerTail341_13 : List Nat := 7639 :: innerTail341_14
def innerCandidates341_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates341_13_checked : candidates edge 4 2311 7639 innerTail341_14 best.bound = innerCandidates341_13 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan341_13 : scan edge (search edge 3) 4 [2311] 2311 innerTail341_13 best = best := by
  rw [innerTail341_13, scan, if_neg (by decide), innerCandidates341_13_checked]
  change scan edge (search edge 3) 4 [2311] 2311 innerTail341_14
    (if 3 ≤ innerCandidates341_13.length then search edge 3 [2311, 7639] innerCandidates341_13 9950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan341_14

def innerTail341_12 : List Nat := 6883 :: innerTail341_13
def innerCandidates341_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates341_12_checked : candidates edge 4 2311 6883 innerTail341_13 best.bound = innerCandidates341_12 := by
  unfold candidates
  have row : edge 6883 = SearchData.row6883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan341_12 : scan edge (search edge 3) 4 [2311] 2311 innerTail341_12 best = best := by
  rw [innerTail341_12, scan, if_neg (by decide), innerCandidates341_12_checked]
  change scan edge (search edge 3) 4 [2311] 2311 innerTail341_13
    (if 3 ≤ innerCandidates341_12.length then search edge 3 [2311, 6883] innerCandidates341_12 9194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan341_13

def innerTail341_11 : List Nat := 6871 :: innerTail341_12
def innerCandidates341_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates341_11_checked : candidates edge 4 2311 6871 innerTail341_12 best.bound = innerCandidates341_11 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan341_11 : scan edge (search edge 3) 4 [2311] 2311 innerTail341_11 best = best := by
  rw [innerTail341_11, scan, if_neg (by decide), innerCandidates341_11_checked]
  change scan edge (search edge 3) 4 [2311] 2311 innerTail341_12
    (if 3 ≤ innerCandidates341_11.length then search edge 3 [2311, 6871] innerCandidates341_11 9182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan341_12

def innerTail341_10 : List Nat := 6619 :: innerTail341_11
def innerCandidates341_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates341_10_checked : candidates edge 4 2311 6619 innerTail341_11 best.bound = innerCandidates341_10 := by
  unfold candidates
  have row : edge 6619 = SearchData.row6619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan341_10 : scan edge (search edge 3) 4 [2311] 2311 innerTail341_10 best = best := by
  rw [innerTail341_10, scan, if_neg (by decide), innerCandidates341_10_checked]
  change scan edge (search edge 3) 4 [2311] 2311 innerTail341_11
    (if 3 ≤ innerCandidates341_10.length then search edge 3 [2311, 6619] innerCandidates341_10 8930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan341_11

def innerTail341_9 : List Nat := 6469 :: innerTail341_10
def innerCandidates341_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates341_9_checked : candidates edge 4 2311 6469 innerTail341_10 best.bound = innerCandidates341_9 := by
  unfold candidates
  have row : edge 6469 = SearchData.row6469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan341_9 : scan edge (search edge 3) 4 [2311] 2311 innerTail341_9 best = best := by
  rw [innerTail341_9, scan, if_neg (by decide), innerCandidates341_9_checked]
  change scan edge (search edge 3) 4 [2311] 2311 innerTail341_10
    (if 3 ≤ innerCandidates341_9.length then search edge 3 [2311, 6469] innerCandidates341_9 8780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan341_10

def innerTail341_8 : List Nat := 6199 :: innerTail341_9
def innerCandidates341_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates341_8_checked : candidates edge 4 2311 6199 innerTail341_9 best.bound = innerCandidates341_8 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan341_8 : scan edge (search edge 3) 4 [2311] 2311 innerTail341_8 best = best := by
  rw [innerTail341_8, scan, if_neg (by decide), innerCandidates341_8_checked]
  change scan edge (search edge 3) 4 [2311] 2311 innerTail341_9
    (if 3 ≤ innerCandidates341_8.length then search edge 3 [2311, 6199] innerCandidates341_8 8510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan341_9

def innerTail341_7 : List Nat := 5179 :: innerTail341_8
def innerCandidates341_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates341_7_checked : candidates edge 4 2311 5179 innerTail341_8 best.bound = innerCandidates341_7 := by
  unfold candidates
  have row : edge 5179 = SearchData.row5179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan341_7 : scan edge (search edge 3) 4 [2311] 2311 innerTail341_7 best = best := by
  rw [innerTail341_7, scan, if_neg (by decide), innerCandidates341_7_checked]
  change scan edge (search edge 3) 4 [2311] 2311 innerTail341_8
    (if 3 ≤ innerCandidates341_7.length then search edge 3 [2311, 5179] innerCandidates341_7 7490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan341_8

def innerTail341_6 : List Nat := 4549 :: innerTail341_7
def innerCandidates341_6 : List Nat := [8287]
set_option maxRecDepth 4096 in
theorem innerCandidates341_6_checked : candidates edge 4 2311 4549 innerTail341_7 best.bound = innerCandidates341_6 := by
  unfold candidates
  have row : edge 4549 = SearchData.row4549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan341_6 : scan edge (search edge 3) 4 [2311] 2311 innerTail341_6 best = best := by
  rw [innerTail341_6, scan, if_neg (by decide), innerCandidates341_6_checked]
  change scan edge (search edge 3) 4 [2311] 2311 innerTail341_7
    (if 3 ≤ innerCandidates341_6.length then search edge 3 [2311, 4549] innerCandidates341_6 6860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan341_7

def innerTail341_5 : List Nat := 3943 :: innerTail341_6
def innerCandidates341_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates341_5_checked : candidates edge 4 2311 3943 innerTail341_6 best.bound = innerCandidates341_5 := by
  unfold candidates
  have row : edge 3943 = SearchData.row3943.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan341_5 : scan edge (search edge 3) 4 [2311] 2311 innerTail341_5 best = best := by
  rw [innerTail341_5, scan, if_neg (by decide), innerCandidates341_5_checked]
  change scan edge (search edge 3) 4 [2311] 2311 innerTail341_6
    (if 3 ≤ innerCandidates341_5.length then search edge 3 [2311, 3943] innerCandidates341_5 6254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan341_6

def innerTail341_4 : List Nat := 3847 :: innerTail341_5
def innerCandidates341_4 : List Nat := [6883]
set_option maxRecDepth 4096 in
theorem innerCandidates341_4_checked : candidates edge 4 2311 3847 innerTail341_5 best.bound = innerCandidates341_4 := by
  unfold candidates
  have row : edge 3847 = SearchData.row3847.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan341_4 : scan edge (search edge 3) 4 [2311] 2311 innerTail341_4 best = best := by
  rw [innerTail341_4, scan, if_neg (by decide), innerCandidates341_4_checked]
  change scan edge (search edge 3) 4 [2311] 2311 innerTail341_5
    (if 3 ≤ innerCandidates341_4.length then search edge 3 [2311, 3847] innerCandidates341_4 6158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan341_5

def innerTail341_3 : List Nat := 3529 :: innerTail341_4
def innerCandidates341_3 : List Nat := [3943]
set_option maxRecDepth 4096 in
theorem innerCandidates341_3_checked : candidates edge 4 2311 3529 innerTail341_4 best.bound = innerCandidates341_3 := by
  unfold candidates
  have row : edge 3529 = SearchData.row3529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan341_3 : scan edge (search edge 3) 4 [2311] 2311 innerTail341_3 best = best := by
  rw [innerTail341_3, scan, if_neg (by decide), innerCandidates341_3_checked]
  change scan edge (search edge 3) 4 [2311] 2311 innerTail341_4
    (if 3 ≤ innerCandidates341_3.length then search edge 3 [2311, 3529] innerCandidates341_3 5840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan341_4

def innerTail341_2 : List Nat := 3313 :: innerTail341_3
def innerCandidates341_2 : List Nat := [6871]
set_option maxRecDepth 4096 in
theorem innerCandidates341_2_checked : candidates edge 4 2311 3313 innerTail341_3 best.bound = innerCandidates341_2 := by
  unfold candidates
  have row : edge 3313 = SearchData.row3313.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan341_2 : scan edge (search edge 3) 4 [2311] 2311 innerTail341_2 best = best := by
  rw [innerTail341_2, scan, if_neg (by decide), innerCandidates341_2_checked]
  change scan edge (search edge 3) 4 [2311] 2311 innerTail341_3
    (if 3 ≤ innerCandidates341_2.length then search edge 3 [2311, 3313] innerCandidates341_2 5624 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan341_3

def innerTail341_1 : List Nat := 3061 :: innerTail341_2
def innerCandidates341_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates341_1_checked : candidates edge 4 2311 3061 innerTail341_2 best.bound = innerCandidates341_1 := by
  unfold candidates
  have row : edge 3061 = SearchData.row3061.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan341_1 : scan edge (search edge 3) 4 [2311] 2311 innerTail341_1 best = best := by
  rw [innerTail341_1, scan, if_neg (by decide), innerCandidates341_1_checked]
  change scan edge (search edge 3) 4 [2311] 2311 innerTail341_2
    (if 3 ≤ innerCandidates341_1.length then search edge 3 [2311, 3061] innerCandidates341_1 5372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan341_2

def innerTail341_0 : List Nat := 2647 :: innerTail341_1
def innerCandidates341_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates341_0_checked : candidates edge 4 2311 2647 innerTail341_1 best.bound = innerCandidates341_0 := by
  unfold candidates
  have row : edge 2647 = PrimePairCache.smallRow2647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan341_0 : scan edge (search edge 3) 4 [2311] 2311 innerTail341_0 best = best := by
  rw [innerTail341_0, scan, if_neg (by decide), innerCandidates341_0_checked]
  change scan edge (search edge 3) 4 [2311] 2311 innerTail341_1
    (if 3 ≤ innerCandidates341_0.length then search edge 3 [2311, 2647] innerCandidates341_0 4958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan341_1

theorem rootBranch341 : search edge 4 [2311] rootCandidates341 2311 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2311] 2311 innerTail341_0 best = best
  exact innerScan341_0

def rootCandidates340 : List Nat := [2687, 3491, 4133, 4451, 5081, 5087, 5417, 5783, 6131, 6257, 6947, 7691, 7817, 7829, 8237]
set_option maxRecDepth 4096 in
theorem rootCandidates340_checked : candidates edge 5 0 2309 tail341 best.bound = rootCandidates340 := by
  unfold candidates
  have row : edge 2309 = PrimePairCache.smallRow2309.contains := by funext q; rfl
  rw [row]
  decide

def innerTail340_15 : List Nat := []
theorem innerScan340_15 : scan edge (search edge 3) 4 [2309] 2309 innerTail340_15 best = best := rfl

def innerTail340_14 : List Nat := 8237 :: innerTail340_15
def innerCandidates340_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates340_14_checked : candidates edge 4 2309 8237 innerTail340_15 best.bound = innerCandidates340_14 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan340_14 : scan edge (search edge 3) 4 [2309] 2309 innerTail340_14 best = best := by
  rw [innerTail340_14, scan, if_neg (by decide), innerCandidates340_14_checked]
  change scan edge (search edge 3) 4 [2309] 2309 innerTail340_15
    (if 3 ≤ innerCandidates340_14.length then search edge 3 [2309, 8237] innerCandidates340_14 10546 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan340_15

def innerTail340_13 : List Nat := 7829 :: innerTail340_14
def innerCandidates340_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates340_13_checked : candidates edge 4 2309 7829 innerTail340_14 best.bound = innerCandidates340_13 := by
  unfold candidates
  have row : edge 7829 = SearchData.row7829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan340_13 : scan edge (search edge 3) 4 [2309] 2309 innerTail340_13 best = best := by
  rw [innerTail340_13, scan, if_neg (by decide), innerCandidates340_13_checked]
  change scan edge (search edge 3) 4 [2309] 2309 innerTail340_14
    (if 3 ≤ innerCandidates340_13.length then search edge 3 [2309, 7829] innerCandidates340_13 10138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan340_14

def innerTail340_12 : List Nat := 7817 :: innerTail340_13
def innerCandidates340_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates340_12_checked : candidates edge 4 2309 7817 innerTail340_13 best.bound = innerCandidates340_12 := by
  unfold candidates
  have row : edge 7817 = SearchData.row7817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan340_12 : scan edge (search edge 3) 4 [2309] 2309 innerTail340_12 best = best := by
  rw [innerTail340_12, scan, if_neg (by decide), innerCandidates340_12_checked]
  change scan edge (search edge 3) 4 [2309] 2309 innerTail340_13
    (if 3 ≤ innerCandidates340_12.length then search edge 3 [2309, 7817] innerCandidates340_12 10126 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan340_13

def innerTail340_11 : List Nat := 7691 :: innerTail340_12
def innerCandidates340_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates340_11_checked : candidates edge 4 2309 7691 innerTail340_12 best.bound = innerCandidates340_11 := by
  unfold candidates
  have row : edge 7691 = SearchData.row7691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan340_11 : scan edge (search edge 3) 4 [2309] 2309 innerTail340_11 best = best := by
  rw [innerTail340_11, scan, if_neg (by decide), innerCandidates340_11_checked]
  change scan edge (search edge 3) 4 [2309] 2309 innerTail340_12
    (if 3 ≤ innerCandidates340_11.length then search edge 3 [2309, 7691] innerCandidates340_11 10000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan340_12

def innerTail340_10 : List Nat := 6947 :: innerTail340_11
def innerCandidates340_10 : List Nat := [7829]
set_option maxRecDepth 4096 in
theorem innerCandidates340_10_checked : candidates edge 4 2309 6947 innerTail340_11 best.bound = innerCandidates340_10 := by
  unfold candidates
  have row : edge 6947 = SearchData.row6947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan340_10 : scan edge (search edge 3) 4 [2309] 2309 innerTail340_10 best = best := by
  rw [innerTail340_10, scan, if_neg (by decide), innerCandidates340_10_checked]
  change scan edge (search edge 3) 4 [2309] 2309 innerTail340_11
    (if 3 ≤ innerCandidates340_10.length then search edge 3 [2309, 6947] innerCandidates340_10 9256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan340_11

def innerTail340_9 : List Nat := 6257 :: innerTail340_10
def innerCandidates340_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates340_9_checked : candidates edge 4 2309 6257 innerTail340_10 best.bound = innerCandidates340_9 := by
  unfold candidates
  have row : edge 6257 = SearchData.row6257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan340_9 : scan edge (search edge 3) 4 [2309] 2309 innerTail340_9 best = best := by
  rw [innerTail340_9, scan, if_neg (by decide), innerCandidates340_9_checked]
  change scan edge (search edge 3) 4 [2309] 2309 innerTail340_10
    (if 3 ≤ innerCandidates340_9.length then search edge 3 [2309, 6257] innerCandidates340_9 8566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan340_10

def innerTail340_8 : List Nat := 6131 :: innerTail340_9
def innerCandidates340_8 : List Nat := [6947]
set_option maxRecDepth 4096 in
theorem innerCandidates340_8_checked : candidates edge 4 2309 6131 innerTail340_9 best.bound = innerCandidates340_8 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan340_8 : scan edge (search edge 3) 4 [2309] 2309 innerTail340_8 best = best := by
  rw [innerTail340_8, scan, if_neg (by decide), innerCandidates340_8_checked]
  change scan edge (search edge 3) 4 [2309] 2309 innerTail340_9
    (if 3 ≤ innerCandidates340_8.length then search edge 3 [2309, 6131] innerCandidates340_8 8440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan340_9

def innerTail340_7 : List Nat := 5783 :: innerTail340_8
def innerCandidates340_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates340_7_checked : candidates edge 4 2309 5783 innerTail340_8 best.bound = innerCandidates340_7 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan340_7 : scan edge (search edge 3) 4 [2309] 2309 innerTail340_7 best = best := by
  rw [innerTail340_7, scan, if_neg (by decide), innerCandidates340_7_checked]
  change scan edge (search edge 3) 4 [2309] 2309 innerTail340_8
    (if 3 ≤ innerCandidates340_7.length then search edge 3 [2309, 5783] innerCandidates340_7 8092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan340_8

def innerTail340_6 : List Nat := 5417 :: innerTail340_7
def innerCandidates340_6 : List Nat := [6131]
set_option maxRecDepth 4096 in
theorem innerCandidates340_6_checked : candidates edge 4 2309 5417 innerTail340_7 best.bound = innerCandidates340_6 := by
  unfold candidates
  have row : edge 5417 = SearchData.row5417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan340_6 : scan edge (search edge 3) 4 [2309] 2309 innerTail340_6 best = best := by
  rw [innerTail340_6, scan, if_neg (by decide), innerCandidates340_6_checked]
  change scan edge (search edge 3) 4 [2309] 2309 innerTail340_7
    (if 3 ≤ innerCandidates340_6.length then search edge 3 [2309, 5417] innerCandidates340_6 7726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan340_7

def innerTail340_5 : List Nat := 5087 :: innerTail340_6
def innerCandidates340_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates340_5_checked : candidates edge 4 2309 5087 innerTail340_6 best.bound = innerCandidates340_5 := by
  unfold candidates
  have row : edge 5087 = SearchData.row5087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan340_5 : scan edge (search edge 3) 4 [2309] 2309 innerTail340_5 best = best := by
  rw [innerTail340_5, scan, if_neg (by decide), innerCandidates340_5_checked]
  change scan edge (search edge 3) 4 [2309] 2309 innerTail340_6
    (if 3 ≤ innerCandidates340_5.length then search edge 3 [2309, 5087] innerCandidates340_5 7396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan340_6

def innerTail340_4 : List Nat := 5081 :: innerTail340_5
def innerCandidates340_4 : List Nat := [5783, 7817]
set_option maxRecDepth 4096 in
theorem innerCandidates340_4_checked : candidates edge 4 2309 5081 innerTail340_5 best.bound = innerCandidates340_4 := by
  unfold candidates
  have row : edge 5081 = SearchData.row5081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan340_4 : scan edge (search edge 3) 4 [2309] 2309 innerTail340_4 best = best := by
  rw [innerTail340_4, scan, if_neg (by decide), innerCandidates340_4_checked]
  change scan edge (search edge 3) 4 [2309] 2309 innerTail340_5
    (if 3 ≤ innerCandidates340_4.length then search edge 3 [2309, 5081] innerCandidates340_4 7390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan340_5

def innerTail340_3 : List Nat := 4451 :: innerTail340_4
def innerCandidates340_3 : List Nat := [6257]
set_option maxRecDepth 4096 in
theorem innerCandidates340_3_checked : candidates edge 4 2309 4451 innerTail340_4 best.bound = innerCandidates340_3 := by
  unfold candidates
  have row : edge 4451 = SearchData.row4451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan340_3 : scan edge (search edge 3) 4 [2309] 2309 innerTail340_3 best = best := by
  rw [innerTail340_3, scan, if_neg (by decide), innerCandidates340_3_checked]
  change scan edge (search edge 3) 4 [2309] 2309 innerTail340_4
    (if 3 ≤ innerCandidates340_3.length then search edge 3 [2309, 4451] innerCandidates340_3 6760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan340_4

def innerTail340_2 : List Nat := 4133 :: innerTail340_3
def innerCandidates340_2 : List Nat := [5081, 6947]
set_option maxRecDepth 4096 in
theorem innerCandidates340_2_checked : candidates edge 4 2309 4133 innerTail340_3 best.bound = innerCandidates340_2 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan340_2 : scan edge (search edge 3) 4 [2309] 2309 innerTail340_2 best = best := by
  rw [innerTail340_2, scan, if_neg (by decide), innerCandidates340_2_checked]
  change scan edge (search edge 3) 4 [2309] 2309 innerTail340_3
    (if 3 ≤ innerCandidates340_2.length then search edge 3 [2309, 4133] innerCandidates340_2 6442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan340_3

def innerTail340_1 : List Nat := 3491 :: innerTail340_2
def innerCandidates340_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates340_1_checked : candidates edge 4 2309 3491 innerTail340_2 best.bound = innerCandidates340_1 := by
  unfold candidates
  have row : edge 3491 = SearchData.row3491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan340_1 : scan edge (search edge 3) 4 [2309] 2309 innerTail340_1 best = best := by
  rw [innerTail340_1, scan, if_neg (by decide), innerCandidates340_1_checked]
  change scan edge (search edge 3) 4 [2309] 2309 innerTail340_2
    (if 3 ≤ innerCandidates340_1.length then search edge 3 [2309, 3491] innerCandidates340_1 5800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan340_2

def innerTail340_0 : List Nat := 2687 :: innerTail340_1
def innerCandidates340_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates340_0_checked : candidates edge 4 2309 2687 innerTail340_1 best.bound = innerCandidates340_0 := by
  unfold candidates
  have row : edge 2687 = SearchData.row2687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan340_0 : scan edge (search edge 3) 4 [2309] 2309 innerTail340_0 best = best := by
  rw [innerTail340_0, scan, if_neg (by decide), innerCandidates340_0_checked]
  change scan edge (search edge 3) 4 [2309] 2309 innerTail340_1
    (if 3 ≤ innerCandidates340_0.length then search edge 3 [2309, 2687] innerCandidates340_0 4996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan340_1

theorem rootBranch340 : search edge 4 [2309] rootCandidates340 2309 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2309] 2309 innerTail340_0 best = best
  exact innerScan340_0

def rootCandidates339 : List Nat := [2381, 2531, 3581, 4073, 4967, 5081, 6089, 6269, 7127, 7937]
set_option maxRecDepth 4096 in
theorem rootCandidates339_checked : candidates edge 5 0 2297 tail340 best.bound = rootCandidates339 := by
  unfold candidates
  have row : edge 2297 = PrimePairCache.smallRow2297.contains := by funext q; rfl
  rw [row]
  decide

def innerTail339_10 : List Nat := []
theorem innerScan339_10 : scan edge (search edge 3) 4 [2297] 2297 innerTail339_10 best = best := rfl

def innerTail339_9 : List Nat := 7937 :: innerTail339_10
def innerCandidates339_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates339_9_checked : candidates edge 4 2297 7937 innerTail339_10 best.bound = innerCandidates339_9 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan339_9 : scan edge (search edge 3) 4 [2297] 2297 innerTail339_9 best = best := by
  rw [innerTail339_9, scan, if_neg (by decide), innerCandidates339_9_checked]
  change scan edge (search edge 3) 4 [2297] 2297 innerTail339_10
    (if 3 ≤ innerCandidates339_9.length then search edge 3 [2297, 7937] innerCandidates339_9 10234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan339_10

def innerTail339_8 : List Nat := 7127 :: innerTail339_9
def innerCandidates339_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates339_8_checked : candidates edge 4 2297 7127 innerTail339_9 best.bound = innerCandidates339_8 := by
  unfold candidates
  have row : edge 7127 = SearchData.row7127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan339_8 : scan edge (search edge 3) 4 [2297] 2297 innerTail339_8 best = best := by
  rw [innerTail339_8, scan, if_neg (by decide), innerCandidates339_8_checked]
  change scan edge (search edge 3) 4 [2297] 2297 innerTail339_9
    (if 3 ≤ innerCandidates339_8.length then search edge 3 [2297, 7127] innerCandidates339_8 9424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan339_9

def innerTail339_7 : List Nat := 6269 :: innerTail339_8
def innerCandidates339_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates339_7_checked : candidates edge 4 2297 6269 innerTail339_8 best.bound = innerCandidates339_7 := by
  unfold candidates
  have row : edge 6269 = SearchData.row6269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan339_7 : scan edge (search edge 3) 4 [2297] 2297 innerTail339_7 best = best := by
  rw [innerTail339_7, scan, if_neg (by decide), innerCandidates339_7_checked]
  change scan edge (search edge 3) 4 [2297] 2297 innerTail339_8
    (if 3 ≤ innerCandidates339_7.length then search edge 3 [2297, 6269] innerCandidates339_7 8566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan339_8

def innerTail339_6 : List Nat := 6089 :: innerTail339_7
def innerCandidates339_6 : List Nat := [7127]
set_option maxRecDepth 4096 in
theorem innerCandidates339_6_checked : candidates edge 4 2297 6089 innerTail339_7 best.bound = innerCandidates339_6 := by
  unfold candidates
  have row : edge 6089 = SearchData.row6089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan339_6 : scan edge (search edge 3) 4 [2297] 2297 innerTail339_6 best = best := by
  rw [innerTail339_6, scan, if_neg (by decide), innerCandidates339_6_checked]
  change scan edge (search edge 3) 4 [2297] 2297 innerTail339_7
    (if 3 ≤ innerCandidates339_6.length then search edge 3 [2297, 6089] innerCandidates339_6 8386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan339_7

def innerTail339_5 : List Nat := 5081 :: innerTail339_6
def innerCandidates339_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates339_5_checked : candidates edge 4 2297 5081 innerTail339_6 best.bound = innerCandidates339_5 := by
  unfold candidates
  have row : edge 5081 = SearchData.row5081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan339_5 : scan edge (search edge 3) 4 [2297] 2297 innerTail339_5 best = best := by
  rw [innerTail339_5, scan, if_neg (by decide), innerCandidates339_5_checked]
  change scan edge (search edge 3) 4 [2297] 2297 innerTail339_6
    (if 3 ≤ innerCandidates339_5.length then search edge 3 [2297, 5081] innerCandidates339_5 7378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan339_6

def innerTail339_4 : List Nat := 4967 :: innerTail339_5
def innerCandidates339_4 : List Nat := [7127]
set_option maxRecDepth 4096 in
theorem innerCandidates339_4_checked : candidates edge 4 2297 4967 innerTail339_5 best.bound = innerCandidates339_4 := by
  unfold candidates
  have row : edge 4967 = SearchData.row4967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan339_4 : scan edge (search edge 3) 4 [2297] 2297 innerTail339_4 best = best := by
  rw [innerTail339_4, scan, if_neg (by decide), innerCandidates339_4_checked]
  change scan edge (search edge 3) 4 [2297] 2297 innerTail339_5
    (if 3 ≤ innerCandidates339_4.length then search edge 3 [2297, 4967] innerCandidates339_4 7264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan339_5

def innerTail339_3 : List Nat := 4073 :: innerTail339_4
def innerCandidates339_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates339_3_checked : candidates edge 4 2297 4073 innerTail339_4 best.bound = innerCandidates339_3 := by
  unfold candidates
  have row : edge 4073 = SearchData.row4073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan339_3 : scan edge (search edge 3) 4 [2297] 2297 innerTail339_3 best = best := by
  rw [innerTail339_3, scan, if_neg (by decide), innerCandidates339_3_checked]
  change scan edge (search edge 3) 4 [2297] 2297 innerTail339_4
    (if 3 ≤ innerCandidates339_3.length then search edge 3 [2297, 4073] innerCandidates339_3 6370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan339_4

def innerTail339_2 : List Nat := 3581 :: innerTail339_3
def innerCandidates339_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates339_2_checked : candidates edge 4 2297 3581 innerTail339_3 best.bound = innerCandidates339_2 := by
  unfold candidates
  have row : edge 3581 = SearchData.row3581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan339_2 : scan edge (search edge 3) 4 [2297] 2297 innerTail339_2 best = best := by
  rw [innerTail339_2, scan, if_neg (by decide), innerCandidates339_2_checked]
  change scan edge (search edge 3) 4 [2297] 2297 innerTail339_3
    (if 3 ≤ innerCandidates339_2.length then search edge 3 [2297, 3581] innerCandidates339_2 5878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan339_3

def innerTail339_1 : List Nat := 2531 :: innerTail339_2
def innerCandidates339_1 : List Nat := [4967]
set_option maxRecDepth 4096 in
theorem innerCandidates339_1_checked : candidates edge 4 2297 2531 innerTail339_2 best.bound = innerCandidates339_1 := by
  unfold candidates
  have row : edge 2531 = PrimePairCache.smallRow2531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan339_1 : scan edge (search edge 3) 4 [2297] 2297 innerTail339_1 best = best := by
  rw [innerTail339_1, scan, if_neg (by decide), innerCandidates339_1_checked]
  change scan edge (search edge 3) 4 [2297] 2297 innerTail339_2
    (if 3 ≤ innerCandidates339_1.length then search edge 3 [2297, 2531] innerCandidates339_1 4828 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan339_2

def innerTail339_0 : List Nat := 2381 :: innerTail339_1
def innerCandidates339_0 : List Nat := [4073]
set_option maxRecDepth 4096 in
theorem innerCandidates339_0_checked : candidates edge 4 2297 2381 innerTail339_1 best.bound = innerCandidates339_0 := by
  unfold candidates
  have row : edge 2381 = PrimePairCache.smallRow2381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan339_0 : scan edge (search edge 3) 4 [2297] 2297 innerTail339_0 best = best := by
  rw [innerTail339_0, scan, if_neg (by decide), innerCandidates339_0_checked]
  change scan edge (search edge 3) 4 [2297] 2297 innerTail339_1
    (if 3 ≤ innerCandidates339_0.length then search edge 3 [2297, 2381] innerCandidates339_0 4678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan339_1

theorem rootBranch339 : search edge 4 [2297] rootCandidates339 2297 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2297] 2297 innerTail339_0 best = best
  exact innerScan339_0

end ProjectEulerSolutions.PrimePairUpper.Search
