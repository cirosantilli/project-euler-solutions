import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates194 : List Nat := [1303, 1459, 1621, 1873, 2017, 2137, 2551, 3229, 3637, 3769, 3931, 4201, 4663, 4861, 5167, 5281, 5953, 6337, 6577, 7207, 7459]
set_option maxRecDepth 4096 in
theorem rootCandidates194_checked : candidates edge 5 0 1201 tail195 best.bound = rootCandidates194 := by
  unfold candidates
  have row : edge 1201 = PrimePairCache.smallRow1201.contains := by funext q; rfl
  rw [row]
  decide

def innerTail194_21 : List Nat := []
theorem innerScan194_21 : scan edge (search edge 3) 4 [1201] 1201 innerTail194_21 best = best := rfl

def innerTail194_20 : List Nat := 7459 :: innerTail194_21
def innerCandidates194_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates194_20_checked : candidates edge 4 1201 7459 innerTail194_21 best.bound = innerCandidates194_20 := by
  unfold candidates
  have row : edge 7459 = SearchData.row7459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan194_20 : scan edge (search edge 3) 4 [1201] 1201 innerTail194_20 best = best := by
  rw [innerTail194_20, scan, if_neg (by decide), innerCandidates194_20_checked]
  change scan edge (search edge 3) 4 [1201] 1201 innerTail194_21
    (if 3 ≤ innerCandidates194_20.length then search edge 3 [1201, 7459] innerCandidates194_20 8660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan194_21

def innerTail194_19 : List Nat := 7207 :: innerTail194_20
def innerCandidates194_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates194_19_checked : candidates edge 4 1201 7207 innerTail194_20 best.bound = innerCandidates194_19 := by
  unfold candidates
  have row : edge 7207 = SearchData.row7207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan194_19 : scan edge (search edge 3) 4 [1201] 1201 innerTail194_19 best = best := by
  rw [innerTail194_19, scan, if_neg (by decide), innerCandidates194_19_checked]
  change scan edge (search edge 3) 4 [1201] 1201 innerTail194_20
    (if 3 ≤ innerCandidates194_19.length then search edge 3 [1201, 7207] innerCandidates194_19 8408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan194_20

def innerTail194_18 : List Nat := 6577 :: innerTail194_19
def innerCandidates194_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates194_18_checked : candidates edge 4 1201 6577 innerTail194_19 best.bound = innerCandidates194_18 := by
  unfold candidates
  have row : edge 6577 = SearchData.row6577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan194_18 : scan edge (search edge 3) 4 [1201] 1201 innerTail194_18 best = best := by
  rw [innerTail194_18, scan, if_neg (by decide), innerCandidates194_18_checked]
  change scan edge (search edge 3) 4 [1201] 1201 innerTail194_19
    (if 3 ≤ innerCandidates194_18.length then search edge 3 [1201, 6577] innerCandidates194_18 7778 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan194_19

def innerTail194_17 : List Nat := 6337 :: innerTail194_18
def innerCandidates194_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates194_17_checked : candidates edge 4 1201 6337 innerTail194_18 best.bound = innerCandidates194_17 := by
  unfold candidates
  have row : edge 6337 = SearchData.row6337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan194_17 : scan edge (search edge 3) 4 [1201] 1201 innerTail194_17 best = best := by
  rw [innerTail194_17, scan, if_neg (by decide), innerCandidates194_17_checked]
  change scan edge (search edge 3) 4 [1201] 1201 innerTail194_18
    (if 3 ≤ innerCandidates194_17.length then search edge 3 [1201, 6337] innerCandidates194_17 7538 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan194_18

def innerTail194_16 : List Nat := 5953 :: innerTail194_17
def innerCandidates194_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates194_16_checked : candidates edge 4 1201 5953 innerTail194_17 best.bound = innerCandidates194_16 := by
  unfold candidates
  have row : edge 5953 = SearchData.row5953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan194_16 : scan edge (search edge 3) 4 [1201] 1201 innerTail194_16 best = best := by
  rw [innerTail194_16, scan, if_neg (by decide), innerCandidates194_16_checked]
  change scan edge (search edge 3) 4 [1201] 1201 innerTail194_17
    (if 3 ≤ innerCandidates194_16.length then search edge 3 [1201, 5953] innerCandidates194_16 7154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan194_17

def innerTail194_15 : List Nat := 5281 :: innerTail194_16
def innerCandidates194_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates194_15_checked : candidates edge 4 1201 5281 innerTail194_16 best.bound = innerCandidates194_15 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan194_15 : scan edge (search edge 3) 4 [1201] 1201 innerTail194_15 best = best := by
  rw [innerTail194_15, scan, if_neg (by decide), innerCandidates194_15_checked]
  change scan edge (search edge 3) 4 [1201] 1201 innerTail194_16
    (if 3 ≤ innerCandidates194_15.length then search edge 3 [1201, 5281] innerCandidates194_15 6482 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan194_16

def innerTail194_14 : List Nat := 5167 :: innerTail194_15
def innerCandidates194_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates194_14_checked : candidates edge 4 1201 5167 innerTail194_15 best.bound = innerCandidates194_14 := by
  unfold candidates
  have row : edge 5167 = SearchData.row5167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan194_14 : scan edge (search edge 3) 4 [1201] 1201 innerTail194_14 best = best := by
  rw [innerTail194_14, scan, if_neg (by decide), innerCandidates194_14_checked]
  change scan edge (search edge 3) 4 [1201] 1201 innerTail194_15
    (if 3 ≤ innerCandidates194_14.length then search edge 3 [1201, 5167] innerCandidates194_14 6368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan194_15

def innerTail194_13 : List Nat := 4861 :: innerTail194_14
def innerCandidates194_13 : List Nat := [5953]
set_option maxRecDepth 4096 in
theorem innerCandidates194_13_checked : candidates edge 4 1201 4861 innerTail194_14 best.bound = innerCandidates194_13 := by
  unfold candidates
  have row : edge 4861 = SearchData.row4861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan194_13 : scan edge (search edge 3) 4 [1201] 1201 innerTail194_13 best = best := by
  rw [innerTail194_13, scan, if_neg (by decide), innerCandidates194_13_checked]
  change scan edge (search edge 3) 4 [1201] 1201 innerTail194_14
    (if 3 ≤ innerCandidates194_13.length then search edge 3 [1201, 4861] innerCandidates194_13 6062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan194_14

def innerTail194_12 : List Nat := 4663 :: innerTail194_13
def innerCandidates194_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates194_12_checked : candidates edge 4 1201 4663 innerTail194_13 best.bound = innerCandidates194_12 := by
  unfold candidates
  have row : edge 4663 = SearchData.row4663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan194_12 : scan edge (search edge 3) 4 [1201] 1201 innerTail194_12 best = best := by
  rw [innerTail194_12, scan, if_neg (by decide), innerCandidates194_12_checked]
  change scan edge (search edge 3) 4 [1201] 1201 innerTail194_13
    (if 3 ≤ innerCandidates194_12.length then search edge 3 [1201, 4663] innerCandidates194_12 5864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan194_13

def innerTail194_11 : List Nat := 4201 :: innerTail194_12
def innerCandidates194_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates194_11_checked : candidates edge 4 1201 4201 innerTail194_12 best.bound = innerCandidates194_11 := by
  unfold candidates
  have row : edge 4201 = SearchData.row4201.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan194_11 : scan edge (search edge 3) 4 [1201] 1201 innerTail194_11 best = best := by
  rw [innerTail194_11, scan, if_neg (by decide), innerCandidates194_11_checked]
  change scan edge (search edge 3) 4 [1201] 1201 innerTail194_12
    (if 3 ≤ innerCandidates194_11.length then search edge 3 [1201, 4201] innerCandidates194_11 5402 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan194_12

def innerTail194_10 : List Nat := 3931 :: innerTail194_11
def innerCandidates194_10 : List Nat := [7459]
set_option maxRecDepth 4096 in
theorem innerCandidates194_10_checked : candidates edge 4 1201 3931 innerTail194_11 best.bound = innerCandidates194_10 := by
  unfold candidates
  have row : edge 3931 = SearchData.row3931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan194_10 : scan edge (search edge 3) 4 [1201] 1201 innerTail194_10 best = best := by
  rw [innerTail194_10, scan, if_neg (by decide), innerCandidates194_10_checked]
  change scan edge (search edge 3) 4 [1201] 1201 innerTail194_11
    (if 3 ≤ innerCandidates194_10.length then search edge 3 [1201, 3931] innerCandidates194_10 5132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan194_11

def innerTail194_9 : List Nat := 3769 :: innerTail194_10
def innerCandidates194_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates194_9_checked : candidates edge 4 1201 3769 innerTail194_10 best.bound = innerCandidates194_9 := by
  unfold candidates
  have row : edge 3769 = SearchData.row3769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan194_9 : scan edge (search edge 3) 4 [1201] 1201 innerTail194_9 best = best := by
  rw [innerTail194_9, scan, if_neg (by decide), innerCandidates194_9_checked]
  change scan edge (search edge 3) 4 [1201] 1201 innerTail194_10
    (if 3 ≤ innerCandidates194_9.length then search edge 3 [1201, 3769] innerCandidates194_9 4970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan194_10

def innerTail194_8 : List Nat := 3637 :: innerTail194_9
def innerCandidates194_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates194_8_checked : candidates edge 4 1201 3637 innerTail194_9 best.bound = innerCandidates194_8 := by
  unfold candidates
  have row : edge 3637 = SearchData.row3637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan194_8 : scan edge (search edge 3) 4 [1201] 1201 innerTail194_8 best = best := by
  rw [innerTail194_8, scan, if_neg (by decide), innerCandidates194_8_checked]
  change scan edge (search edge 3) 4 [1201] 1201 innerTail194_9
    (if 3 ≤ innerCandidates194_8.length then search edge 3 [1201, 3637] innerCandidates194_8 4838 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan194_9

def innerTail194_7 : List Nat := 3229 :: innerTail194_8
def innerCandidates194_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates194_7_checked : candidates edge 4 1201 3229 innerTail194_8 best.bound = innerCandidates194_7 := by
  unfold candidates
  have row : edge 3229 = SearchData.row3229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan194_7 : scan edge (search edge 3) 4 [1201] 1201 innerTail194_7 best = best := by
  rw [innerTail194_7, scan, if_neg (by decide), innerCandidates194_7_checked]
  change scan edge (search edge 3) 4 [1201] 1201 innerTail194_8
    (if 3 ≤ innerCandidates194_7.length then search edge 3 [1201, 3229] innerCandidates194_7 4430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan194_8

def innerTail194_6 : List Nat := 2551 :: innerTail194_7
def innerCandidates194_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates194_6_checked : candidates edge 4 1201 2551 innerTail194_7 best.bound = innerCandidates194_6 := by
  unfold candidates
  have row : edge 2551 = PrimePairCache.smallRow2551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan194_6 : scan edge (search edge 3) 4 [1201] 1201 innerTail194_6 best = best := by
  rw [innerTail194_6, scan, if_neg (by decide), innerCandidates194_6_checked]
  change scan edge (search edge 3) 4 [1201] 1201 innerTail194_7
    (if 3 ≤ innerCandidates194_6.length then search edge 3 [1201, 2551] innerCandidates194_6 3752 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan194_7

def innerTail194_5 : List Nat := 2137 :: innerTail194_6
def innerCandidates194_5 : List Nat := [7459]
set_option maxRecDepth 4096 in
theorem innerCandidates194_5_checked : candidates edge 4 1201 2137 innerTail194_6 best.bound = innerCandidates194_5 := by
  unfold candidates
  have row : edge 2137 = PrimePairCache.smallRow2137.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan194_5 : scan edge (search edge 3) 4 [1201] 1201 innerTail194_5 best = best := by
  rw [innerTail194_5, scan, if_neg (by decide), innerCandidates194_5_checked]
  change scan edge (search edge 3) 4 [1201] 1201 innerTail194_6
    (if 3 ≤ innerCandidates194_5.length then search edge 3 [1201, 2137] innerCandidates194_5 3338 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan194_6

def innerTail194_4 : List Nat := 2017 :: innerTail194_5
def innerCandidates194_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates194_4_checked : candidates edge 4 1201 2017 innerTail194_5 best.bound = innerCandidates194_4 := by
  unfold candidates
  have row : edge 2017 = PrimePairCache.smallRow2017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan194_4 : scan edge (search edge 3) 4 [1201] 1201 innerTail194_4 best = best := by
  rw [innerTail194_4, scan, if_neg (by decide), innerCandidates194_4_checked]
  change scan edge (search edge 3) 4 [1201] 1201 innerTail194_5
    (if 3 ≤ innerCandidates194_4.length then search edge 3 [1201, 2017] innerCandidates194_4 3218 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan194_5

def innerTail194_3 : List Nat := 1873 :: innerTail194_4
def innerCandidates194_3 : List Nat := [2017]
set_option maxRecDepth 4096 in
theorem innerCandidates194_3_checked : candidates edge 4 1201 1873 innerTail194_4 best.bound = innerCandidates194_3 := by
  unfold candidates
  have row : edge 1873 = PrimePairCache.smallRow1873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan194_3 : scan edge (search edge 3) 4 [1201] 1201 innerTail194_3 best = best := by
  rw [innerTail194_3, scan, if_neg (by decide), innerCandidates194_3_checked]
  change scan edge (search edge 3) 4 [1201] 1201 innerTail194_4
    (if 3 ≤ innerCandidates194_3.length then search edge 3 [1201, 1873] innerCandidates194_3 3074 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan194_4

def innerTail194_2 : List Nat := 1621 :: innerTail194_3
def innerCandidates194_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates194_2_checked : candidates edge 4 1201 1621 innerTail194_3 best.bound = innerCandidates194_2 := by
  unfold candidates
  have row : edge 1621 = PrimePairCache.smallRow1621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan194_2 : scan edge (search edge 3) 4 [1201] 1201 innerTail194_2 best = best := by
  rw [innerTail194_2, scan, if_neg (by decide), innerCandidates194_2_checked]
  change scan edge (search edge 3) 4 [1201] 1201 innerTail194_3
    (if 3 ≤ innerCandidates194_2.length then search edge 3 [1201, 1621] innerCandidates194_2 2822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan194_3

def innerTail194_1 : List Nat := 1459 :: innerTail194_2
def innerCandidates194_1 : List Nat := [1873]
set_option maxRecDepth 4096 in
theorem innerCandidates194_1_checked : candidates edge 4 1201 1459 innerTail194_2 best.bound = innerCandidates194_1 := by
  unfold candidates
  have row : edge 1459 = PrimePairCache.smallRow1459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan194_1 : scan edge (search edge 3) 4 [1201] 1201 innerTail194_1 best = best := by
  rw [innerTail194_1, scan, if_neg (by decide), innerCandidates194_1_checked]
  change scan edge (search edge 3) 4 [1201] 1201 innerTail194_2
    (if 3 ≤ innerCandidates194_1.length then search edge 3 [1201, 1459] innerCandidates194_1 2660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan194_2

def innerTail194_0 : List Nat := 1303 :: innerTail194_1
def innerCandidates194_0 : List Nat := [7207]
set_option maxRecDepth 4096 in
theorem innerCandidates194_0_checked : candidates edge 4 1201 1303 innerTail194_1 best.bound = innerCandidates194_0 := by
  unfold candidates
  have row : edge 1303 = PrimePairCache.smallRow1303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan194_0 : scan edge (search edge 3) 4 [1201] 1201 innerTail194_0 best = best := by
  rw [innerTail194_0, scan, if_neg (by decide), innerCandidates194_0_checked]
  change scan edge (search edge 3) 4 [1201] 1201 innerTail194_1
    (if 3 ≤ innerCandidates194_0.length then search edge 3 [1201, 1303] innerCandidates194_0 2504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan194_1

theorem rootBranch194 : search edge 4 [1201] rootCandidates194 1201 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1201] 1201 innerTail194_0 best = best
  exact innerScan194_0

def rootCandidates193 : List Nat := [1553, 1889, 1901, 2153, 2441, 2777, 3671, 3923, 4133, 4931, 5009, 5507, 5519, 5801, 6131, 6689, 6779, 7103, 7193, 7559, 7829, 8039]
set_option maxRecDepth 4096 in
theorem rootCandidates193_checked : candidates edge 5 0 1193 tail194 best.bound = rootCandidates193 := by
  unfold candidates
  have row : edge 1193 = PrimePairCache.smallRow1193.contains := by funext q; rfl
  rw [row]
  decide

def innerTail193_22 : List Nat := []
theorem innerScan193_22 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_22 best = best := rfl

def innerTail193_21 : List Nat := 8039 :: innerTail193_22
def innerCandidates193_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates193_21_checked : candidates edge 4 1193 8039 innerTail193_22 best.bound = innerCandidates193_21 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan193_21 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_21 best = best := by
  rw [innerTail193_21, scan, if_neg (by decide), innerCandidates193_21_checked]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_22
    (if 3 ≤ innerCandidates193_21.length then search edge 3 [1193, 8039] innerCandidates193_21 9232 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan193_22

def innerTail193_20 : List Nat := 7829 :: innerTail193_21
def innerCandidates193_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates193_20_checked : candidates edge 4 1193 7829 innerTail193_21 best.bound = innerCandidates193_20 := by
  unfold candidates
  have row : edge 7829 = SearchData.row7829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan193_20 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_20 best = best := by
  rw [innerTail193_20, scan, if_neg (by decide), innerCandidates193_20_checked]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_21
    (if 3 ≤ innerCandidates193_20.length then search edge 3 [1193, 7829] innerCandidates193_20 9022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan193_21

def innerTail193_19 : List Nat := 7559 :: innerTail193_20
def innerCandidates193_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates193_19_checked : candidates edge 4 1193 7559 innerTail193_20 best.bound = innerCandidates193_19 := by
  unfold candidates
  have row : edge 7559 = SearchData.row7559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan193_19 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_19 best = best := by
  rw [innerTail193_19, scan, if_neg (by decide), innerCandidates193_19_checked]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_20
    (if 3 ≤ innerCandidates193_19.length then search edge 3 [1193, 7559] innerCandidates193_19 8752 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan193_20

def innerTail193_18 : List Nat := 7193 :: innerTail193_19
def innerCandidates193_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates193_18_checked : candidates edge 4 1193 7193 innerTail193_19 best.bound = innerCandidates193_18 := by
  unfold candidates
  have row : edge 7193 = SearchData.row7193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan193_18 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_18 best = best := by
  rw [innerTail193_18, scan, if_neg (by decide), innerCandidates193_18_checked]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_19
    (if 3 ≤ innerCandidates193_18.length then search edge 3 [1193, 7193] innerCandidates193_18 8386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan193_19

def innerTail193_17 : List Nat := 7103 :: innerTail193_18
def innerCandidates193_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates193_17_checked : candidates edge 4 1193 7103 innerTail193_18 best.bound = innerCandidates193_17 := by
  unfold candidates
  have row : edge 7103 = SearchData.row7103.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan193_17 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_17 best = best := by
  rw [innerTail193_17, scan, if_neg (by decide), innerCandidates193_17_checked]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_18
    (if 3 ≤ innerCandidates193_17.length then search edge 3 [1193, 7103] innerCandidates193_17 8296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan193_18

def innerTail193_16 : List Nat := 6779 :: innerTail193_17
def innerCandidates193_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates193_16_checked : candidates edge 4 1193 6779 innerTail193_17 best.bound = innerCandidates193_16 := by
  unfold candidates
  have row : edge 6779 = SearchData.row6779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan193_16 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_16 best = best := by
  rw [innerTail193_16, scan, if_neg (by decide), innerCandidates193_16_checked]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_17
    (if 3 ≤ innerCandidates193_16.length then search edge 3 [1193, 6779] innerCandidates193_16 7972 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan193_17

def innerTail193_15 : List Nat := 6689 :: innerTail193_16
def innerCandidates193_15 : List Nat := [6779]
set_option maxRecDepth 4096 in
theorem innerCandidates193_15_checked : candidates edge 4 1193 6689 innerTail193_16 best.bound = innerCandidates193_15 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan193_15 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_15 best = best := by
  rw [innerTail193_15, scan, if_neg (by decide), innerCandidates193_15_checked]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_16
    (if 3 ≤ innerCandidates193_15.length then search edge 3 [1193, 6689] innerCandidates193_15 7882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan193_16

def innerTail193_14 : List Nat := 6131 :: innerTail193_15
def innerCandidates193_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates193_14_checked : candidates edge 4 1193 6131 innerTail193_15 best.bound = innerCandidates193_14 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan193_14 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_14 best = best := by
  rw [innerTail193_14, scan, if_neg (by decide), innerCandidates193_14_checked]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_15
    (if 3 ≤ innerCandidates193_14.length then search edge 3 [1193, 6131] innerCandidates193_14 7324 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan193_15

def innerTail193_13 : List Nat := 5801 :: innerTail193_14
def innerCandidates193_13 : List Nat := [6131]
set_option maxRecDepth 4096 in
theorem innerCandidates193_13_checked : candidates edge 4 1193 5801 innerTail193_14 best.bound = innerCandidates193_13 := by
  unfold candidates
  have row : edge 5801 = SearchData.row5801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan193_13 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_13 best = best := by
  rw [innerTail193_13, scan, if_neg (by decide), innerCandidates193_13_checked]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_14
    (if 3 ≤ innerCandidates193_13.length then search edge 3 [1193, 5801] innerCandidates193_13 6994 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan193_14

def innerTail193_12 : List Nat := 5519 :: innerTail193_13
def innerCandidates193_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates193_12_checked : candidates edge 4 1193 5519 innerTail193_13 best.bound = innerCandidates193_12 := by
  unfold candidates
  have row : edge 5519 = SearchData.row5519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan193_12 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_12 best = best := by
  rw [innerTail193_12, scan, if_neg (by decide), innerCandidates193_12_checked]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_13
    (if 3 ≤ innerCandidates193_12.length then search edge 3 [1193, 5519] innerCandidates193_12 6712 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan193_13

def innerTail193_11 : List Nat := 5507 :: innerTail193_12
def innerCandidates193_11 : List Nat := [7829]
set_option maxRecDepth 4096 in
theorem innerCandidates193_11_checked : candidates edge 4 1193 5507 innerTail193_12 best.bound = innerCandidates193_11 := by
  unfold candidates
  have row : edge 5507 = SearchData.row5507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan193_11 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_11 best = best := by
  rw [innerTail193_11, scan, if_neg (by decide), innerCandidates193_11_checked]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_12
    (if 3 ≤ innerCandidates193_11.length then search edge 3 [1193, 5507] innerCandidates193_11 6700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan193_12

def innerTail193_10 : List Nat := 5009 :: innerTail193_11
def innerCandidates193_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates193_10_checked : candidates edge 4 1193 5009 innerTail193_11 best.bound = innerCandidates193_10 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan193_10 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_10 best = best := by
  rw [innerTail193_10, scan, if_neg (by decide), innerCandidates193_10_checked]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_11
    (if 3 ≤ innerCandidates193_10.length then search edge 3 [1193, 5009] innerCandidates193_10 6202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan193_11

def innerTail193_9 : List Nat := 4931 :: innerTail193_10
def innerCandidates193_9 : List Nat := [5009]
set_option maxRecDepth 4096 in
theorem innerCandidates193_9_checked : candidates edge 4 1193 4931 innerTail193_10 best.bound = innerCandidates193_9 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan193_9 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_9 best = best := by
  rw [innerTail193_9, scan, if_neg (by decide), innerCandidates193_9_checked]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_10
    (if 3 ≤ innerCandidates193_9.length then search edge 3 [1193, 4931] innerCandidates193_9 6124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan193_10

def innerTail193_8 : List Nat := 4133 :: innerTail193_9
def innerCandidates193_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates193_8_checked : candidates edge 4 1193 4133 innerTail193_9 best.bound = innerCandidates193_8 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan193_8 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_8 best = best := by
  rw [innerTail193_8, scan, if_neg (by decide), innerCandidates193_8_checked]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_9
    (if 3 ≤ innerCandidates193_8.length then search edge 3 [1193, 4133] innerCandidates193_8 5326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan193_9

def innerTail193_7 : List Nat := 3923 :: innerTail193_8
def innerCandidates193_7 : List Nat := [4133, 5519]
set_option maxRecDepth 4096 in
theorem innerCandidates193_7_checked : candidates edge 4 1193 3923 innerTail193_8 best.bound = innerCandidates193_7 := by
  unfold candidates
  have row : edge 3923 = SearchData.row3923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan193_7 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_7 best = best := by
  rw [innerTail193_7, scan, if_neg (by decide), innerCandidates193_7_checked]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_8
    (if 3 ≤ innerCandidates193_7.length then search edge 3 [1193, 3923] innerCandidates193_7 5116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan193_8

def innerTail193_6 : List Nat := 3671 :: innerTail193_7
def innerCandidates193_6 : List Nat := [6131]
set_option maxRecDepth 4096 in
theorem innerCandidates193_6_checked : candidates edge 4 1193 3671 innerTail193_7 best.bound = innerCandidates193_6 := by
  unfold candidates
  have row : edge 3671 = SearchData.row3671.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan193_6 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_6 best = best := by
  rw [innerTail193_6, scan, if_neg (by decide), innerCandidates193_6_checked]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_7
    (if 3 ≤ innerCandidates193_6.length then search edge 3 [1193, 3671] innerCandidates193_6 4864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan193_7

def innerTail193_5 : List Nat := 2777 :: innerTail193_6
def innerCandidates193_5 : List Nat := [8039]
set_option maxRecDepth 4096 in
theorem innerCandidates193_5_checked : candidates edge 4 1193 2777 innerTail193_6 best.bound = innerCandidates193_5 := by
  unfold candidates
  have row : edge 2777 = SearchData.row2777.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan193_5 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_5 best = best := by
  rw [innerTail193_5, scan, if_neg (by decide), innerCandidates193_5_checked]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_6
    (if 3 ≤ innerCandidates193_5.length then search edge 3 [1193, 2777] innerCandidates193_5 3970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan193_6

def innerTail193_4 : List Nat := 2441 :: innerTail193_5
def innerCandidates193_4 : List Nat := [3671]
set_option maxRecDepth 4096 in
theorem innerCandidates193_4_checked : candidates edge 4 1193 2441 innerTail193_5 best.bound = innerCandidates193_4 := by
  unfold candidates
  have row : edge 2441 = PrimePairCache.smallRow2441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan193_4 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_4 best = best := by
  rw [innerTail193_4, scan, if_neg (by decide), innerCandidates193_4_checked]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_5
    (if 3 ≤ innerCandidates193_4.length then search edge 3 [1193, 2441] innerCandidates193_4 3634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan193_5

def innerTail193_3 : List Nat := 2153 :: innerTail193_4
def innerCandidates193_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates193_3_checked : candidates edge 4 1193 2153 innerTail193_4 best.bound = innerCandidates193_3 := by
  unfold candidates
  have row : edge 2153 = PrimePairCache.smallRow2153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan193_3 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_3 best = best := by
  rw [innerTail193_3, scan, if_neg (by decide), innerCandidates193_3_checked]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_4
    (if 3 ≤ innerCandidates193_3.length then search edge 3 [1193, 2153] innerCandidates193_3 3346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan193_4

def innerTail193_2 : List Nat := 1901 :: innerTail193_3
def innerCandidates193_2 : List Nat := [2153]
set_option maxRecDepth 4096 in
theorem innerCandidates193_2_checked : candidates edge 4 1193 1901 innerTail193_3 best.bound = innerCandidates193_2 := by
  unfold candidates
  have row : edge 1901 = PrimePairCache.smallRow1901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan193_2 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_2 best = best := by
  rw [innerTail193_2, scan, if_neg (by decide), innerCandidates193_2_checked]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_3
    (if 3 ≤ innerCandidates193_2.length then search edge 3 [1193, 1901] innerCandidates193_2 3094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan193_3

def innerTail193_1 : List Nat := 1889 :: innerTail193_2
def innerCandidates193_1 : List Nat := [2777]
set_option maxRecDepth 4096 in
theorem innerCandidates193_1_checked : candidates edge 4 1193 1889 innerTail193_2 best.bound = innerCandidates193_1 := by
  unfold candidates
  have row : edge 1889 = PrimePairCache.smallRow1889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan193_1 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_1 best = best := by
  rw [innerTail193_1, scan, if_neg (by decide), innerCandidates193_1_checked]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_2
    (if 3 ≤ innerCandidates193_1.length then search edge 3 [1193, 1889] innerCandidates193_1 3082 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan193_2

def innerTail193_0 : List Nat := 1553 :: innerTail193_1
def innerCandidates193_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates193_0_checked : candidates edge 4 1193 1553 innerTail193_1 best.bound = innerCandidates193_0 := by
  unfold candidates
  have row : edge 1553 = PrimePairCache.smallRow1553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan193_0 : scan edge (search edge 3) 4 [1193] 1193 innerTail193_0 best = best := by
  rw [innerTail193_0, scan, if_neg (by decide), innerCandidates193_0_checked]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_1
    (if 3 ≤ innerCandidates193_0.length then search edge 3 [1193, 1553] innerCandidates193_0 2746 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan193_1

theorem rootBranch193 : search edge 4 [1193] rootCandidates193 1193 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1193] 1193 innerTail193_0 best = best
  exact innerScan193_0

def rootCandidates192 : List Nat := [1361, 1451, 2879, 4001, 4523, 4649, 4937, 5477, 5861, 6203, 6569, 6701, 6833, 6857, 6983, 7079, 7253, 8093, 8117]
set_option maxRecDepth 4096 in
theorem rootCandidates192_checked : candidates edge 5 0 1187 tail193 best.bound = rootCandidates192 := by
  unfold candidates
  have row : edge 1187 = PrimePairCache.smallRow1187.contains := by funext q; rfl
  rw [row]
  decide

def innerTail192_19 : List Nat := []
theorem innerScan192_19 : scan edge (search edge 3) 4 [1187] 1187 innerTail192_19 best = best := rfl

def innerTail192_18 : List Nat := 8117 :: innerTail192_19
def innerCandidates192_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates192_18_checked : candidates edge 4 1187 8117 innerTail192_19 best.bound = innerCandidates192_18 := by
  unfold candidates
  have row : edge 8117 = SearchData.row8117.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan192_18 : scan edge (search edge 3) 4 [1187] 1187 innerTail192_18 best = best := by
  rw [innerTail192_18, scan, if_neg (by decide), innerCandidates192_18_checked]
  change scan edge (search edge 3) 4 [1187] 1187 innerTail192_19
    (if 3 ≤ innerCandidates192_18.length then search edge 3 [1187, 8117] innerCandidates192_18 9304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan192_19

def innerTail192_17 : List Nat := 8093 :: innerTail192_18
def innerCandidates192_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates192_17_checked : candidates edge 4 1187 8093 innerTail192_18 best.bound = innerCandidates192_17 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan192_17 : scan edge (search edge 3) 4 [1187] 1187 innerTail192_17 best = best := by
  rw [innerTail192_17, scan, if_neg (by decide), innerCandidates192_17_checked]
  change scan edge (search edge 3) 4 [1187] 1187 innerTail192_18
    (if 3 ≤ innerCandidates192_17.length then search edge 3 [1187, 8093] innerCandidates192_17 9280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan192_18

def innerTail192_16 : List Nat := 7253 :: innerTail192_17
def innerCandidates192_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates192_16_checked : candidates edge 4 1187 7253 innerTail192_17 best.bound = innerCandidates192_16 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan192_16 : scan edge (search edge 3) 4 [1187] 1187 innerTail192_16 best = best := by
  rw [innerTail192_16, scan, if_neg (by decide), innerCandidates192_16_checked]
  change scan edge (search edge 3) 4 [1187] 1187 innerTail192_17
    (if 3 ≤ innerCandidates192_16.length then search edge 3 [1187, 7253] innerCandidates192_16 8440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan192_17

def innerTail192_15 : List Nat := 7079 :: innerTail192_16
def innerCandidates192_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates192_15_checked : candidates edge 4 1187 7079 innerTail192_16 best.bound = innerCandidates192_15 := by
  unfold candidates
  have row : edge 7079 = SearchData.row7079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan192_15 : scan edge (search edge 3) 4 [1187] 1187 innerTail192_15 best = best := by
  rw [innerTail192_15, scan, if_neg (by decide), innerCandidates192_15_checked]
  change scan edge (search edge 3) 4 [1187] 1187 innerTail192_16
    (if 3 ≤ innerCandidates192_15.length then search edge 3 [1187, 7079] innerCandidates192_15 8266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan192_16

def innerTail192_14 : List Nat := 6983 :: innerTail192_15
def innerCandidates192_14 : List Nat := [8117]
set_option maxRecDepth 4096 in
theorem innerCandidates192_14_checked : candidates edge 4 1187 6983 innerTail192_15 best.bound = innerCandidates192_14 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan192_14 : scan edge (search edge 3) 4 [1187] 1187 innerTail192_14 best = best := by
  rw [innerTail192_14, scan, if_neg (by decide), innerCandidates192_14_checked]
  change scan edge (search edge 3) 4 [1187] 1187 innerTail192_15
    (if 3 ≤ innerCandidates192_14.length then search edge 3 [1187, 6983] innerCandidates192_14 8170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan192_15

def innerTail192_13 : List Nat := 6857 :: innerTail192_14
def innerCandidates192_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates192_13_checked : candidates edge 4 1187 6857 innerTail192_14 best.bound = innerCandidates192_13 := by
  unfold candidates
  have row : edge 6857 = SearchData.row6857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan192_13 : scan edge (search edge 3) 4 [1187] 1187 innerTail192_13 best = best := by
  rw [innerTail192_13, scan, if_neg (by decide), innerCandidates192_13_checked]
  change scan edge (search edge 3) 4 [1187] 1187 innerTail192_14
    (if 3 ≤ innerCandidates192_13.length then search edge 3 [1187, 6857] innerCandidates192_13 8044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan192_14

def innerTail192_12 : List Nat := 6833 :: innerTail192_13
def innerCandidates192_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates192_12_checked : candidates edge 4 1187 6833 innerTail192_13 best.bound = innerCandidates192_12 := by
  unfold candidates
  have row : edge 6833 = SearchData.row6833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan192_12 : scan edge (search edge 3) 4 [1187] 1187 innerTail192_12 best = best := by
  rw [innerTail192_12, scan, if_neg (by decide), innerCandidates192_12_checked]
  change scan edge (search edge 3) 4 [1187] 1187 innerTail192_13
    (if 3 ≤ innerCandidates192_12.length then search edge 3 [1187, 6833] innerCandidates192_12 8020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan192_13

def innerTail192_11 : List Nat := 6701 :: innerTail192_12
def innerCandidates192_11 : List Nat := [8117]
set_option maxRecDepth 4096 in
theorem innerCandidates192_11_checked : candidates edge 4 1187 6701 innerTail192_12 best.bound = innerCandidates192_11 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan192_11 : scan edge (search edge 3) 4 [1187] 1187 innerTail192_11 best = best := by
  rw [innerTail192_11, scan, if_neg (by decide), innerCandidates192_11_checked]
  change scan edge (search edge 3) 4 [1187] 1187 innerTail192_12
    (if 3 ≤ innerCandidates192_11.length then search edge 3 [1187, 6701] innerCandidates192_11 7888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan192_12

def innerTail192_10 : List Nat := 6569 :: innerTail192_11
def innerCandidates192_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates192_10_checked : candidates edge 4 1187 6569 innerTail192_11 best.bound = innerCandidates192_10 := by
  unfold candidates
  have row : edge 6569 = SearchData.row6569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan192_10 : scan edge (search edge 3) 4 [1187] 1187 innerTail192_10 best = best := by
  rw [innerTail192_10, scan, if_neg (by decide), innerCandidates192_10_checked]
  change scan edge (search edge 3) 4 [1187] 1187 innerTail192_11
    (if 3 ≤ innerCandidates192_10.length then search edge 3 [1187, 6569] innerCandidates192_10 7756 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan192_11

def innerTail192_9 : List Nat := 6203 :: innerTail192_10
def innerCandidates192_9 : List Nat := [7253]
set_option maxRecDepth 4096 in
theorem innerCandidates192_9_checked : candidates edge 4 1187 6203 innerTail192_10 best.bound = innerCandidates192_9 := by
  unfold candidates
  have row : edge 6203 = SearchData.row6203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan192_9 : scan edge (search edge 3) 4 [1187] 1187 innerTail192_9 best = best := by
  rw [innerTail192_9, scan, if_neg (by decide), innerCandidates192_9_checked]
  change scan edge (search edge 3) 4 [1187] 1187 innerTail192_10
    (if 3 ≤ innerCandidates192_9.length then search edge 3 [1187, 6203] innerCandidates192_9 7390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan192_10

def innerTail192_8 : List Nat := 5861 :: innerTail192_9
def innerCandidates192_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates192_8_checked : candidates edge 4 1187 5861 innerTail192_9 best.bound = innerCandidates192_8 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan192_8 : scan edge (search edge 3) 4 [1187] 1187 innerTail192_8 best = best := by
  rw [innerTail192_8, scan, if_neg (by decide), innerCandidates192_8_checked]
  change scan edge (search edge 3) 4 [1187] 1187 innerTail192_9
    (if 3 ≤ innerCandidates192_8.length then search edge 3 [1187, 5861] innerCandidates192_8 7048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan192_9

def innerTail192_7 : List Nat := 5477 :: innerTail192_8
def innerCandidates192_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates192_7_checked : candidates edge 4 1187 5477 innerTail192_8 best.bound = innerCandidates192_7 := by
  unfold candidates
  have row : edge 5477 = SearchData.row5477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan192_7 : scan edge (search edge 3) 4 [1187] 1187 innerTail192_7 best = best := by
  rw [innerTail192_7, scan, if_neg (by decide), innerCandidates192_7_checked]
  change scan edge (search edge 3) 4 [1187] 1187 innerTail192_8
    (if 3 ≤ innerCandidates192_7.length then search edge 3 [1187, 5477] innerCandidates192_7 6664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan192_8

def innerTail192_6 : List Nat := 4937 :: innerTail192_7
def innerCandidates192_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates192_6_checked : candidates edge 4 1187 4937 innerTail192_7 best.bound = innerCandidates192_6 := by
  unfold candidates
  have row : edge 4937 = SearchData.row4937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan192_6 : scan edge (search edge 3) 4 [1187] 1187 innerTail192_6 best = best := by
  rw [innerTail192_6, scan, if_neg (by decide), innerCandidates192_6_checked]
  change scan edge (search edge 3) 4 [1187] 1187 innerTail192_7
    (if 3 ≤ innerCandidates192_6.length then search edge 3 [1187, 4937] innerCandidates192_6 6124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan192_7

def innerTail192_5 : List Nat := 4649 :: innerTail192_6
def innerCandidates192_5 : List Nat := [8117]
set_option maxRecDepth 4096 in
theorem innerCandidates192_5_checked : candidates edge 4 1187 4649 innerTail192_6 best.bound = innerCandidates192_5 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan192_5 : scan edge (search edge 3) 4 [1187] 1187 innerTail192_5 best = best := by
  rw [innerTail192_5, scan, if_neg (by decide), innerCandidates192_5_checked]
  change scan edge (search edge 3) 4 [1187] 1187 innerTail192_6
    (if 3 ≤ innerCandidates192_5.length then search edge 3 [1187, 4649] innerCandidates192_5 5836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan192_6

def innerTail192_4 : List Nat := 4523 :: innerTail192_5
def innerCandidates192_4 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates192_4_checked : candidates edge 4 1187 4523 innerTail192_5 best.bound = innerCandidates192_4 := by
  unfold candidates
  have row : edge 4523 = SearchData.row4523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan192_4 : scan edge (search edge 3) 4 [1187] 1187 innerTail192_4 best = best := by
  rw [innerTail192_4, scan, if_neg (by decide), innerCandidates192_4_checked]
  change scan edge (search edge 3) 4 [1187] 1187 innerTail192_5
    (if 3 ≤ innerCandidates192_4.length then search edge 3 [1187, 4523] innerCandidates192_4 5710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan192_5

def innerTail192_3 : List Nat := 4001 :: innerTail192_4
def innerCandidates192_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates192_3_checked : candidates edge 4 1187 4001 innerTail192_4 best.bound = innerCandidates192_3 := by
  unfold candidates
  have row : edge 4001 = SearchData.row4001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan192_3 : scan edge (search edge 3) 4 [1187] 1187 innerTail192_3 best = best := by
  rw [innerTail192_3, scan, if_neg (by decide), innerCandidates192_3_checked]
  change scan edge (search edge 3) 4 [1187] 1187 innerTail192_4
    (if 3 ≤ innerCandidates192_3.length then search edge 3 [1187, 4001] innerCandidates192_3 5188 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan192_4

def innerTail192_2 : List Nat := 2879 :: innerTail192_3
def innerCandidates192_2 : List Nat := [6701]
set_option maxRecDepth 4096 in
theorem innerCandidates192_2_checked : candidates edge 4 1187 2879 innerTail192_3 best.bound = innerCandidates192_2 := by
  unfold candidates
  have row : edge 2879 = SearchData.row2879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan192_2 : scan edge (search edge 3) 4 [1187] 1187 innerTail192_2 best = best := by
  rw [innerTail192_2, scan, if_neg (by decide), innerCandidates192_2_checked]
  change scan edge (search edge 3) 4 [1187] 1187 innerTail192_3
    (if 3 ≤ innerCandidates192_2.length then search edge 3 [1187, 2879] innerCandidates192_2 4066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan192_3

def innerTail192_1 : List Nat := 1451 :: innerTail192_2
def innerCandidates192_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates192_1_checked : candidates edge 4 1187 1451 innerTail192_2 best.bound = innerCandidates192_1 := by
  unfold candidates
  have row : edge 1451 = PrimePairCache.smallRow1451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan192_1 : scan edge (search edge 3) 4 [1187] 1187 innerTail192_1 best = best := by
  rw [innerTail192_1, scan, if_neg (by decide), innerCandidates192_1_checked]
  change scan edge (search edge 3) 4 [1187] 1187 innerTail192_2
    (if 3 ≤ innerCandidates192_1.length then search edge 3 [1187, 1451] innerCandidates192_1 2638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan192_2

def innerTail192_0 : List Nat := 1361 :: innerTail192_1
def innerCandidates192_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates192_0_checked : candidates edge 4 1187 1361 innerTail192_1 best.bound = innerCandidates192_0 := by
  unfold candidates
  have row : edge 1361 = PrimePairCache.smallRow1361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan192_0 : scan edge (search edge 3) 4 [1187] 1187 innerTail192_0 best = best := by
  rw [innerTail192_0, scan, if_neg (by decide), innerCandidates192_0_checked]
  change scan edge (search edge 3) 4 [1187] 1187 innerTail192_1
    (if 3 ≤ innerCandidates192_0.length then search edge 3 [1187, 1361] innerCandidates192_0 2548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan192_1

theorem rootBranch192 : search edge 4 [1187] rootCandidates192 1187 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1187] 1187 innerTail192_0 best = best
  exact innerScan192_0

end ProjectEulerSolutions.PrimePairUpper.Search
