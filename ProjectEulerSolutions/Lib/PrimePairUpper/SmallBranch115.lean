import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates338 : List Nat := [2671, 2683, 3727, 3853, 5197, 5281, 5569, 5749, 5827, 6829, 7039, 7333, 8011]
set_option maxRecDepth 4096 in
theorem rootCandidates338_checked : candidates edge 5 0 2293 tail339 best.bound = rootCandidates338 := by
  unfold candidates
  have row : edge 2293 = PrimePairCache.smallRow2293.contains := by funext q; rfl
  rw [row]
  decide

def innerTail338_13 : List Nat := []
theorem innerScan338_13 : scan edge (search edge 3) 4 [2293] 2293 innerTail338_13 best = best := rfl

def innerTail338_12 : List Nat := 8011 :: innerTail338_13
def innerCandidates338_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates338_12_checked : candidates edge 4 2293 8011 innerTail338_13 best.bound = innerCandidates338_12 := by
  unfold candidates
  have row : edge 8011 = SearchData.row8011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan338_12 : scan edge (search edge 3) 4 [2293] 2293 innerTail338_12 best = best := by
  rw [innerTail338_12, scan, if_neg (by decide), innerCandidates338_12_checked]
  change scan edge (search edge 3) 4 [2293] 2293 innerTail338_13
    (if 3 ≤ innerCandidates338_12.length then search edge 3 [2293, 8011] innerCandidates338_12 10304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan338_13

def innerTail338_11 : List Nat := 7333 :: innerTail338_12
def innerCandidates338_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates338_11_checked : candidates edge 4 2293 7333 innerTail338_12 best.bound = innerCandidates338_11 := by
  unfold candidates
  have row : edge 7333 = SearchData.row7333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan338_11 : scan edge (search edge 3) 4 [2293] 2293 innerTail338_11 best = best := by
  rw [innerTail338_11, scan, if_neg (by decide), innerCandidates338_11_checked]
  change scan edge (search edge 3) 4 [2293] 2293 innerTail338_12
    (if 3 ≤ innerCandidates338_11.length then search edge 3 [2293, 7333] innerCandidates338_11 9626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan338_12

def innerTail338_10 : List Nat := 7039 :: innerTail338_11
def innerCandidates338_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates338_10_checked : candidates edge 4 2293 7039 innerTail338_11 best.bound = innerCandidates338_10 := by
  unfold candidates
  have row : edge 7039 = SearchData.row7039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan338_10 : scan edge (search edge 3) 4 [2293] 2293 innerTail338_10 best = best := by
  rw [innerTail338_10, scan, if_neg (by decide), innerCandidates338_10_checked]
  change scan edge (search edge 3) 4 [2293] 2293 innerTail338_11
    (if 3 ≤ innerCandidates338_10.length then search edge 3 [2293, 7039] innerCandidates338_10 9332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan338_11

def innerTail338_9 : List Nat := 6829 :: innerTail338_10
def innerCandidates338_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates338_9_checked : candidates edge 4 2293 6829 innerTail338_10 best.bound = innerCandidates338_9 := by
  unfold candidates
  have row : edge 6829 = SearchData.row6829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan338_9 : scan edge (search edge 3) 4 [2293] 2293 innerTail338_9 best = best := by
  rw [innerTail338_9, scan, if_neg (by decide), innerCandidates338_9_checked]
  change scan edge (search edge 3) 4 [2293] 2293 innerTail338_10
    (if 3 ≤ innerCandidates338_9.length then search edge 3 [2293, 6829] innerCandidates338_9 9122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan338_10

def innerTail338_8 : List Nat := 5827 :: innerTail338_9
def innerCandidates338_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates338_8_checked : candidates edge 4 2293 5827 innerTail338_9 best.bound = innerCandidates338_8 := by
  unfold candidates
  have row : edge 5827 = SearchData.row5827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan338_8 : scan edge (search edge 3) 4 [2293] 2293 innerTail338_8 best = best := by
  rw [innerTail338_8, scan, if_neg (by decide), innerCandidates338_8_checked]
  change scan edge (search edge 3) 4 [2293] 2293 innerTail338_9
    (if 3 ≤ innerCandidates338_8.length then search edge 3 [2293, 5827] innerCandidates338_8 8120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan338_9

def innerTail338_7 : List Nat := 5749 :: innerTail338_8
def innerCandidates338_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates338_7_checked : candidates edge 4 2293 5749 innerTail338_8 best.bound = innerCandidates338_7 := by
  unfold candidates
  have row : edge 5749 = SearchData.row5749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan338_7 : scan edge (search edge 3) 4 [2293] 2293 innerTail338_7 best = best := by
  rw [innerTail338_7, scan, if_neg (by decide), innerCandidates338_7_checked]
  change scan edge (search edge 3) 4 [2293] 2293 innerTail338_8
    (if 3 ≤ innerCandidates338_7.length then search edge 3 [2293, 5749] innerCandidates338_7 8042 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan338_8

def innerTail338_6 : List Nat := 5569 :: innerTail338_7
def innerCandidates338_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates338_6_checked : candidates edge 4 2293 5569 innerTail338_7 best.bound = innerCandidates338_6 := by
  unfold candidates
  have row : edge 5569 = SearchData.row5569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan338_6 : scan edge (search edge 3) 4 [2293] 2293 innerTail338_6 best = best := by
  rw [innerTail338_6, scan, if_neg (by decide), innerCandidates338_6_checked]
  change scan edge (search edge 3) 4 [2293] 2293 innerTail338_7
    (if 3 ≤ innerCandidates338_6.length then search edge 3 [2293, 5569] innerCandidates338_6 7862 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan338_7

def innerTail338_5 : List Nat := 5281 :: innerTail338_6
def innerCandidates338_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates338_5_checked : candidates edge 4 2293 5281 innerTail338_6 best.bound = innerCandidates338_5 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan338_5 : scan edge (search edge 3) 4 [2293] 2293 innerTail338_5 best = best := by
  rw [innerTail338_5, scan, if_neg (by decide), innerCandidates338_5_checked]
  change scan edge (search edge 3) 4 [2293] 2293 innerTail338_6
    (if 3 ≤ innerCandidates338_5.length then search edge 3 [2293, 5281] innerCandidates338_5 7574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan338_6

def innerTail338_4 : List Nat := 5197 :: innerTail338_5
def innerCandidates338_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates338_4_checked : candidates edge 4 2293 5197 innerTail338_5 best.bound = innerCandidates338_4 := by
  unfold candidates
  have row : edge 5197 = SearchData.row5197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan338_4 : scan edge (search edge 3) 4 [2293] 2293 innerTail338_4 best = best := by
  rw [innerTail338_4, scan, if_neg (by decide), innerCandidates338_4_checked]
  change scan edge (search edge 3) 4 [2293] 2293 innerTail338_5
    (if 3 ≤ innerCandidates338_4.length then search edge 3 [2293, 5197] innerCandidates338_4 7490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan338_5

def innerTail338_3 : List Nat := 3853 :: innerTail338_4
def innerCandidates338_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates338_3_checked : candidates edge 4 2293 3853 innerTail338_4 best.bound = innerCandidates338_3 := by
  unfold candidates
  have row : edge 3853 = SearchData.row3853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan338_3 : scan edge (search edge 3) 4 [2293] 2293 innerTail338_3 best = best := by
  rw [innerTail338_3, scan, if_neg (by decide), innerCandidates338_3_checked]
  change scan edge (search edge 3) 4 [2293] 2293 innerTail338_4
    (if 3 ≤ innerCandidates338_3.length then search edge 3 [2293, 3853] innerCandidates338_3 6146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan338_4

def innerTail338_2 : List Nat := 3727 :: innerTail338_3
def innerCandidates338_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates338_2_checked : candidates edge 4 2293 3727 innerTail338_3 best.bound = innerCandidates338_2 := by
  unfold candidates
  have row : edge 3727 = SearchData.row3727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan338_2 : scan edge (search edge 3) 4 [2293] 2293 innerTail338_2 best = best := by
  rw [innerTail338_2, scan, if_neg (by decide), innerCandidates338_2_checked]
  change scan edge (search edge 3) 4 [2293] 2293 innerTail338_3
    (if 3 ≤ innerCandidates338_2.length then search edge 3 [2293, 3727] innerCandidates338_2 6020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan338_3

def innerTail338_1 : List Nat := 2683 :: innerTail338_2
def innerCandidates338_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates338_1_checked : candidates edge 4 2293 2683 innerTail338_2 best.bound = innerCandidates338_1 := by
  unfold candidates
  have row : edge 2683 = SearchData.row2683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan338_1 : scan edge (search edge 3) 4 [2293] 2293 innerTail338_1 best = best := by
  rw [innerTail338_1, scan, if_neg (by decide), innerCandidates338_1_checked]
  change scan edge (search edge 3) 4 [2293] 2293 innerTail338_2
    (if 3 ≤ innerCandidates338_1.length then search edge 3 [2293, 2683] innerCandidates338_1 4976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan338_2

def innerTail338_0 : List Nat := 2671 :: innerTail338_1
def innerCandidates338_0 : List Nat := [3853, 6829]
set_option maxRecDepth 4096 in
theorem innerCandidates338_0_checked : candidates edge 4 2293 2671 innerTail338_1 best.bound = innerCandidates338_0 := by
  unfold candidates
  have row : edge 2671 = SearchData.row2671.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan338_0 : scan edge (search edge 3) 4 [2293] 2293 innerTail338_0 best = best := by
  rw [innerTail338_0, scan, if_neg (by decide), innerCandidates338_0_checked]
  change scan edge (search edge 3) 4 [2293] 2293 innerTail338_1
    (if 3 ≤ innerCandidates338_0.length then search edge 3 [2293, 2671] innerCandidates338_0 4964 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan338_1

theorem rootBranch338 : search edge 4 [2293] rootCandidates338 2293 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2293] 2293 innerTail338_0 best = best
  exact innerScan338_0

def rootCandidates337 : List Nat := [2341, 2683, 3187, 3457, 3607, 3943, 4339, 4483, 4861, 4933, 5059, 5077, 5431, 5857, 6277, 6793, 7537, 7681, 8101, 8221, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates337_checked : candidates edge 5 0 2287 tail338 best.bound = rootCandidates337 := by
  unfold candidates
  have row : edge 2287 = PrimePairCache.smallRow2287.contains := by funext q; rfl
  rw [row]
  decide

def innerTail337_21 : List Nat := []
theorem innerScan337_21 : scan edge (search edge 3) 4 [2287] 2287 innerTail337_21 best = best := rfl

def innerTail337_20 : List Nat := 8377 :: innerTail337_21
def innerCandidates337_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates337_20_checked : candidates edge 4 2287 8377 innerTail337_21 best.bound = innerCandidates337_20 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan337_20 : scan edge (search edge 3) 4 [2287] 2287 innerTail337_20 best = best := by
  rw [innerTail337_20, scan, if_neg (by decide), innerCandidates337_20_checked]
  change scan edge (search edge 3) 4 [2287] 2287 innerTail337_21
    (if 3 ≤ innerCandidates337_20.length then search edge 3 [2287, 8377] innerCandidates337_20 10664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan337_21

def innerTail337_19 : List Nat := 8221 :: innerTail337_20
def innerCandidates337_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates337_19_checked : candidates edge 4 2287 8221 innerTail337_20 best.bound = innerCandidates337_19 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan337_19 : scan edge (search edge 3) 4 [2287] 2287 innerTail337_19 best = best := by
  rw [innerTail337_19, scan, if_neg (by decide), innerCandidates337_19_checked]
  change scan edge (search edge 3) 4 [2287] 2287 innerTail337_20
    (if 3 ≤ innerCandidates337_19.length then search edge 3 [2287, 8221] innerCandidates337_19 10508 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan337_20

def innerTail337_18 : List Nat := 8101 :: innerTail337_19
def innerCandidates337_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates337_18_checked : candidates edge 4 2287 8101 innerTail337_19 best.bound = innerCandidates337_18 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan337_18 : scan edge (search edge 3) 4 [2287] 2287 innerTail337_18 best = best := by
  rw [innerTail337_18, scan, if_neg (by decide), innerCandidates337_18_checked]
  change scan edge (search edge 3) 4 [2287] 2287 innerTail337_19
    (if 3 ≤ innerCandidates337_18.length then search edge 3 [2287, 8101] innerCandidates337_18 10388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan337_19

def innerTail337_17 : List Nat := 7681 :: innerTail337_18
def innerCandidates337_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates337_17_checked : candidates edge 4 2287 7681 innerTail337_18 best.bound = innerCandidates337_17 := by
  unfold candidates
  have row : edge 7681 = SearchData.row7681.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan337_17 : scan edge (search edge 3) 4 [2287] 2287 innerTail337_17 best = best := by
  rw [innerTail337_17, scan, if_neg (by decide), innerCandidates337_17_checked]
  change scan edge (search edge 3) 4 [2287] 2287 innerTail337_18
    (if 3 ≤ innerCandidates337_17.length then search edge 3 [2287, 7681] innerCandidates337_17 9968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan337_18

def innerTail337_16 : List Nat := 7537 :: innerTail337_17
def innerCandidates337_16 : List Nat := [8101]
set_option maxRecDepth 4096 in
theorem innerCandidates337_16_checked : candidates edge 4 2287 7537 innerTail337_17 best.bound = innerCandidates337_16 := by
  unfold candidates
  have row : edge 7537 = SearchData.row7537.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan337_16 : scan edge (search edge 3) 4 [2287] 2287 innerTail337_16 best = best := by
  rw [innerTail337_16, scan, if_neg (by decide), innerCandidates337_16_checked]
  change scan edge (search edge 3) 4 [2287] 2287 innerTail337_17
    (if 3 ≤ innerCandidates337_16.length then search edge 3 [2287, 7537] innerCandidates337_16 9824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan337_17

def innerTail337_15 : List Nat := 6793 :: innerTail337_16
def innerCandidates337_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates337_15_checked : candidates edge 4 2287 6793 innerTail337_16 best.bound = innerCandidates337_15 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan337_15 : scan edge (search edge 3) 4 [2287] 2287 innerTail337_15 best = best := by
  rw [innerTail337_15, scan, if_neg (by decide), innerCandidates337_15_checked]
  change scan edge (search edge 3) 4 [2287] 2287 innerTail337_16
    (if 3 ≤ innerCandidates337_15.length then search edge 3 [2287, 6793] innerCandidates337_15 9080 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan337_16

def innerTail337_14 : List Nat := 6277 :: innerTail337_15
def innerCandidates337_14 : List Nat := [7681]
set_option maxRecDepth 4096 in
theorem innerCandidates337_14_checked : candidates edge 4 2287 6277 innerTail337_15 best.bound = innerCandidates337_14 := by
  unfold candidates
  have row : edge 6277 = SearchData.row6277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan337_14 : scan edge (search edge 3) 4 [2287] 2287 innerTail337_14 best = best := by
  rw [innerTail337_14, scan, if_neg (by decide), innerCandidates337_14_checked]
  change scan edge (search edge 3) 4 [2287] 2287 innerTail337_15
    (if 3 ≤ innerCandidates337_14.length then search edge 3 [2287, 6277] innerCandidates337_14 8564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan337_15

def innerTail337_13 : List Nat := 5857 :: innerTail337_14
def innerCandidates337_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates337_13_checked : candidates edge 4 2287 5857 innerTail337_14 best.bound = innerCandidates337_13 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan337_13 : scan edge (search edge 3) 4 [2287] 2287 innerTail337_13 best = best := by
  rw [innerTail337_13, scan, if_neg (by decide), innerCandidates337_13_checked]
  change scan edge (search edge 3) 4 [2287] 2287 innerTail337_14
    (if 3 ≤ innerCandidates337_13.length then search edge 3 [2287, 5857] innerCandidates337_13 8144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan337_14

def innerTail337_12 : List Nat := 5431 :: innerTail337_13
def innerCandidates337_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates337_12_checked : candidates edge 4 2287 5431 innerTail337_13 best.bound = innerCandidates337_12 := by
  unfold candidates
  have row : edge 5431 = SearchData.row5431.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan337_12 : scan edge (search edge 3) 4 [2287] 2287 innerTail337_12 best = best := by
  rw [innerTail337_12, scan, if_neg (by decide), innerCandidates337_12_checked]
  change scan edge (search edge 3) 4 [2287] 2287 innerTail337_13
    (if 3 ≤ innerCandidates337_12.length then search edge 3 [2287, 5431] innerCandidates337_12 7718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan337_13

def innerTail337_11 : List Nat := 5077 :: innerTail337_12
def innerCandidates337_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates337_11_checked : candidates edge 4 2287 5077 innerTail337_12 best.bound = innerCandidates337_11 := by
  unfold candidates
  have row : edge 5077 = SearchData.row5077.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan337_11 : scan edge (search edge 3) 4 [2287] 2287 innerTail337_11 best = best := by
  rw [innerTail337_11, scan, if_neg (by decide), innerCandidates337_11_checked]
  change scan edge (search edge 3) 4 [2287] 2287 innerTail337_12
    (if 3 ≤ innerCandidates337_11.length then search edge 3 [2287, 5077] innerCandidates337_11 7364 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan337_12

def innerTail337_10 : List Nat := 5059 :: innerTail337_11
def innerCandidates337_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates337_10_checked : candidates edge 4 2287 5059 innerTail337_11 best.bound = innerCandidates337_10 := by
  unfold candidates
  have row : edge 5059 = SearchData.row5059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan337_10 : scan edge (search edge 3) 4 [2287] 2287 innerTail337_10 best = best := by
  rw [innerTail337_10, scan, if_neg (by decide), innerCandidates337_10_checked]
  change scan edge (search edge 3) 4 [2287] 2287 innerTail337_11
    (if 3 ≤ innerCandidates337_10.length then search edge 3 [2287, 5059] innerCandidates337_10 7346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan337_11

def innerTail337_9 : List Nat := 4933 :: innerTail337_10
def innerCandidates337_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates337_9_checked : candidates edge 4 2287 4933 innerTail337_10 best.bound = innerCandidates337_9 := by
  unfold candidates
  have row : edge 4933 = SearchData.row4933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan337_9 : scan edge (search edge 3) 4 [2287] 2287 innerTail337_9 best = best := by
  rw [innerTail337_9, scan, if_neg (by decide), innerCandidates337_9_checked]
  change scan edge (search edge 3) 4 [2287] 2287 innerTail337_10
    (if 3 ≤ innerCandidates337_9.length then search edge 3 [2287, 4933] innerCandidates337_9 7220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan337_10

def innerTail337_8 : List Nat := 4861 :: innerTail337_9
def innerCandidates337_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates337_8_checked : candidates edge 4 2287 4861 innerTail337_9 best.bound = innerCandidates337_8 := by
  unfold candidates
  have row : edge 4861 = SearchData.row4861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan337_8 : scan edge (search edge 3) 4 [2287] 2287 innerTail337_8 best = best := by
  rw [innerTail337_8, scan, if_neg (by decide), innerCandidates337_8_checked]
  change scan edge (search edge 3) 4 [2287] 2287 innerTail337_9
    (if 3 ≤ innerCandidates337_8.length then search edge 3 [2287, 4861] innerCandidates337_8 7148 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan337_9

def innerTail337_7 : List Nat := 4483 :: innerTail337_8
def innerCandidates337_7 : List Nat := [6793, 7537]
set_option maxRecDepth 4096 in
theorem innerCandidates337_7_checked : candidates edge 4 2287 4483 innerTail337_8 best.bound = innerCandidates337_7 := by
  unfold candidates
  have row : edge 4483 = SearchData.row4483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan337_7 : scan edge (search edge 3) 4 [2287] 2287 innerTail337_7 best = best := by
  rw [innerTail337_7, scan, if_neg (by decide), innerCandidates337_7_checked]
  change scan edge (search edge 3) 4 [2287] 2287 innerTail337_8
    (if 3 ≤ innerCandidates337_7.length then search edge 3 [2287, 4483] innerCandidates337_7 6770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan337_8

def innerTail337_6 : List Nat := 4339 :: innerTail337_7
def innerCandidates337_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates337_6_checked : candidates edge 4 2287 4339 innerTail337_7 best.bound = innerCandidates337_6 := by
  unfold candidates
  have row : edge 4339 = SearchData.row4339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan337_6 : scan edge (search edge 3) 4 [2287] 2287 innerTail337_6 best = best := by
  rw [innerTail337_6, scan, if_neg (by decide), innerCandidates337_6_checked]
  change scan edge (search edge 3) 4 [2287] 2287 innerTail337_7
    (if 3 ≤ innerCandidates337_6.length then search edge 3 [2287, 4339] innerCandidates337_6 6626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan337_7

def innerTail337_5 : List Nat := 3943 :: innerTail337_6
def innerCandidates337_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates337_5_checked : candidates edge 4 2287 3943 innerTail337_6 best.bound = innerCandidates337_5 := by
  unfold candidates
  have row : edge 3943 = SearchData.row3943.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan337_5 : scan edge (search edge 3) 4 [2287] 2287 innerTail337_5 best = best := by
  rw [innerTail337_5, scan, if_neg (by decide), innerCandidates337_5_checked]
  change scan edge (search edge 3) 4 [2287] 2287 innerTail337_6
    (if 3 ≤ innerCandidates337_5.length then search edge 3 [2287, 3943] innerCandidates337_5 6230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan337_6

def innerTail337_4 : List Nat := 3607 :: innerTail337_5
def innerCandidates337_4 : List Nat := [5077, 8101]
set_option maxRecDepth 4096 in
theorem innerCandidates337_4_checked : candidates edge 4 2287 3607 innerTail337_5 best.bound = innerCandidates337_4 := by
  unfold candidates
  have row : edge 3607 = SearchData.row3607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan337_4 : scan edge (search edge 3) 4 [2287] 2287 innerTail337_4 best = best := by
  rw [innerTail337_4, scan, if_neg (by decide), innerCandidates337_4_checked]
  change scan edge (search edge 3) 4 [2287] 2287 innerTail337_5
    (if 3 ≤ innerCandidates337_4.length then search edge 3 [2287, 3607] innerCandidates337_4 5894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan337_5

def innerTail337_3 : List Nat := 3457 :: innerTail337_4
def innerCandidates337_3 : List Nat := [8377]
set_option maxRecDepth 4096 in
theorem innerCandidates337_3_checked : candidates edge 4 2287 3457 innerTail337_4 best.bound = innerCandidates337_3 := by
  unfold candidates
  have row : edge 3457 = SearchData.row3457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan337_3 : scan edge (search edge 3) 4 [2287] 2287 innerTail337_3 best = best := by
  rw [innerTail337_3, scan, if_neg (by decide), innerCandidates337_3_checked]
  change scan edge (search edge 3) 4 [2287] 2287 innerTail337_4
    (if 3 ≤ innerCandidates337_3.length then search edge 3 [2287, 3457] innerCandidates337_3 5744 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan337_4

def innerTail337_2 : List Nat := 3187 :: innerTail337_3
def innerCandidates337_2 : List Nat := [8101]
set_option maxRecDepth 4096 in
theorem innerCandidates337_2_checked : candidates edge 4 2287 3187 innerTail337_3 best.bound = innerCandidates337_2 := by
  unfold candidates
  have row : edge 3187 = SearchData.row3187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan337_2 : scan edge (search edge 3) 4 [2287] 2287 innerTail337_2 best = best := by
  rw [innerTail337_2, scan, if_neg (by decide), innerCandidates337_2_checked]
  change scan edge (search edge 3) 4 [2287] 2287 innerTail337_3
    (if 3 ≤ innerCandidates337_2.length then search edge 3 [2287, 3187] innerCandidates337_2 5474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan337_3

def innerTail337_1 : List Nat := 2683 :: innerTail337_2
def innerCandidates337_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates337_1_checked : candidates edge 4 2287 2683 innerTail337_2 best.bound = innerCandidates337_1 := by
  unfold candidates
  have row : edge 2683 = SearchData.row2683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan337_1 : scan edge (search edge 3) 4 [2287] 2287 innerTail337_1 best = best := by
  rw [innerTail337_1, scan, if_neg (by decide), innerCandidates337_1_checked]
  change scan edge (search edge 3) 4 [2287] 2287 innerTail337_2
    (if 3 ≤ innerCandidates337_1.length then search edge 3 [2287, 2683] innerCandidates337_1 4970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan337_2

def innerTail337_0 : List Nat := 2341 :: innerTail337_1
def innerCandidates337_0 : List Nat := [8377]
set_option maxRecDepth 4096 in
theorem innerCandidates337_0_checked : candidates edge 4 2287 2341 innerTail337_1 best.bound = innerCandidates337_0 := by
  unfold candidates
  have row : edge 2341 = PrimePairCache.smallRow2341.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan337_0 : scan edge (search edge 3) 4 [2287] 2287 innerTail337_0 best = best := by
  rw [innerTail337_0, scan, if_neg (by decide), innerCandidates337_0_checked]
  change scan edge (search edge 3) 4 [2287] 2287 innerTail337_1
    (if 3 ≤ innerCandidates337_0.length then search edge 3 [2287, 2341] innerCandidates337_0 4628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan337_1

theorem rootBranch337 : search edge 4 [2287] rootCandidates337 2287 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2287] 2287 innerTail337_0 best = best
  exact innerScan337_0

def rootCandidates336 : List Nat := [2551, 2731, 2887, 4201, 4273, 4591, 5857, 6361, 6373, 6481, 6793, 7351, 8179]
set_option maxRecDepth 4096 in
theorem rootCandidates336_checked : candidates edge 5 0 2281 tail337 best.bound = rootCandidates336 := by
  unfold candidates
  have row : edge 2281 = PrimePairCache.smallRow2281.contains := by funext q; rfl
  rw [row]
  decide

def innerTail336_13 : List Nat := []
theorem innerScan336_13 : scan edge (search edge 3) 4 [2281] 2281 innerTail336_13 best = best := rfl

def innerTail336_12 : List Nat := 8179 :: innerTail336_13
def innerCandidates336_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates336_12_checked : candidates edge 4 2281 8179 innerTail336_13 best.bound = innerCandidates336_12 := by
  unfold candidates
  have row : edge 8179 = SearchData.row8179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan336_12 : scan edge (search edge 3) 4 [2281] 2281 innerTail336_12 best = best := by
  rw [innerTail336_12, scan, if_neg (by decide), innerCandidates336_12_checked]
  change scan edge (search edge 3) 4 [2281] 2281 innerTail336_13
    (if 3 ≤ innerCandidates336_12.length then search edge 3 [2281, 8179] innerCandidates336_12 10460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan336_13

def innerTail336_11 : List Nat := 7351 :: innerTail336_12
def innerCandidates336_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates336_11_checked : candidates edge 4 2281 7351 innerTail336_12 best.bound = innerCandidates336_11 := by
  unfold candidates
  have row : edge 7351 = SearchData.row7351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan336_11 : scan edge (search edge 3) 4 [2281] 2281 innerTail336_11 best = best := by
  rw [innerTail336_11, scan, if_neg (by decide), innerCandidates336_11_checked]
  change scan edge (search edge 3) 4 [2281] 2281 innerTail336_12
    (if 3 ≤ innerCandidates336_11.length then search edge 3 [2281, 7351] innerCandidates336_11 9632 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan336_12

def innerTail336_10 : List Nat := 6793 :: innerTail336_11
def innerCandidates336_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates336_10_checked : candidates edge 4 2281 6793 innerTail336_11 best.bound = innerCandidates336_10 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan336_10 : scan edge (search edge 3) 4 [2281] 2281 innerTail336_10 best = best := by
  rw [innerTail336_10, scan, if_neg (by decide), innerCandidates336_10_checked]
  change scan edge (search edge 3) 4 [2281] 2281 innerTail336_11
    (if 3 ≤ innerCandidates336_10.length then search edge 3 [2281, 6793] innerCandidates336_10 9074 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan336_11

def innerTail336_9 : List Nat := 6481 :: innerTail336_10
def innerCandidates336_9 : List Nat := [6793]
set_option maxRecDepth 4096 in
theorem innerCandidates336_9_checked : candidates edge 4 2281 6481 innerTail336_10 best.bound = innerCandidates336_9 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan336_9 : scan edge (search edge 3) 4 [2281] 2281 innerTail336_9 best = best := by
  rw [innerTail336_9, scan, if_neg (by decide), innerCandidates336_9_checked]
  change scan edge (search edge 3) 4 [2281] 2281 innerTail336_10
    (if 3 ≤ innerCandidates336_9.length then search edge 3 [2281, 6481] innerCandidates336_9 8762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan336_10

def innerTail336_8 : List Nat := 6373 :: innerTail336_9
def innerCandidates336_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates336_8_checked : candidates edge 4 2281 6373 innerTail336_9 best.bound = innerCandidates336_8 := by
  unfold candidates
  have row : edge 6373 = SearchData.row6373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan336_8 : scan edge (search edge 3) 4 [2281] 2281 innerTail336_8 best = best := by
  rw [innerTail336_8, scan, if_neg (by decide), innerCandidates336_8_checked]
  change scan edge (search edge 3) 4 [2281] 2281 innerTail336_9
    (if 3 ≤ innerCandidates336_8.length then search edge 3 [2281, 6373] innerCandidates336_8 8654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan336_9

def innerTail336_7 : List Nat := 6361 :: innerTail336_8
def innerCandidates336_7 : List Nat := [6481]
set_option maxRecDepth 4096 in
theorem innerCandidates336_7_checked : candidates edge 4 2281 6361 innerTail336_8 best.bound = innerCandidates336_7 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan336_7 : scan edge (search edge 3) 4 [2281] 2281 innerTail336_7 best = best := by
  rw [innerTail336_7, scan, if_neg (by decide), innerCandidates336_7_checked]
  change scan edge (search edge 3) 4 [2281] 2281 innerTail336_8
    (if 3 ≤ innerCandidates336_7.length then search edge 3 [2281, 6361] innerCandidates336_7 8642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan336_8

def innerTail336_6 : List Nat := 5857 :: innerTail336_7
def innerCandidates336_6 : List Nat := [6481]
set_option maxRecDepth 4096 in
theorem innerCandidates336_6_checked : candidates edge 4 2281 5857 innerTail336_7 best.bound = innerCandidates336_6 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan336_6 : scan edge (search edge 3) 4 [2281] 2281 innerTail336_6 best = best := by
  rw [innerTail336_6, scan, if_neg (by decide), innerCandidates336_6_checked]
  change scan edge (search edge 3) 4 [2281] 2281 innerTail336_7
    (if 3 ≤ innerCandidates336_6.length then search edge 3 [2281, 5857] innerCandidates336_6 8138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan336_7

def innerTail336_5 : List Nat := 4591 :: innerTail336_6
def innerCandidates336_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates336_5_checked : candidates edge 4 2281 4591 innerTail336_6 best.bound = innerCandidates336_5 := by
  unfold candidates
  have row : edge 4591 = SearchData.row4591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan336_5 : scan edge (search edge 3) 4 [2281] 2281 innerTail336_5 best = best := by
  rw [innerTail336_5, scan, if_neg (by decide), innerCandidates336_5_checked]
  change scan edge (search edge 3) 4 [2281] 2281 innerTail336_6
    (if 3 ≤ innerCandidates336_5.length then search edge 3 [2281, 4591] innerCandidates336_5 6872 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan336_6

def innerTail336_4 : List Nat := 4273 :: innerTail336_5
def innerCandidates336_4 : List Nat := [6373]
set_option maxRecDepth 4096 in
theorem innerCandidates336_4_checked : candidates edge 4 2281 4273 innerTail336_5 best.bound = innerCandidates336_4 := by
  unfold candidates
  have row : edge 4273 = SearchData.row4273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan336_4 : scan edge (search edge 3) 4 [2281] 2281 innerTail336_4 best = best := by
  rw [innerTail336_4, scan, if_neg (by decide), innerCandidates336_4_checked]
  change scan edge (search edge 3) 4 [2281] 2281 innerTail336_5
    (if 3 ≤ innerCandidates336_4.length then search edge 3 [2281, 4273] innerCandidates336_4 6554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan336_5

def innerTail336_3 : List Nat := 4201 :: innerTail336_4
def innerCandidates336_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates336_3_checked : candidates edge 4 2281 4201 innerTail336_4 best.bound = innerCandidates336_3 := by
  unfold candidates
  have row : edge 4201 = SearchData.row4201.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan336_3 : scan edge (search edge 3) 4 [2281] 2281 innerTail336_3 best = best := by
  rw [innerTail336_3, scan, if_neg (by decide), innerCandidates336_3_checked]
  change scan edge (search edge 3) 4 [2281] 2281 innerTail336_4
    (if 3 ≤ innerCandidates336_3.length then search edge 3 [2281, 4201] innerCandidates336_3 6482 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan336_4

def innerTail336_2 : List Nat := 2887 :: innerTail336_3
def innerCandidates336_2 : List Nat := [4273]
set_option maxRecDepth 4096 in
theorem innerCandidates336_2_checked : candidates edge 4 2281 2887 innerTail336_3 best.bound = innerCandidates336_2 := by
  unfold candidates
  have row : edge 2887 = SearchData.row2887.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan336_2 : scan edge (search edge 3) 4 [2281] 2281 innerTail336_2 best = best := by
  rw [innerTail336_2, scan, if_neg (by decide), innerCandidates336_2_checked]
  change scan edge (search edge 3) 4 [2281] 2281 innerTail336_3
    (if 3 ≤ innerCandidates336_2.length then search edge 3 [2281, 2887] innerCandidates336_2 5168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan336_3

def innerTail336_1 : List Nat := 2731 :: innerTail336_2
def innerCandidates336_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates336_1_checked : candidates edge 4 2281 2731 innerTail336_2 best.bound = innerCandidates336_1 := by
  unfold candidates
  have row : edge 2731 = SearchData.row2731.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan336_1 : scan edge (search edge 3) 4 [2281] 2281 innerTail336_1 best = best := by
  rw [innerTail336_1, scan, if_neg (by decide), innerCandidates336_1_checked]
  change scan edge (search edge 3) 4 [2281] 2281 innerTail336_2
    (if 3 ≤ innerCandidates336_1.length then search edge 3 [2281, 2731] innerCandidates336_1 5012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan336_2

def innerTail336_0 : List Nat := 2551 :: innerTail336_1
def innerCandidates336_0 : List Nat := [2887, 4591, 6793]
set_option maxRecDepth 4096 in
theorem innerCandidates336_0_checked : candidates edge 4 2281 2551 innerTail336_1 best.bound = innerCandidates336_0 := by
  unfold candidates
  have row : edge 2551 = PrimePairCache.smallRow2551.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild336_0 : search edge 3 [2281, 2551] innerCandidates336_0 4832 best = best := by decide

theorem innerScan336_0 : scan edge (search edge 3) 4 [2281] 2281 innerTail336_0 best = best := by
  rw [innerTail336_0, scan, if_neg (by decide), innerCandidates336_0_checked]
  change scan edge (search edge 3) 4 [2281] 2281 innerTail336_1
    (if 3 ≤ innerCandidates336_0.length then search edge 3 [2281, 2551] innerCandidates336_0 4832 best else best) = best
  rw [if_pos (by decide), innerChild336_0]
  exact innerScan336_1

theorem rootBranch336 : search edge 4 [2281] rootCandidates336 2281 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2281] 2281 innerTail336_0 best = best
  exact innerScan336_0

def rootCandidates335 : List Nat := [2609, 2789, 2903, 2999, 3041, 3461, 3761, 4973, 5399, 5711, 5861, 5897, 7283, 8147, 8297]
set_option maxRecDepth 4096 in
theorem rootCandidates335_checked : candidates edge 5 0 2273 tail336 best.bound = rootCandidates335 := by
  unfold candidates
  have row : edge 2273 = PrimePairCache.smallRow2273.contains := by funext q; rfl
  rw [row]
  decide

def innerTail335_15 : List Nat := []
theorem innerScan335_15 : scan edge (search edge 3) 4 [2273] 2273 innerTail335_15 best = best := rfl

def innerTail335_14 : List Nat := 8297 :: innerTail335_15
def innerCandidates335_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates335_14_checked : candidates edge 4 2273 8297 innerTail335_15 best.bound = innerCandidates335_14 := by
  unfold candidates
  have row : edge 8297 = SearchData.row8297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan335_14 : scan edge (search edge 3) 4 [2273] 2273 innerTail335_14 best = best := by
  rw [innerTail335_14, scan, if_neg (by decide), innerCandidates335_14_checked]
  change scan edge (search edge 3) 4 [2273] 2273 innerTail335_15
    (if 3 ≤ innerCandidates335_14.length then search edge 3 [2273, 8297] innerCandidates335_14 10570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan335_15

def innerTail335_13 : List Nat := 8147 :: innerTail335_14
def innerCandidates335_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates335_13_checked : candidates edge 4 2273 8147 innerTail335_14 best.bound = innerCandidates335_13 := by
  unfold candidates
  have row : edge 8147 = SearchData.row8147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan335_13 : scan edge (search edge 3) 4 [2273] 2273 innerTail335_13 best = best := by
  rw [innerTail335_13, scan, if_neg (by decide), innerCandidates335_13_checked]
  change scan edge (search edge 3) 4 [2273] 2273 innerTail335_14
    (if 3 ≤ innerCandidates335_13.length then search edge 3 [2273, 8147] innerCandidates335_13 10420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan335_14

def innerTail335_12 : List Nat := 7283 :: innerTail335_13
def innerCandidates335_12 : List Nat := [8147]
set_option maxRecDepth 4096 in
theorem innerCandidates335_12_checked : candidates edge 4 2273 7283 innerTail335_13 best.bound = innerCandidates335_12 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan335_12 : scan edge (search edge 3) 4 [2273] 2273 innerTail335_12 best = best := by
  rw [innerTail335_12, scan, if_neg (by decide), innerCandidates335_12_checked]
  change scan edge (search edge 3) 4 [2273] 2273 innerTail335_13
    (if 3 ≤ innerCandidates335_12.length then search edge 3 [2273, 7283] innerCandidates335_12 9556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan335_13

def innerTail335_11 : List Nat := 5897 :: innerTail335_12
def innerCandidates335_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates335_11_checked : candidates edge 4 2273 5897 innerTail335_12 best.bound = innerCandidates335_11 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan335_11 : scan edge (search edge 3) 4 [2273] 2273 innerTail335_11 best = best := by
  rw [innerTail335_11, scan, if_neg (by decide), innerCandidates335_11_checked]
  change scan edge (search edge 3) 4 [2273] 2273 innerTail335_12
    (if 3 ≤ innerCandidates335_11.length then search edge 3 [2273, 5897] innerCandidates335_11 8170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan335_12

def innerTail335_10 : List Nat := 5861 :: innerTail335_11
def innerCandidates335_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates335_10_checked : candidates edge 4 2273 5861 innerTail335_11 best.bound = innerCandidates335_10 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan335_10 : scan edge (search edge 3) 4 [2273] 2273 innerTail335_10 best = best := by
  rw [innerTail335_10, scan, if_neg (by decide), innerCandidates335_10_checked]
  change scan edge (search edge 3) 4 [2273] 2273 innerTail335_11
    (if 3 ≤ innerCandidates335_10.length then search edge 3 [2273, 5861] innerCandidates335_10 8134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan335_11

def innerTail335_9 : List Nat := 5711 :: innerTail335_10
def innerCandidates335_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates335_9_checked : candidates edge 4 2273 5711 innerTail335_10 best.bound = innerCandidates335_9 := by
  unfold candidates
  have row : edge 5711 = SearchData.row5711.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan335_9 : scan edge (search edge 3) 4 [2273] 2273 innerTail335_9 best = best := by
  rw [innerTail335_9, scan, if_neg (by decide), innerCandidates335_9_checked]
  change scan edge (search edge 3) 4 [2273] 2273 innerTail335_10
    (if 3 ≤ innerCandidates335_9.length then search edge 3 [2273, 5711] innerCandidates335_9 7984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan335_10

def innerTail335_8 : List Nat := 5399 :: innerTail335_9
def innerCandidates335_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates335_8_checked : candidates edge 4 2273 5399 innerTail335_9 best.bound = innerCandidates335_8 := by
  unfold candidates
  have row : edge 5399 = SearchData.row5399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan335_8 : scan edge (search edge 3) 4 [2273] 2273 innerTail335_8 best = best := by
  rw [innerTail335_8, scan, if_neg (by decide), innerCandidates335_8_checked]
  change scan edge (search edge 3) 4 [2273] 2273 innerTail335_9
    (if 3 ≤ innerCandidates335_8.length then search edge 3 [2273, 5399] innerCandidates335_8 7672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan335_9

def innerTail335_7 : List Nat := 4973 :: innerTail335_8
def innerCandidates335_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates335_7_checked : candidates edge 4 2273 4973 innerTail335_8 best.bound = innerCandidates335_7 := by
  unfold candidates
  have row : edge 4973 = SearchData.row4973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan335_7 : scan edge (search edge 3) 4 [2273] 2273 innerTail335_7 best = best := by
  rw [innerTail335_7, scan, if_neg (by decide), innerCandidates335_7_checked]
  change scan edge (search edge 3) 4 [2273] 2273 innerTail335_8
    (if 3 ≤ innerCandidates335_7.length then search edge 3 [2273, 4973] innerCandidates335_7 7246 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan335_8

def innerTail335_6 : List Nat := 3761 :: innerTail335_7
def innerCandidates335_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates335_6_checked : candidates edge 4 2273 3761 innerTail335_7 best.bound = innerCandidates335_6 := by
  unfold candidates
  have row : edge 3761 = SearchData.row3761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan335_6 : scan edge (search edge 3) 4 [2273] 2273 innerTail335_6 best = best := by
  rw [innerTail335_6, scan, if_neg (by decide), innerCandidates335_6_checked]
  change scan edge (search edge 3) 4 [2273] 2273 innerTail335_7
    (if 3 ≤ innerCandidates335_6.length then search edge 3 [2273, 3761] innerCandidates335_6 6034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan335_7

def innerTail335_5 : List Nat := 3461 :: innerTail335_6
def innerCandidates335_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates335_5_checked : candidates edge 4 2273 3461 innerTail335_6 best.bound = innerCandidates335_5 := by
  unfold candidates
  have row : edge 3461 = SearchData.row3461.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan335_5 : scan edge (search edge 3) 4 [2273] 2273 innerTail335_5 best = best := by
  rw [innerTail335_5, scan, if_neg (by decide), innerCandidates335_5_checked]
  change scan edge (search edge 3) 4 [2273] 2273 innerTail335_6
    (if 3 ≤ innerCandidates335_5.length then search edge 3 [2273, 3461] innerCandidates335_5 5734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan335_6

def innerTail335_4 : List Nat := 3041 :: innerTail335_5
def innerCandidates335_4 : List Nat := [5897]
set_option maxRecDepth 4096 in
theorem innerCandidates335_4_checked : candidates edge 4 2273 3041 innerTail335_5 best.bound = innerCandidates335_4 := by
  unfold candidates
  have row : edge 3041 = SearchData.row3041.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan335_4 : scan edge (search edge 3) 4 [2273] 2273 innerTail335_4 best = best := by
  rw [innerTail335_4, scan, if_neg (by decide), innerCandidates335_4_checked]
  change scan edge (search edge 3) 4 [2273] 2273 innerTail335_5
    (if 3 ≤ innerCandidates335_4.length then search edge 3 [2273, 3041] innerCandidates335_4 5314 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan335_5

def innerTail335_3 : List Nat := 2999 :: innerTail335_4
def innerCandidates335_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates335_3_checked : candidates edge 4 2273 2999 innerTail335_4 best.bound = innerCandidates335_3 := by
  unfold candidates
  have row : edge 2999 = SearchData.row2999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan335_3 : scan edge (search edge 3) 4 [2273] 2273 innerTail335_3 best = best := by
  rw [innerTail335_3, scan, if_neg (by decide), innerCandidates335_3_checked]
  change scan edge (search edge 3) 4 [2273] 2273 innerTail335_4
    (if 3 ≤ innerCandidates335_3.length then search edge 3 [2273, 2999] innerCandidates335_3 5272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan335_4

def innerTail335_2 : List Nat := 2903 :: innerTail335_3
def innerCandidates335_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates335_2_checked : candidates edge 4 2273 2903 innerTail335_3 best.bound = innerCandidates335_2 := by
  unfold candidates
  have row : edge 2903 = SearchData.row2903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan335_2 : scan edge (search edge 3) 4 [2273] 2273 innerTail335_2 best = best := by
  rw [innerTail335_2, scan, if_neg (by decide), innerCandidates335_2_checked]
  change scan edge (search edge 3) 4 [2273] 2273 innerTail335_3
    (if 3 ≤ innerCandidates335_2.length then search edge 3 [2273, 2903] innerCandidates335_2 5176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan335_3

def innerTail335_1 : List Nat := 2789 :: innerTail335_2
def innerCandidates335_1 : List Nat := [3461]
set_option maxRecDepth 4096 in
theorem innerCandidates335_1_checked : candidates edge 4 2273 2789 innerTail335_2 best.bound = innerCandidates335_1 := by
  unfold candidates
  have row : edge 2789 = SearchData.row2789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan335_1 : scan edge (search edge 3) 4 [2273] 2273 innerTail335_1 best = best := by
  rw [innerTail335_1, scan, if_neg (by decide), innerCandidates335_1_checked]
  change scan edge (search edge 3) 4 [2273] 2273 innerTail335_2
    (if 3 ≤ innerCandidates335_1.length then search edge 3 [2273, 2789] innerCandidates335_1 5062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan335_2

def innerTail335_0 : List Nat := 2609 :: innerTail335_1
def innerCandidates335_0 : List Nat := [4973, 8297]
set_option maxRecDepth 4096 in
theorem innerCandidates335_0_checked : candidates edge 4 2273 2609 innerTail335_1 best.bound = innerCandidates335_0 := by
  unfold candidates
  have row : edge 2609 = PrimePairCache.smallRow2609.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan335_0 : scan edge (search edge 3) 4 [2273] 2273 innerTail335_0 best = best := by
  rw [innerTail335_0, scan, if_neg (by decide), innerCandidates335_0_checked]
  change scan edge (search edge 3) 4 [2273] 2273 innerTail335_1
    (if 3 ≤ innerCandidates335_0.length then search edge 3 [2273, 2609] innerCandidates335_0 4882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan335_1

theorem rootBranch335 : search edge 4 [2273] rootCandidates335 2273 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2273] 2273 innerTail335_0 best = best
  exact innerScan335_0

end ProjectEulerSolutions.PrimePairUpper.Search
