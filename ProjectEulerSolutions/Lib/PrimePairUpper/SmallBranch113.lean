import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates345 : List Nat := [2731, 2749, 2833, 2851, 3049, 3067, 3253, 4327, 4639, 4663, 6199, 6469, 6883, 7057, 7417, 7681, 7873, 7879]
set_option maxRecDepth 4096 in
theorem rootCandidates345_checked : candidates edge 5 0 2347 tail346 best.bound = rootCandidates345 := by
  unfold candidates
  have row : edge 2347 = PrimePairCache.smallRow2347.contains := by funext q; rfl
  rw [row]
  decide

def innerTail345_18 : List Nat := []
theorem innerScan345_18 : scan edge (search edge 3) 4 [2347] 2347 innerTail345_18 best = best := rfl

def innerTail345_17 : List Nat := 7879 :: innerTail345_18
def innerCandidates345_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates345_17_checked : candidates edge 4 2347 7879 innerTail345_18 best.bound = innerCandidates345_17 := by
  unfold candidates
  have row : edge 7879 = SearchData.row7879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan345_17 : scan edge (search edge 3) 4 [2347] 2347 innerTail345_17 best = best := by
  rw [innerTail345_17, scan, if_neg (by decide), innerCandidates345_17_checked]
  change scan edge (search edge 3) 4 [2347] 2347 innerTail345_18
    (if 3 ≤ innerCandidates345_17.length then search edge 3 [2347, 7879] innerCandidates345_17 10226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan345_18

def innerTail345_16 : List Nat := 7873 :: innerTail345_17
def innerCandidates345_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates345_16_checked : candidates edge 4 2347 7873 innerTail345_17 best.bound = innerCandidates345_16 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan345_16 : scan edge (search edge 3) 4 [2347] 2347 innerTail345_16 best = best := by
  rw [innerTail345_16, scan, if_neg (by decide), innerCandidates345_16_checked]
  change scan edge (search edge 3) 4 [2347] 2347 innerTail345_17
    (if 3 ≤ innerCandidates345_16.length then search edge 3 [2347, 7873] innerCandidates345_16 10220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan345_17

def innerTail345_15 : List Nat := 7681 :: innerTail345_16
def innerCandidates345_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates345_15_checked : candidates edge 4 2347 7681 innerTail345_16 best.bound = innerCandidates345_15 := by
  unfold candidates
  have row : edge 7681 = SearchData.row7681.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan345_15 : scan edge (search edge 3) 4 [2347] 2347 innerTail345_15 best = best := by
  rw [innerTail345_15, scan, if_neg (by decide), innerCandidates345_15_checked]
  change scan edge (search edge 3) 4 [2347] 2347 innerTail345_16
    (if 3 ≤ innerCandidates345_15.length then search edge 3 [2347, 7681] innerCandidates345_15 10028 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan345_16

def innerTail345_14 : List Nat := 7417 :: innerTail345_15
def innerCandidates345_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates345_14_checked : candidates edge 4 2347 7417 innerTail345_15 best.bound = innerCandidates345_14 := by
  unfold candidates
  have row : edge 7417 = SearchData.row7417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan345_14 : scan edge (search edge 3) 4 [2347] 2347 innerTail345_14 best = best := by
  rw [innerTail345_14, scan, if_neg (by decide), innerCandidates345_14_checked]
  change scan edge (search edge 3) 4 [2347] 2347 innerTail345_15
    (if 3 ≤ innerCandidates345_14.length then search edge 3 [2347, 7417] innerCandidates345_14 9764 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan345_15

def innerTail345_13 : List Nat := 7057 :: innerTail345_14
def innerCandidates345_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates345_13_checked : candidates edge 4 2347 7057 innerTail345_14 best.bound = innerCandidates345_13 := by
  unfold candidates
  have row : edge 7057 = SearchData.row7057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan345_13 : scan edge (search edge 3) 4 [2347] 2347 innerTail345_13 best = best := by
  rw [innerTail345_13, scan, if_neg (by decide), innerCandidates345_13_checked]
  change scan edge (search edge 3) 4 [2347] 2347 innerTail345_14
    (if 3 ≤ innerCandidates345_13.length then search edge 3 [2347, 7057] innerCandidates345_13 9404 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan345_14

def innerTail345_12 : List Nat := 6883 :: innerTail345_13
def innerCandidates345_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates345_12_checked : candidates edge 4 2347 6883 innerTail345_13 best.bound = innerCandidates345_12 := by
  unfold candidates
  have row : edge 6883 = SearchData.row6883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan345_12 : scan edge (search edge 3) 4 [2347] 2347 innerTail345_12 best = best := by
  rw [innerTail345_12, scan, if_neg (by decide), innerCandidates345_12_checked]
  change scan edge (search edge 3) 4 [2347] 2347 innerTail345_13
    (if 3 ≤ innerCandidates345_12.length then search edge 3 [2347, 6883] innerCandidates345_12 9230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan345_13

def innerTail345_11 : List Nat := 6469 :: innerTail345_12
def innerCandidates345_11 : List Nat := [7681, 7879]
set_option maxRecDepth 4096 in
theorem innerCandidates345_11_checked : candidates edge 4 2347 6469 innerTail345_12 best.bound = innerCandidates345_11 := by
  unfold candidates
  have row : edge 6469 = SearchData.row6469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan345_11 : scan edge (search edge 3) 4 [2347] 2347 innerTail345_11 best = best := by
  rw [innerTail345_11, scan, if_neg (by decide), innerCandidates345_11_checked]
  change scan edge (search edge 3) 4 [2347] 2347 innerTail345_12
    (if 3 ≤ innerCandidates345_11.length then search edge 3 [2347, 6469] innerCandidates345_11 8816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan345_12

def innerTail345_10 : List Nat := 6199 :: innerTail345_11
def innerCandidates345_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates345_10_checked : candidates edge 4 2347 6199 innerTail345_11 best.bound = innerCandidates345_10 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan345_10 : scan edge (search edge 3) 4 [2347] 2347 innerTail345_10 best = best := by
  rw [innerTail345_10, scan, if_neg (by decide), innerCandidates345_10_checked]
  change scan edge (search edge 3) 4 [2347] 2347 innerTail345_11
    (if 3 ≤ innerCandidates345_10.length then search edge 3 [2347, 6199] innerCandidates345_10 8546 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan345_11

def innerTail345_9 : List Nat := 4663 :: innerTail345_10
def innerCandidates345_9 : List Nat := [7879]
set_option maxRecDepth 4096 in
theorem innerCandidates345_9_checked : candidates edge 4 2347 4663 innerTail345_10 best.bound = innerCandidates345_9 := by
  unfold candidates
  have row : edge 4663 = SearchData.row4663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan345_9 : scan edge (search edge 3) 4 [2347] 2347 innerTail345_9 best = best := by
  rw [innerTail345_9, scan, if_neg (by decide), innerCandidates345_9_checked]
  change scan edge (search edge 3) 4 [2347] 2347 innerTail345_10
    (if 3 ≤ innerCandidates345_9.length then search edge 3 [2347, 4663] innerCandidates345_9 7010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan345_10

def innerTail345_8 : List Nat := 4639 :: innerTail345_9
def innerCandidates345_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates345_8_checked : candidates edge 4 2347 4639 innerTail345_9 best.bound = innerCandidates345_8 := by
  unfold candidates
  have row : edge 4639 = SearchData.row4639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan345_8 : scan edge (search edge 3) 4 [2347] 2347 innerTail345_8 best = best := by
  rw [innerTail345_8, scan, if_neg (by decide), innerCandidates345_8_checked]
  change scan edge (search edge 3) 4 [2347] 2347 innerTail345_9
    (if 3 ≤ innerCandidates345_8.length then search edge 3 [2347, 4639] innerCandidates345_8 6986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan345_9

def innerTail345_7 : List Nat := 4327 :: innerTail345_8
def innerCandidates345_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates345_7_checked : candidates edge 4 2347 4327 innerTail345_8 best.bound = innerCandidates345_7 := by
  unfold candidates
  have row : edge 4327 = SearchData.row4327.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan345_7 : scan edge (search edge 3) 4 [2347] 2347 innerTail345_7 best = best := by
  rw [innerTail345_7, scan, if_neg (by decide), innerCandidates345_7_checked]
  change scan edge (search edge 3) 4 [2347] 2347 innerTail345_8
    (if 3 ≤ innerCandidates345_7.length then search edge 3 [2347, 4327] innerCandidates345_7 6674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan345_8

def innerTail345_6 : List Nat := 3253 :: innerTail345_7
def innerCandidates345_6 : List Nat := [7873]
set_option maxRecDepth 4096 in
theorem innerCandidates345_6_checked : candidates edge 4 2347 3253 innerTail345_7 best.bound = innerCandidates345_6 := by
  unfold candidates
  have row : edge 3253 = SearchData.row3253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan345_6 : scan edge (search edge 3) 4 [2347] 2347 innerTail345_6 best = best := by
  rw [innerTail345_6, scan, if_neg (by decide), innerCandidates345_6_checked]
  change scan edge (search edge 3) 4 [2347] 2347 innerTail345_7
    (if 3 ≤ innerCandidates345_6.length then search edge 3 [2347, 3253] innerCandidates345_6 5600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan345_7

def innerTail345_5 : List Nat := 3067 :: innerTail345_6
def innerCandidates345_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates345_5_checked : candidates edge 4 2347 3067 innerTail345_6 best.bound = innerCandidates345_5 := by
  unfold candidates
  have row : edge 3067 = SearchData.row3067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan345_5 : scan edge (search edge 3) 4 [2347] 2347 innerTail345_5 best = best := by
  rw [innerTail345_5, scan, if_neg (by decide), innerCandidates345_5_checked]
  change scan edge (search edge 3) 4 [2347] 2347 innerTail345_6
    (if 3 ≤ innerCandidates345_5.length then search edge 3 [2347, 3067] innerCandidates345_5 5414 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan345_6

def innerTail345_4 : List Nat := 3049 :: innerTail345_5
def innerCandidates345_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates345_4_checked : candidates edge 4 2347 3049 innerTail345_5 best.bound = innerCandidates345_4 := by
  unfold candidates
  have row : edge 3049 = SearchData.row3049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan345_4 : scan edge (search edge 3) 4 [2347] 2347 innerTail345_4 best = best := by
  rw [innerTail345_4, scan, if_neg (by decide), innerCandidates345_4_checked]
  change scan edge (search edge 3) 4 [2347] 2347 innerTail345_5
    (if 3 ≤ innerCandidates345_4.length then search edge 3 [2347, 3049] innerCandidates345_4 5396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan345_5

def innerTail345_3 : List Nat := 2851 :: innerTail345_4
def innerCandidates345_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates345_3_checked : candidates edge 4 2347 2851 innerTail345_4 best.bound = innerCandidates345_3 := by
  unfold candidates
  have row : edge 2851 = SearchData.row2851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan345_3 : scan edge (search edge 3) 4 [2347] 2347 innerTail345_3 best = best := by
  rw [innerTail345_3, scan, if_neg (by decide), innerCandidates345_3_checked]
  change scan edge (search edge 3) 4 [2347] 2347 innerTail345_4
    (if 3 ≤ innerCandidates345_3.length then search edge 3 [2347, 2851] innerCandidates345_3 5198 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan345_4

def innerTail345_2 : List Nat := 2833 :: innerTail345_3
def innerCandidates345_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates345_2_checked : candidates edge 4 2347 2833 innerTail345_3 best.bound = innerCandidates345_2 := by
  unfold candidates
  have row : edge 2833 = SearchData.row2833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan345_2 : scan edge (search edge 3) 4 [2347] 2347 innerTail345_2 best = best := by
  rw [innerTail345_2, scan, if_neg (by decide), innerCandidates345_2_checked]
  change scan edge (search edge 3) 4 [2347] 2347 innerTail345_3
    (if 3 ≤ innerCandidates345_2.length then search edge 3 [2347, 2833] innerCandidates345_2 5180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan345_3

def innerTail345_1 : List Nat := 2749 :: innerTail345_2
def innerCandidates345_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates345_1_checked : candidates edge 4 2347 2749 innerTail345_2 best.bound = innerCandidates345_1 := by
  unfold candidates
  have row : edge 2749 = SearchData.row2749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan345_1 : scan edge (search edge 3) 4 [2347] 2347 innerTail345_1 best = best := by
  rw [innerTail345_1, scan, if_neg (by decide), innerCandidates345_1_checked]
  change scan edge (search edge 3) 4 [2347] 2347 innerTail345_2
    (if 3 ≤ innerCandidates345_1.length then search edge 3 [2347, 2749] innerCandidates345_1 5096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan345_2

def innerTail345_0 : List Nat := 2731 :: innerTail345_1
def innerCandidates345_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates345_0_checked : candidates edge 4 2347 2731 innerTail345_1 best.bound = innerCandidates345_0 := by
  unfold candidates
  have row : edge 2731 = SearchData.row2731.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan345_0 : scan edge (search edge 3) 4 [2347] 2347 innerTail345_0 best = best := by
  rw [innerTail345_0, scan, if_neg (by decide), innerCandidates345_0_checked]
  change scan edge (search edge 3) 4 [2347] 2347 innerTail345_1
    (if 3 ≤ innerCandidates345_0.length then search edge 3 [2347, 2731] innerCandidates345_0 5078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan345_1

theorem rootBranch345 : search edge 4 [2347] rootCandidates345 2347 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2347] 2347 innerTail345_0 best = best
  exact innerScan345_0

def rootCandidates344 : List Nat := [2833, 3259, 3373, 3967, 5107, 6571, 7489, 7879, 8179, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates344_checked : candidates edge 5 0 2341 tail345 best.bound = rootCandidates344 := by
  unfold candidates
  have row : edge 2341 = PrimePairCache.smallRow2341.contains := by funext q; rfl
  rw [row]
  decide

def innerTail344_10 : List Nat := []
theorem innerScan344_10 : scan edge (search edge 3) 4 [2341] 2341 innerTail344_10 best = best := rfl

def innerTail344_9 : List Nat := 8377 :: innerTail344_10
def innerCandidates344_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates344_9_checked : candidates edge 4 2341 8377 innerTail344_10 best.bound = innerCandidates344_9 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan344_9 : scan edge (search edge 3) 4 [2341] 2341 innerTail344_9 best = best := by
  rw [innerTail344_9, scan, if_neg (by decide), innerCandidates344_9_checked]
  change scan edge (search edge 3) 4 [2341] 2341 innerTail344_10
    (if 3 ≤ innerCandidates344_9.length then search edge 3 [2341, 8377] innerCandidates344_9 10718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan344_10

def innerTail344_8 : List Nat := 8179 :: innerTail344_9
def innerCandidates344_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates344_8_checked : candidates edge 4 2341 8179 innerTail344_9 best.bound = innerCandidates344_8 := by
  unfold candidates
  have row : edge 8179 = SearchData.row8179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan344_8 : scan edge (search edge 3) 4 [2341] 2341 innerTail344_8 best = best := by
  rw [innerTail344_8, scan, if_neg (by decide), innerCandidates344_8_checked]
  change scan edge (search edge 3) 4 [2341] 2341 innerTail344_9
    (if 3 ≤ innerCandidates344_8.length then search edge 3 [2341, 8179] innerCandidates344_8 10520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan344_9

def innerTail344_7 : List Nat := 7879 :: innerTail344_8
def innerCandidates344_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates344_7_checked : candidates edge 4 2341 7879 innerTail344_8 best.bound = innerCandidates344_7 := by
  unfold candidates
  have row : edge 7879 = SearchData.row7879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan344_7 : scan edge (search edge 3) 4 [2341] 2341 innerTail344_7 best = best := by
  rw [innerTail344_7, scan, if_neg (by decide), innerCandidates344_7_checked]
  change scan edge (search edge 3) 4 [2341] 2341 innerTail344_8
    (if 3 ≤ innerCandidates344_7.length then search edge 3 [2341, 7879] innerCandidates344_7 10220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan344_8

def innerTail344_6 : List Nat := 7489 :: innerTail344_7
def innerCandidates344_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates344_6_checked : candidates edge 4 2341 7489 innerTail344_7 best.bound = innerCandidates344_6 := by
  unfold candidates
  have row : edge 7489 = SearchData.row7489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan344_6 : scan edge (search edge 3) 4 [2341] 2341 innerTail344_6 best = best := by
  rw [innerTail344_6, scan, if_neg (by decide), innerCandidates344_6_checked]
  change scan edge (search edge 3) 4 [2341] 2341 innerTail344_7
    (if 3 ≤ innerCandidates344_6.length then search edge 3 [2341, 7489] innerCandidates344_6 9830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan344_7

def innerTail344_5 : List Nat := 6571 :: innerTail344_6
def innerCandidates344_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates344_5_checked : candidates edge 4 2341 6571 innerTail344_6 best.bound = innerCandidates344_5 := by
  unfold candidates
  have row : edge 6571 = SearchData.row6571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan344_5 : scan edge (search edge 3) 4 [2341] 2341 innerTail344_5 best = best := by
  rw [innerTail344_5, scan, if_neg (by decide), innerCandidates344_5_checked]
  change scan edge (search edge 3) 4 [2341] 2341 innerTail344_6
    (if 3 ≤ innerCandidates344_5.length then search edge 3 [2341, 6571] innerCandidates344_5 8912 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan344_6

def innerTail344_4 : List Nat := 5107 :: innerTail344_5
def innerCandidates344_4 : List Nat := [7879]
set_option maxRecDepth 4096 in
theorem innerCandidates344_4_checked : candidates edge 4 2341 5107 innerTail344_5 best.bound = innerCandidates344_4 := by
  unfold candidates
  have row : edge 5107 = SearchData.row5107.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan344_4 : scan edge (search edge 3) 4 [2341] 2341 innerTail344_4 best = best := by
  rw [innerTail344_4, scan, if_neg (by decide), innerCandidates344_4_checked]
  change scan edge (search edge 3) 4 [2341] 2341 innerTail344_5
    (if 3 ≤ innerCandidates344_4.length then search edge 3 [2341, 5107] innerCandidates344_4 7448 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan344_5

def innerTail344_3 : List Nat := 3967 :: innerTail344_4
def innerCandidates344_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates344_3_checked : candidates edge 4 2341 3967 innerTail344_4 best.bound = innerCandidates344_3 := by
  unfold candidates
  have row : edge 3967 = SearchData.row3967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan344_3 : scan edge (search edge 3) 4 [2341] 2341 innerTail344_3 best = best := by
  rw [innerTail344_3, scan, if_neg (by decide), innerCandidates344_3_checked]
  change scan edge (search edge 3) 4 [2341] 2341 innerTail344_4
    (if 3 ≤ innerCandidates344_3.length then search edge 3 [2341, 3967] innerCandidates344_3 6308 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan344_4

def innerTail344_2 : List Nat := 3373 :: innerTail344_3
def innerCandidates344_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates344_2_checked : candidates edge 4 2341 3373 innerTail344_3 best.bound = innerCandidates344_2 := by
  unfold candidates
  have row : edge 3373 = SearchData.row3373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan344_2 : scan edge (search edge 3) 4 [2341] 2341 innerTail344_2 best = best := by
  rw [innerTail344_2, scan, if_neg (by decide), innerCandidates344_2_checked]
  change scan edge (search edge 3) 4 [2341] 2341 innerTail344_3
    (if 3 ≤ innerCandidates344_2.length then search edge 3 [2341, 3373] innerCandidates344_2 5714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan344_3

def innerTail344_1 : List Nat := 3259 :: innerTail344_2
def innerCandidates344_1 : List Nat := [5107]
set_option maxRecDepth 4096 in
theorem innerCandidates344_1_checked : candidates edge 4 2341 3259 innerTail344_2 best.bound = innerCandidates344_1 := by
  unfold candidates
  have row : edge 3259 = SearchData.row3259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan344_1 : scan edge (search edge 3) 4 [2341] 2341 innerTail344_1 best = best := by
  rw [innerTail344_1, scan, if_neg (by decide), innerCandidates344_1_checked]
  change scan edge (search edge 3) 4 [2341] 2341 innerTail344_2
    (if 3 ≤ innerCandidates344_1.length then search edge 3 [2341, 3259] innerCandidates344_1 5600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan344_2

def innerTail344_0 : List Nat := 2833 :: innerTail344_1
def innerCandidates344_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates344_0_checked : candidates edge 4 2341 2833 innerTail344_1 best.bound = innerCandidates344_0 := by
  unfold candidates
  have row : edge 2833 = SearchData.row2833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan344_0 : scan edge (search edge 3) 4 [2341] 2341 innerTail344_0 best = best := by
  rw [innerTail344_0, scan, if_neg (by decide), innerCandidates344_0_checked]
  change scan edge (search edge 3) 4 [2341] 2341 innerTail344_1
    (if 3 ≤ innerCandidates344_0.length then search edge 3 [2341, 2833] innerCandidates344_0 5174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan344_1

theorem rootBranch344 : search edge 4 [2341] rootCandidates344 2341 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2341] 2341 innerTail344_0 best = best
  exact innerScan344_0

def rootCandidates343 : List Nat := [2543, 2729, 3347, 3371, 4127, 4337, 5039, 5081, 5303, 6047, 6101, 6287, 7001, 7013, 7247, 7307, 7487, 7499, 7841, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates343_checked : candidates edge 5 0 2339 tail344 best.bound = rootCandidates343 := by
  unfold candidates
  have row : edge 2339 = PrimePairCache.smallRow2339.contains := by funext q; rfl
  rw [row]
  decide

def innerTail343_20 : List Nat := []
theorem innerScan343_20 : scan edge (search edge 3) 4 [2339] 2339 innerTail343_20 best = best := rfl

def innerTail343_19 : List Nat := 8369 :: innerTail343_20
def innerCandidates343_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates343_19_checked : candidates edge 4 2339 8369 innerTail343_20 best.bound = innerCandidates343_19 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan343_19 : scan edge (search edge 3) 4 [2339] 2339 innerTail343_19 best = best := by
  rw [innerTail343_19, scan, if_neg (by decide), innerCandidates343_19_checked]
  change scan edge (search edge 3) 4 [2339] 2339 innerTail343_20
    (if 3 ≤ innerCandidates343_19.length then search edge 3 [2339, 8369] innerCandidates343_19 10708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan343_20

def innerTail343_18 : List Nat := 7841 :: innerTail343_19
def innerCandidates343_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates343_18_checked : candidates edge 4 2339 7841 innerTail343_19 best.bound = innerCandidates343_18 := by
  unfold candidates
  have row : edge 7841 = SearchData.row7841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan343_18 : scan edge (search edge 3) 4 [2339] 2339 innerTail343_18 best = best := by
  rw [innerTail343_18, scan, if_neg (by decide), innerCandidates343_18_checked]
  change scan edge (search edge 3) 4 [2339] 2339 innerTail343_19
    (if 3 ≤ innerCandidates343_18.length then search edge 3 [2339, 7841] innerCandidates343_18 10180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan343_19

def innerTail343_17 : List Nat := 7499 :: innerTail343_18
def innerCandidates343_17 : List Nat := [7841, 8369]
set_option maxRecDepth 4096 in
theorem innerCandidates343_17_checked : candidates edge 4 2339 7499 innerTail343_18 best.bound = innerCandidates343_17 := by
  unfold candidates
  have row : edge 7499 = SearchData.row7499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan343_17 : scan edge (search edge 3) 4 [2339] 2339 innerTail343_17 best = best := by
  rw [innerTail343_17, scan, if_neg (by decide), innerCandidates343_17_checked]
  change scan edge (search edge 3) 4 [2339] 2339 innerTail343_18
    (if 3 ≤ innerCandidates343_17.length then search edge 3 [2339, 7499] innerCandidates343_17 9838 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan343_18

def innerTail343_16 : List Nat := 7487 :: innerTail343_17
def innerCandidates343_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates343_16_checked : candidates edge 4 2339 7487 innerTail343_17 best.bound = innerCandidates343_16 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan343_16 : scan edge (search edge 3) 4 [2339] 2339 innerTail343_16 best = best := by
  rw [innerTail343_16, scan, if_neg (by decide), innerCandidates343_16_checked]
  change scan edge (search edge 3) 4 [2339] 2339 innerTail343_17
    (if 3 ≤ innerCandidates343_16.length then search edge 3 [2339, 7487] innerCandidates343_16 9826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan343_17

def innerTail343_15 : List Nat := 7307 :: innerTail343_16
def innerCandidates343_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates343_15_checked : candidates edge 4 2339 7307 innerTail343_16 best.bound = innerCandidates343_15 := by
  unfold candidates
  have row : edge 7307 = SearchData.row7307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan343_15 : scan edge (search edge 3) 4 [2339] 2339 innerTail343_15 best = best := by
  rw [innerTail343_15, scan, if_neg (by decide), innerCandidates343_15_checked]
  change scan edge (search edge 3) 4 [2339] 2339 innerTail343_16
    (if 3 ≤ innerCandidates343_15.length then search edge 3 [2339, 7307] innerCandidates343_15 9646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan343_16

def innerTail343_14 : List Nat := 7247 :: innerTail343_15
def innerCandidates343_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates343_14_checked : candidates edge 4 2339 7247 innerTail343_15 best.bound = innerCandidates343_14 := by
  unfold candidates
  have row : edge 7247 = SearchData.row7247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan343_14 : scan edge (search edge 3) 4 [2339] 2339 innerTail343_14 best = best := by
  rw [innerTail343_14, scan, if_neg (by decide), innerCandidates343_14_checked]
  change scan edge (search edge 3) 4 [2339] 2339 innerTail343_15
    (if 3 ≤ innerCandidates343_14.length then search edge 3 [2339, 7247] innerCandidates343_14 9586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan343_15

def innerTail343_13 : List Nat := 7013 :: innerTail343_14
def innerCandidates343_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates343_13_checked : candidates edge 4 2339 7013 innerTail343_14 best.bound = innerCandidates343_13 := by
  unfold candidates
  have row : edge 7013 = SearchData.row7013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan343_13 : scan edge (search edge 3) 4 [2339] 2339 innerTail343_13 best = best := by
  rw [innerTail343_13, scan, if_neg (by decide), innerCandidates343_13_checked]
  change scan edge (search edge 3) 4 [2339] 2339 innerTail343_14
    (if 3 ≤ innerCandidates343_13.length then search edge 3 [2339, 7013] innerCandidates343_13 9352 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan343_14

def innerTail343_12 : List Nat := 7001 :: innerTail343_13
def innerCandidates343_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates343_12_checked : candidates edge 4 2339 7001 innerTail343_13 best.bound = innerCandidates343_12 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan343_12 : scan edge (search edge 3) 4 [2339] 2339 innerTail343_12 best = best := by
  rw [innerTail343_12, scan, if_neg (by decide), innerCandidates343_12_checked]
  change scan edge (search edge 3) 4 [2339] 2339 innerTail343_13
    (if 3 ≤ innerCandidates343_12.length then search edge 3 [2339, 7001] innerCandidates343_12 9340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan343_13

def innerTail343_11 : List Nat := 6287 :: innerTail343_12
def innerCandidates343_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates343_11_checked : candidates edge 4 2339 6287 innerTail343_12 best.bound = innerCandidates343_11 := by
  unfold candidates
  have row : edge 6287 = SearchData.row6287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan343_11 : scan edge (search edge 3) 4 [2339] 2339 innerTail343_11 best = best := by
  rw [innerTail343_11, scan, if_neg (by decide), innerCandidates343_11_checked]
  change scan edge (search edge 3) 4 [2339] 2339 innerTail343_12
    (if 3 ≤ innerCandidates343_11.length then search edge 3 [2339, 6287] innerCandidates343_11 8626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan343_12

def innerTail343_10 : List Nat := 6101 :: innerTail343_11
def innerCandidates343_10 : List Nat := [7487]
set_option maxRecDepth 4096 in
theorem innerCandidates343_10_checked : candidates edge 4 2339 6101 innerTail343_11 best.bound = innerCandidates343_10 := by
  unfold candidates
  have row : edge 6101 = SearchData.row6101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan343_10 : scan edge (search edge 3) 4 [2339] 2339 innerTail343_10 best = best := by
  rw [innerTail343_10, scan, if_neg (by decide), innerCandidates343_10_checked]
  change scan edge (search edge 3) 4 [2339] 2339 innerTail343_11
    (if 3 ≤ innerCandidates343_10.length then search edge 3 [2339, 6101] innerCandidates343_10 8440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan343_11

def innerTail343_9 : List Nat := 6047 :: innerTail343_10
def innerCandidates343_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates343_9_checked : candidates edge 4 2339 6047 innerTail343_10 best.bound = innerCandidates343_9 := by
  unfold candidates
  have row : edge 6047 = SearchData.row6047.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan343_9 : scan edge (search edge 3) 4 [2339] 2339 innerTail343_9 best = best := by
  rw [innerTail343_9, scan, if_neg (by decide), innerCandidates343_9_checked]
  change scan edge (search edge 3) 4 [2339] 2339 innerTail343_10
    (if 3 ≤ innerCandidates343_9.length then search edge 3 [2339, 6047] innerCandidates343_9 8386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan343_10

def innerTail343_8 : List Nat := 5303 :: innerTail343_9
def innerCandidates343_8 : List Nat := [6101]
set_option maxRecDepth 4096 in
theorem innerCandidates343_8_checked : candidates edge 4 2339 5303 innerTail343_9 best.bound = innerCandidates343_8 := by
  unfold candidates
  have row : edge 5303 = SearchData.row5303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan343_8 : scan edge (search edge 3) 4 [2339] 2339 innerTail343_8 best = best := by
  rw [innerTail343_8, scan, if_neg (by decide), innerCandidates343_8_checked]
  change scan edge (search edge 3) 4 [2339] 2339 innerTail343_9
    (if 3 ≤ innerCandidates343_8.length then search edge 3 [2339, 5303] innerCandidates343_8 7642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan343_9

def innerTail343_7 : List Nat := 5081 :: innerTail343_8
def innerCandidates343_7 : List Nat := [7307]
set_option maxRecDepth 4096 in
theorem innerCandidates343_7_checked : candidates edge 4 2339 5081 innerTail343_8 best.bound = innerCandidates343_7 := by
  unfold candidates
  have row : edge 5081 = SearchData.row5081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan343_7 : scan edge (search edge 3) 4 [2339] 2339 innerTail343_7 best = best := by
  rw [innerTail343_7, scan, if_neg (by decide), innerCandidates343_7_checked]
  change scan edge (search edge 3) 4 [2339] 2339 innerTail343_8
    (if 3 ≤ innerCandidates343_7.length then search edge 3 [2339, 5081] innerCandidates343_7 7420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan343_8

def innerTail343_6 : List Nat := 5039 :: innerTail343_7
def innerCandidates343_6 : List Nat := [7001]
set_option maxRecDepth 4096 in
theorem innerCandidates343_6_checked : candidates edge 4 2339 5039 innerTail343_7 best.bound = innerCandidates343_6 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan343_6 : scan edge (search edge 3) 4 [2339] 2339 innerTail343_6 best = best := by
  rw [innerTail343_6, scan, if_neg (by decide), innerCandidates343_6_checked]
  change scan edge (search edge 3) 4 [2339] 2339 innerTail343_7
    (if 3 ≤ innerCandidates343_6.length then search edge 3 [2339, 5039] innerCandidates343_6 7378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan343_7

def innerTail343_5 : List Nat := 4337 :: innerTail343_6
def innerCandidates343_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates343_5_checked : candidates edge 4 2339 4337 innerTail343_6 best.bound = innerCandidates343_5 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan343_5 : scan edge (search edge 3) 4 [2339] 2339 innerTail343_5 best = best := by
  rw [innerTail343_5, scan, if_neg (by decide), innerCandidates343_5_checked]
  change scan edge (search edge 3) 4 [2339] 2339 innerTail343_6
    (if 3 ≤ innerCandidates343_5.length then search edge 3 [2339, 4337] innerCandidates343_5 6676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan343_6

def innerTail343_4 : List Nat := 4127 :: innerTail343_5
def innerCandidates343_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates343_4_checked : candidates edge 4 2339 4127 innerTail343_5 best.bound = innerCandidates343_4 := by
  unfold candidates
  have row : edge 4127 = SearchData.row4127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan343_4 : scan edge (search edge 3) 4 [2339] 2339 innerTail343_4 best = best := by
  rw [innerTail343_4, scan, if_neg (by decide), innerCandidates343_4_checked]
  change scan edge (search edge 3) 4 [2339] 2339 innerTail343_5
    (if 3 ≤ innerCandidates343_4.length then search edge 3 [2339, 4127] innerCandidates343_4 6466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan343_5

def innerTail343_3 : List Nat := 3371 :: innerTail343_4
def innerCandidates343_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates343_3_checked : candidates edge 4 2339 3371 innerTail343_4 best.bound = innerCandidates343_3 := by
  unfold candidates
  have row : edge 3371 = SearchData.row3371.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan343_3 : scan edge (search edge 3) 4 [2339] 2339 innerTail343_3 best = best := by
  rw [innerTail343_3, scan, if_neg (by decide), innerCandidates343_3_checked]
  change scan edge (search edge 3) 4 [2339] 2339 innerTail343_4
    (if 3 ≤ innerCandidates343_3.length then search edge 3 [2339, 3371] innerCandidates343_3 5710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan343_4

def innerTail343_2 : List Nat := 3347 :: innerTail343_3
def innerCandidates343_2 : List Nat := [6101]
set_option maxRecDepth 4096 in
theorem innerCandidates343_2_checked : candidates edge 4 2339 3347 innerTail343_3 best.bound = innerCandidates343_2 := by
  unfold candidates
  have row : edge 3347 = SearchData.row3347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan343_2 : scan edge (search edge 3) 4 [2339] 2339 innerTail343_2 best = best := by
  rw [innerTail343_2, scan, if_neg (by decide), innerCandidates343_2_checked]
  change scan edge (search edge 3) 4 [2339] 2339 innerTail343_3
    (if 3 ≤ innerCandidates343_2.length then search edge 3 [2339, 3347] innerCandidates343_2 5686 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan343_3

def innerTail343_1 : List Nat := 2729 :: innerTail343_2
def innerCandidates343_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates343_1_checked : candidates edge 4 2339 2729 innerTail343_2 best.bound = innerCandidates343_1 := by
  unfold candidates
  have row : edge 2729 = SearchData.row2729.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan343_1 : scan edge (search edge 3) 4 [2339] 2339 innerTail343_1 best = best := by
  rw [innerTail343_1, scan, if_neg (by decide), innerCandidates343_1_checked]
  change scan edge (search edge 3) 4 [2339] 2339 innerTail343_2
    (if 3 ≤ innerCandidates343_1.length then search edge 3 [2339, 2729] innerCandidates343_1 5068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan343_2

def innerTail343_0 : List Nat := 2543 :: innerTail343_1
def innerCandidates343_0 : List Nat := [3371, 4337]
set_option maxRecDepth 4096 in
theorem innerCandidates343_0_checked : candidates edge 4 2339 2543 innerTail343_1 best.bound = innerCandidates343_0 := by
  unfold candidates
  have row : edge 2543 = PrimePairCache.smallRow2543.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan343_0 : scan edge (search edge 3) 4 [2339] 2339 innerTail343_0 best = best := by
  rw [innerTail343_0, scan, if_neg (by decide), innerCandidates343_0_checked]
  change scan edge (search edge 3) 4 [2339] 2339 innerTail343_1
    (if 3 ≤ innerCandidates343_0.length then search edge 3 [2339, 2543] innerCandidates343_0 4882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan343_1

theorem rootBranch343 : search edge 4 [2339] rootCandidates343 2339 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2339] 2339 innerTail343_0 best = best
  exact innerScan343_0

end ProjectEulerSolutions.PrimePairUpper.Search
