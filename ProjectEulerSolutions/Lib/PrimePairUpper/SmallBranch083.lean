import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates465 : List Nat := [3359, 3701, 3803, 3989, 4013, 4349, 4583, 4679, 4787, 4871, 5381, 5393, 5507, 6869, 7229, 7349, 7901]
set_option maxRecDepth 4096 in
theorem rootCandidates465_checked : candidates edge 5 0 3323 tail466 best.bound = rootCandidates465 := by
  unfold candidates
  have row : edge 3323 = SearchData.row3323.contains := by funext q; rfl
  rw [row]
  decide

def innerTail465_17 : List Nat := []
theorem innerScan465_17 : scan edge (search edge 3) 4 [3323] 3323 innerTail465_17 best = best := rfl

def innerTail465_16 : List Nat := 7901 :: innerTail465_17
def innerCandidates465_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates465_16_checked : candidates edge 4 3323 7901 innerTail465_17 best.bound = innerCandidates465_16 := by
  unfold candidates
  have row : edge 7901 = SearchData.row7901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan465_16 : scan edge (search edge 3) 4 [3323] 3323 innerTail465_16 best = best := by
  rw [innerTail465_16, scan, if_neg (by decide), innerCandidates465_16_checked]
  change scan edge (search edge 3) 4 [3323] 3323 innerTail465_17
    (if 3 ≤ innerCandidates465_16.length then search edge 3 [3323, 7901] innerCandidates465_16 11224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan465_17

def innerTail465_15 : List Nat := 7349 :: innerTail465_16
def innerCandidates465_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates465_15_checked : candidates edge 4 3323 7349 innerTail465_16 best.bound = innerCandidates465_15 := by
  unfold candidates
  have row : edge 7349 = SearchData.row7349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan465_15 : scan edge (search edge 3) 4 [3323] 3323 innerTail465_15 best = best := by
  rw [innerTail465_15, scan, if_neg (by decide), innerCandidates465_15_checked]
  change scan edge (search edge 3) 4 [3323] 3323 innerTail465_16
    (if 3 ≤ innerCandidates465_15.length then search edge 3 [3323, 7349] innerCandidates465_15 10672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan465_16

def innerTail465_14 : List Nat := 7229 :: innerTail465_15
def innerCandidates465_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates465_14_checked : candidates edge 4 3323 7229 innerTail465_15 best.bound = innerCandidates465_14 := by
  unfold candidates
  have row : edge 7229 = SearchData.row7229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan465_14 : scan edge (search edge 3) 4 [3323] 3323 innerTail465_14 best = best := by
  rw [innerTail465_14, scan, if_neg (by decide), innerCandidates465_14_checked]
  change scan edge (search edge 3) 4 [3323] 3323 innerTail465_15
    (if 3 ≤ innerCandidates465_14.length then search edge 3 [3323, 7229] innerCandidates465_14 10552 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan465_15

def innerTail465_13 : List Nat := 6869 :: innerTail465_14
def innerCandidates465_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates465_13_checked : candidates edge 4 3323 6869 innerTail465_14 best.bound = innerCandidates465_13 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan465_13 : scan edge (search edge 3) 4 [3323] 3323 innerTail465_13 best = best := by
  rw [innerTail465_13, scan, if_neg (by decide), innerCandidates465_13_checked]
  change scan edge (search edge 3) 4 [3323] 3323 innerTail465_14
    (if 3 ≤ innerCandidates465_13.length then search edge 3 [3323, 6869] innerCandidates465_13 10192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan465_14

def innerTail465_12 : List Nat := 5507 :: innerTail465_13
def innerCandidates465_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates465_12_checked : candidates edge 4 3323 5507 innerTail465_13 best.bound = innerCandidates465_12 := by
  unfold candidates
  have row : edge 5507 = SearchData.row5507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan465_12 : scan edge (search edge 3) 4 [3323] 3323 innerTail465_12 best = best := by
  rw [innerTail465_12, scan, if_neg (by decide), innerCandidates465_12_checked]
  change scan edge (search edge 3) 4 [3323] 3323 innerTail465_13
    (if 3 ≤ innerCandidates465_12.length then search edge 3 [3323, 5507] innerCandidates465_12 8830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan465_13

def innerTail465_11 : List Nat := 5393 :: innerTail465_12
def innerCandidates465_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates465_11_checked : candidates edge 4 3323 5393 innerTail465_12 best.bound = innerCandidates465_11 := by
  unfold candidates
  have row : edge 5393 = SearchData.row5393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan465_11 : scan edge (search edge 3) 4 [3323] 3323 innerTail465_11 best = best := by
  rw [innerTail465_11, scan, if_neg (by decide), innerCandidates465_11_checked]
  change scan edge (search edge 3) 4 [3323] 3323 innerTail465_12
    (if 3 ≤ innerCandidates465_11.length then search edge 3 [3323, 5393] innerCandidates465_11 8716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan465_12

def innerTail465_10 : List Nat := 5381 :: innerTail465_11
def innerCandidates465_10 : List Nat := [5507]
set_option maxRecDepth 4096 in
theorem innerCandidates465_10_checked : candidates edge 4 3323 5381 innerTail465_11 best.bound = innerCandidates465_10 := by
  unfold candidates
  have row : edge 5381 = SearchData.row5381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan465_10 : scan edge (search edge 3) 4 [3323] 3323 innerTail465_10 best = best := by
  rw [innerTail465_10, scan, if_neg (by decide), innerCandidates465_10_checked]
  change scan edge (search edge 3) 4 [3323] 3323 innerTail465_11
    (if 3 ≤ innerCandidates465_10.length then search edge 3 [3323, 5381] innerCandidates465_10 8704 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan465_11

def innerTail465_9 : List Nat := 4871 :: innerTail465_10
def innerCandidates465_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates465_9_checked : candidates edge 4 3323 4871 innerTail465_10 best.bound = innerCandidates465_9 := by
  unfold candidates
  have row : edge 4871 = SearchData.row4871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan465_9 : scan edge (search edge 3) 4 [3323] 3323 innerTail465_9 best = best := by
  rw [innerTail465_9, scan, if_neg (by decide), innerCandidates465_9_checked]
  change scan edge (search edge 3) 4 [3323] 3323 innerTail465_10
    (if 3 ≤ innerCandidates465_9.length then search edge 3 [3323, 4871] innerCandidates465_9 8194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan465_10

def innerTail465_8 : List Nat := 4787 :: innerTail465_9
def innerCandidates465_8 : List Nat := [7229]
set_option maxRecDepth 4096 in
theorem innerCandidates465_8_checked : candidates edge 4 3323 4787 innerTail465_9 best.bound = innerCandidates465_8 := by
  unfold candidates
  have row : edge 4787 = SearchData.row4787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan465_8 : scan edge (search edge 3) 4 [3323] 3323 innerTail465_8 best = best := by
  rw [innerTail465_8, scan, if_neg (by decide), innerCandidates465_8_checked]
  change scan edge (search edge 3) 4 [3323] 3323 innerTail465_9
    (if 3 ≤ innerCandidates465_8.length then search edge 3 [3323, 4787] innerCandidates465_8 8110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan465_9

def innerTail465_7 : List Nat := 4679 :: innerTail465_8
def innerCandidates465_7 : List Nat := [5393]
set_option maxRecDepth 4096 in
theorem innerCandidates465_7_checked : candidates edge 4 3323 4679 innerTail465_8 best.bound = innerCandidates465_7 := by
  unfold candidates
  have row : edge 4679 = SearchData.row4679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan465_7 : scan edge (search edge 3) 4 [3323] 3323 innerTail465_7 best = best := by
  rw [innerTail465_7, scan, if_neg (by decide), innerCandidates465_7_checked]
  change scan edge (search edge 3) 4 [3323] 3323 innerTail465_8
    (if 3 ≤ innerCandidates465_7.length then search edge 3 [3323, 4679] innerCandidates465_7 8002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan465_8

def innerTail465_6 : List Nat := 4583 :: innerTail465_7
def innerCandidates465_6 : List Nat := [4787, 4871]
set_option maxRecDepth 4096 in
theorem innerCandidates465_6_checked : candidates edge 4 3323 4583 innerTail465_7 best.bound = innerCandidates465_6 := by
  unfold candidates
  have row : edge 4583 = SearchData.row4583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan465_6 : scan edge (search edge 3) 4 [3323] 3323 innerTail465_6 best = best := by
  rw [innerTail465_6, scan, if_neg (by decide), innerCandidates465_6_checked]
  change scan edge (search edge 3) 4 [3323] 3323 innerTail465_7
    (if 3 ≤ innerCandidates465_6.length then search edge 3 [3323, 4583] innerCandidates465_6 7906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan465_7

def innerTail465_5 : List Nat := 4349 :: innerTail465_6
def innerCandidates465_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates465_5_checked : candidates edge 4 3323 4349 innerTail465_6 best.bound = innerCandidates465_5 := by
  unfold candidates
  have row : edge 4349 = SearchData.row4349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan465_5 : scan edge (search edge 3) 4 [3323] 3323 innerTail465_5 best = best := by
  rw [innerTail465_5, scan, if_neg (by decide), innerCandidates465_5_checked]
  change scan edge (search edge 3) 4 [3323] 3323 innerTail465_6
    (if 3 ≤ innerCandidates465_5.length then search edge 3 [3323, 4349] innerCandidates465_5 7672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan465_6

def innerTail465_4 : List Nat := 4013 :: innerTail465_5
def innerCandidates465_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates465_4_checked : candidates edge 4 3323 4013 innerTail465_5 best.bound = innerCandidates465_4 := by
  unfold candidates
  have row : edge 4013 = SearchData.row4013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan465_4 : scan edge (search edge 3) 4 [3323] 3323 innerTail465_4 best = best := by
  rw [innerTail465_4, scan, if_neg (by decide), innerCandidates465_4_checked]
  change scan edge (search edge 3) 4 [3323] 3323 innerTail465_5
    (if 3 ≤ innerCandidates465_4.length then search edge 3 [3323, 4013] innerCandidates465_4 7336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan465_5

def innerTail465_3 : List Nat := 3989 :: innerTail465_4
def innerCandidates465_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates465_3_checked : candidates edge 4 3323 3989 innerTail465_4 best.bound = innerCandidates465_3 := by
  unfold candidates
  have row : edge 3989 = SearchData.row3989.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan465_3 : scan edge (search edge 3) 4 [3323] 3323 innerTail465_3 best = best := by
  rw [innerTail465_3, scan, if_neg (by decide), innerCandidates465_3_checked]
  change scan edge (search edge 3) 4 [3323] 3323 innerTail465_4
    (if 3 ≤ innerCandidates465_3.length then search edge 3 [3323, 3989] innerCandidates465_3 7312 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan465_4

def innerTail465_2 : List Nat := 3803 :: innerTail465_3
def innerCandidates465_2 : List Nat := [5507]
set_option maxRecDepth 4096 in
theorem innerCandidates465_2_checked : candidates edge 4 3323 3803 innerTail465_3 best.bound = innerCandidates465_2 := by
  unfold candidates
  have row : edge 3803 = SearchData.row3803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan465_2 : scan edge (search edge 3) 4 [3323] 3323 innerTail465_2 best = best := by
  rw [innerTail465_2, scan, if_neg (by decide), innerCandidates465_2_checked]
  change scan edge (search edge 3) 4 [3323] 3323 innerTail465_3
    (if 3 ≤ innerCandidates465_2.length then search edge 3 [3323, 3803] innerCandidates465_2 7126 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan465_3

def innerTail465_1 : List Nat := 3701 :: innerTail465_2
def innerCandidates465_1 : List Nat := [6869]
set_option maxRecDepth 4096 in
theorem innerCandidates465_1_checked : candidates edge 4 3323 3701 innerTail465_2 best.bound = innerCandidates465_1 := by
  unfold candidates
  have row : edge 3701 = SearchData.row3701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan465_1 : scan edge (search edge 3) 4 [3323] 3323 innerTail465_1 best = best := by
  rw [innerTail465_1, scan, if_neg (by decide), innerCandidates465_1_checked]
  change scan edge (search edge 3) 4 [3323] 3323 innerTail465_2
    (if 3 ≤ innerCandidates465_1.length then search edge 3 [3323, 3701] innerCandidates465_1 7024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan465_2

def innerTail465_0 : List Nat := 3359 :: innerTail465_1
def innerCandidates465_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates465_0_checked : candidates edge 4 3323 3359 innerTail465_1 best.bound = innerCandidates465_0 := by
  unfold candidates
  have row : edge 3359 = SearchData.row3359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan465_0 : scan edge (search edge 3) 4 [3323] 3323 innerTail465_0 best = best := by
  rw [innerTail465_0, scan, if_neg (by decide), innerCandidates465_0_checked]
  change scan edge (search edge 3) 4 [3323] 3323 innerTail465_1
    (if 3 ≤ innerCandidates465_0.length then search edge 3 [3323, 3359] innerCandidates465_0 6682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan465_1

theorem rootBranch465 : search edge 4 [3323] rootCandidates465 3323 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3323] 3323 innerTail465_0 best = best
  exact innerScan465_0

def rootCandidates464 : List Nat := [3373, 3613, 4339, 4363, 4507, 5209, 5419, 6367, 6481, 6577, 7321, 7669, 7951, 8101]
set_option maxRecDepth 4096 in
theorem rootCandidates464_checked : candidates edge 5 0 3319 tail465 best.bound = rootCandidates464 := by
  unfold candidates
  have row : edge 3319 = SearchData.row3319.contains := by funext q; rfl
  rw [row]
  decide

def innerTail464_14 : List Nat := []
theorem innerScan464_14 : scan edge (search edge 3) 4 [3319] 3319 innerTail464_14 best = best := rfl

def innerTail464_13 : List Nat := 8101 :: innerTail464_14
def innerCandidates464_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates464_13_checked : candidates edge 4 3319 8101 innerTail464_14 best.bound = innerCandidates464_13 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan464_13 : scan edge (search edge 3) 4 [3319] 3319 innerTail464_13 best = best := by
  rw [innerTail464_13, scan, if_neg (by decide), innerCandidates464_13_checked]
  change scan edge (search edge 3) 4 [3319] 3319 innerTail464_14
    (if 3 ≤ innerCandidates464_13.length then search edge 3 [3319, 8101] innerCandidates464_13 11420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan464_14

def innerTail464_12 : List Nat := 7951 :: innerTail464_13
def innerCandidates464_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates464_12_checked : candidates edge 4 3319 7951 innerTail464_13 best.bound = innerCandidates464_12 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan464_12 : scan edge (search edge 3) 4 [3319] 3319 innerTail464_12 best = best := by
  rw [innerTail464_12, scan, if_neg (by decide), innerCandidates464_12_checked]
  change scan edge (search edge 3) 4 [3319] 3319 innerTail464_13
    (if 3 ≤ innerCandidates464_12.length then search edge 3 [3319, 7951] innerCandidates464_12 11270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan464_13

def innerTail464_11 : List Nat := 7669 :: innerTail464_12
def innerCandidates464_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates464_11_checked : candidates edge 4 3319 7669 innerTail464_12 best.bound = innerCandidates464_11 := by
  unfold candidates
  have row : edge 7669 = SearchData.row7669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan464_11 : scan edge (search edge 3) 4 [3319] 3319 innerTail464_11 best = best := by
  rw [innerTail464_11, scan, if_neg (by decide), innerCandidates464_11_checked]
  change scan edge (search edge 3) 4 [3319] 3319 innerTail464_12
    (if 3 ≤ innerCandidates464_11.length then search edge 3 [3319, 7669] innerCandidates464_11 10988 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan464_12

def innerTail464_10 : List Nat := 7321 :: innerTail464_11
def innerCandidates464_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates464_10_checked : candidates edge 4 3319 7321 innerTail464_11 best.bound = innerCandidates464_10 := by
  unfold candidates
  have row : edge 7321 = SearchData.row7321.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan464_10 : scan edge (search edge 3) 4 [3319] 3319 innerTail464_10 best = best := by
  rw [innerTail464_10, scan, if_neg (by decide), innerCandidates464_10_checked]
  change scan edge (search edge 3) 4 [3319] 3319 innerTail464_11
    (if 3 ≤ innerCandidates464_10.length then search edge 3 [3319, 7321] innerCandidates464_10 10640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan464_11

def innerTail464_9 : List Nat := 6577 :: innerTail464_10
def innerCandidates464_9 : List Nat := [7669]
set_option maxRecDepth 4096 in
theorem innerCandidates464_9_checked : candidates edge 4 3319 6577 innerTail464_10 best.bound = innerCandidates464_9 := by
  unfold candidates
  have row : edge 6577 = SearchData.row6577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan464_9 : scan edge (search edge 3) 4 [3319] 3319 innerTail464_9 best = best := by
  rw [innerTail464_9, scan, if_neg (by decide), innerCandidates464_9_checked]
  change scan edge (search edge 3) 4 [3319] 3319 innerTail464_10
    (if 3 ≤ innerCandidates464_9.length then search edge 3 [3319, 6577] innerCandidates464_9 9896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan464_10

def innerTail464_8 : List Nat := 6481 :: innerTail464_9
def innerCandidates464_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates464_8_checked : candidates edge 4 3319 6481 innerTail464_9 best.bound = innerCandidates464_8 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan464_8 : scan edge (search edge 3) 4 [3319] 3319 innerTail464_8 best = best := by
  rw [innerTail464_8, scan, if_neg (by decide), innerCandidates464_8_checked]
  change scan edge (search edge 3) 4 [3319] 3319 innerTail464_9
    (if 3 ≤ innerCandidates464_8.length then search edge 3 [3319, 6481] innerCandidates464_8 9800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan464_9

def innerTail464_7 : List Nat := 6367 :: innerTail464_8
def innerCandidates464_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates464_7_checked : candidates edge 4 3319 6367 innerTail464_8 best.bound = innerCandidates464_7 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan464_7 : scan edge (search edge 3) 4 [3319] 3319 innerTail464_7 best = best := by
  rw [innerTail464_7, scan, if_neg (by decide), innerCandidates464_7_checked]
  change scan edge (search edge 3) 4 [3319] 3319 innerTail464_8
    (if 3 ≤ innerCandidates464_7.length then search edge 3 [3319, 6367] innerCandidates464_7 9686 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan464_8

def innerTail464_6 : List Nat := 5419 :: innerTail464_7
def innerCandidates464_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates464_6_checked : candidates edge 4 3319 5419 innerTail464_7 best.bound = innerCandidates464_6 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan464_6 : scan edge (search edge 3) 4 [3319] 3319 innerTail464_6 best = best := by
  rw [innerTail464_6, scan, if_neg (by decide), innerCandidates464_6_checked]
  change scan edge (search edge 3) 4 [3319] 3319 innerTail464_7
    (if 3 ≤ innerCandidates464_6.length then search edge 3 [3319, 5419] innerCandidates464_6 8738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan464_7

def innerTail464_5 : List Nat := 5209 :: innerTail464_6
def innerCandidates464_5 : List Nat := [5419]
set_option maxRecDepth 4096 in
theorem innerCandidates464_5_checked : candidates edge 4 3319 5209 innerTail464_6 best.bound = innerCandidates464_5 := by
  unfold candidates
  have row : edge 5209 = SearchData.row5209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan464_5 : scan edge (search edge 3) 4 [3319] 3319 innerTail464_5 best = best := by
  rw [innerTail464_5, scan, if_neg (by decide), innerCandidates464_5_checked]
  change scan edge (search edge 3) 4 [3319] 3319 innerTail464_6
    (if 3 ≤ innerCandidates464_5.length then search edge 3 [3319, 5209] innerCandidates464_5 8528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan464_6

def innerTail464_4 : List Nat := 4507 :: innerTail464_5
def innerCandidates464_4 : List Nat := [5419]
set_option maxRecDepth 4096 in
theorem innerCandidates464_4_checked : candidates edge 4 3319 4507 innerTail464_5 best.bound = innerCandidates464_4 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan464_4 : scan edge (search edge 3) 4 [3319] 3319 innerTail464_4 best = best := by
  rw [innerTail464_4, scan, if_neg (by decide), innerCandidates464_4_checked]
  change scan edge (search edge 3) 4 [3319] 3319 innerTail464_5
    (if 3 ≤ innerCandidates464_4.length then search edge 3 [3319, 4507] innerCandidates464_4 7826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan464_5

def innerTail464_3 : List Nat := 4363 :: innerTail464_4
def innerCandidates464_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates464_3_checked : candidates edge 4 3319 4363 innerTail464_4 best.bound = innerCandidates464_3 := by
  unfold candidates
  have row : edge 4363 = SearchData.row4363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan464_3 : scan edge (search edge 3) 4 [3319] 3319 innerTail464_3 best = best := by
  rw [innerTail464_3, scan, if_neg (by decide), innerCandidates464_3_checked]
  change scan edge (search edge 3) 4 [3319] 3319 innerTail464_4
    (if 3 ≤ innerCandidates464_3.length then search edge 3 [3319, 4363] innerCandidates464_3 7682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan464_4

def innerTail464_2 : List Nat := 4339 :: innerTail464_3
def innerCandidates464_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates464_2_checked : candidates edge 4 3319 4339 innerTail464_3 best.bound = innerCandidates464_2 := by
  unfold candidates
  have row : edge 4339 = SearchData.row4339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan464_2 : scan edge (search edge 3) 4 [3319] 3319 innerTail464_2 best = best := by
  rw [innerTail464_2, scan, if_neg (by decide), innerCandidates464_2_checked]
  change scan edge (search edge 3) 4 [3319] 3319 innerTail464_3
    (if 3 ≤ innerCandidates464_2.length then search edge 3 [3319, 4339] innerCandidates464_2 7658 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan464_3

def innerTail464_1 : List Nat := 3613 :: innerTail464_2
def innerCandidates464_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates464_1_checked : candidates edge 4 3319 3613 innerTail464_2 best.bound = innerCandidates464_1 := by
  unfold candidates
  have row : edge 3613 = SearchData.row3613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan464_1 : scan edge (search edge 3) 4 [3319] 3319 innerTail464_1 best = best := by
  rw [innerTail464_1, scan, if_neg (by decide), innerCandidates464_1_checked]
  change scan edge (search edge 3) 4 [3319] 3319 innerTail464_2
    (if 3 ≤ innerCandidates464_1.length then search edge 3 [3319, 3613] innerCandidates464_1 6932 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan464_2

def innerTail464_0 : List Nat := 3373 :: innerTail464_1
def innerCandidates464_0 : List Nat := [4339]
set_option maxRecDepth 4096 in
theorem innerCandidates464_0_checked : candidates edge 4 3319 3373 innerTail464_1 best.bound = innerCandidates464_0 := by
  unfold candidates
  have row : edge 3373 = SearchData.row3373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan464_0 : scan edge (search edge 3) 4 [3319] 3319 innerTail464_0 best = best := by
  rw [innerTail464_0, scan, if_neg (by decide), innerCandidates464_0_checked]
  change scan edge (search edge 3) 4 [3319] 3319 innerTail464_1
    (if 3 ≤ innerCandidates464_0.length then search edge 3 [3319, 3373] innerCandidates464_0 6692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan464_1

theorem rootBranch464 : search edge 4 [3319] rootCandidates464 3319 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3319] 3319 innerTail464_0 best = best
  exact innerScan464_0

def rootCandidates463 : List Nat := [3673, 3733, 4111, 4159, 4663, 4723, 4831, 4951, 6421, 6637, 6871, 8059, 8089]
set_option maxRecDepth 4096 in
theorem rootCandidates463_checked : candidates edge 5 0 3313 tail464 best.bound = rootCandidates463 := by
  unfold candidates
  have row : edge 3313 = SearchData.row3313.contains := by funext q; rfl
  rw [row]
  decide

def innerTail463_13 : List Nat := []
theorem innerScan463_13 : scan edge (search edge 3) 4 [3313] 3313 innerTail463_13 best = best := rfl

def innerTail463_12 : List Nat := 8089 :: innerTail463_13
def innerCandidates463_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates463_12_checked : candidates edge 4 3313 8089 innerTail463_13 best.bound = innerCandidates463_12 := by
  unfold candidates
  have row : edge 8089 = SearchData.row8089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan463_12 : scan edge (search edge 3) 4 [3313] 3313 innerTail463_12 best = best := by
  rw [innerTail463_12, scan, if_neg (by decide), innerCandidates463_12_checked]
  change scan edge (search edge 3) 4 [3313] 3313 innerTail463_13
    (if 3 ≤ innerCandidates463_12.length then search edge 3 [3313, 8089] innerCandidates463_12 11402 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan463_13

def innerTail463_11 : List Nat := 8059 :: innerTail463_12
def innerCandidates463_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates463_11_checked : candidates edge 4 3313 8059 innerTail463_12 best.bound = innerCandidates463_11 := by
  unfold candidates
  have row : edge 8059 = SearchData.row8059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan463_11 : scan edge (search edge 3) 4 [3313] 3313 innerTail463_11 best = best := by
  rw [innerTail463_11, scan, if_neg (by decide), innerCandidates463_11_checked]
  change scan edge (search edge 3) 4 [3313] 3313 innerTail463_12
    (if 3 ≤ innerCandidates463_11.length then search edge 3 [3313, 8059] innerCandidates463_11 11372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan463_12

def innerTail463_10 : List Nat := 6871 :: innerTail463_11
def innerCandidates463_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates463_10_checked : candidates edge 4 3313 6871 innerTail463_11 best.bound = innerCandidates463_10 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan463_10 : scan edge (search edge 3) 4 [3313] 3313 innerTail463_10 best = best := by
  rw [innerTail463_10, scan, if_neg (by decide), innerCandidates463_10_checked]
  change scan edge (search edge 3) 4 [3313] 3313 innerTail463_11
    (if 3 ≤ innerCandidates463_10.length then search edge 3 [3313, 6871] innerCandidates463_10 10184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan463_11

def innerTail463_9 : List Nat := 6637 :: innerTail463_10
def innerCandidates463_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates463_9_checked : candidates edge 4 3313 6637 innerTail463_10 best.bound = innerCandidates463_9 := by
  unfold candidates
  have row : edge 6637 = SearchData.row6637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan463_9 : scan edge (search edge 3) 4 [3313] 3313 innerTail463_9 best = best := by
  rw [innerTail463_9, scan, if_neg (by decide), innerCandidates463_9_checked]
  change scan edge (search edge 3) 4 [3313] 3313 innerTail463_10
    (if 3 ≤ innerCandidates463_9.length then search edge 3 [3313, 6637] innerCandidates463_9 9950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan463_10

def innerTail463_8 : List Nat := 6421 :: innerTail463_9
def innerCandidates463_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates463_8_checked : candidates edge 4 3313 6421 innerTail463_9 best.bound = innerCandidates463_8 := by
  unfold candidates
  have row : edge 6421 = SearchData.row6421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan463_8 : scan edge (search edge 3) 4 [3313] 3313 innerTail463_8 best = best := by
  rw [innerTail463_8, scan, if_neg (by decide), innerCandidates463_8_checked]
  change scan edge (search edge 3) 4 [3313] 3313 innerTail463_9
    (if 3 ≤ innerCandidates463_8.length then search edge 3 [3313, 6421] innerCandidates463_8 9734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan463_9

def innerTail463_7 : List Nat := 4951 :: innerTail463_8
def innerCandidates463_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates463_7_checked : candidates edge 4 3313 4951 innerTail463_8 best.bound = innerCandidates463_7 := by
  unfold candidates
  have row : edge 4951 = SearchData.row4951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan463_7 : scan edge (search edge 3) 4 [3313] 3313 innerTail463_7 best = best := by
  rw [innerTail463_7, scan, if_neg (by decide), innerCandidates463_7_checked]
  change scan edge (search edge 3) 4 [3313] 3313 innerTail463_8
    (if 3 ≤ innerCandidates463_7.length then search edge 3 [3313, 4951] innerCandidates463_7 8264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan463_8

def innerTail463_6 : List Nat := 4831 :: innerTail463_7
def innerCandidates463_6 : List Nat := [4951]
set_option maxRecDepth 4096 in
theorem innerCandidates463_6_checked : candidates edge 4 3313 4831 innerTail463_7 best.bound = innerCandidates463_6 := by
  unfold candidates
  have row : edge 4831 = SearchData.row4831.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan463_6 : scan edge (search edge 3) 4 [3313] 3313 innerTail463_6 best = best := by
  rw [innerTail463_6, scan, if_neg (by decide), innerCandidates463_6_checked]
  change scan edge (search edge 3) 4 [3313] 3313 innerTail463_7
    (if 3 ≤ innerCandidates463_6.length then search edge 3 [3313, 4831] innerCandidates463_6 8144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan463_7

def innerTail463_5 : List Nat := 4723 :: innerTail463_6
def innerCandidates463_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates463_5_checked : candidates edge 4 3313 4723 innerTail463_6 best.bound = innerCandidates463_5 := by
  unfold candidates
  have row : edge 4723 = SearchData.row4723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan463_5 : scan edge (search edge 3) 4 [3313] 3313 innerTail463_5 best = best := by
  rw [innerTail463_5, scan, if_neg (by decide), innerCandidates463_5_checked]
  change scan edge (search edge 3) 4 [3313] 3313 innerTail463_6
    (if 3 ≤ innerCandidates463_5.length then search edge 3 [3313, 4723] innerCandidates463_5 8036 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan463_6

def innerTail463_4 : List Nat := 4663 :: innerTail463_5
def innerCandidates463_4 : List Nat := [6637]
set_option maxRecDepth 4096 in
theorem innerCandidates463_4_checked : candidates edge 4 3313 4663 innerTail463_5 best.bound = innerCandidates463_4 := by
  unfold candidates
  have row : edge 4663 = SearchData.row4663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan463_4 : scan edge (search edge 3) 4 [3313] 3313 innerTail463_4 best = best := by
  rw [innerTail463_4, scan, if_neg (by decide), innerCandidates463_4_checked]
  change scan edge (search edge 3) 4 [3313] 3313 innerTail463_5
    (if 3 ≤ innerCandidates463_4.length then search edge 3 [3313, 4663] innerCandidates463_4 7976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan463_5

def innerTail463_3 : List Nat := 4159 :: innerTail463_4
def innerCandidates463_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates463_3_checked : candidates edge 4 3313 4159 innerTail463_4 best.bound = innerCandidates463_3 := by
  unfold candidates
  have row : edge 4159 = SearchData.row4159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan463_3 : scan edge (search edge 3) 4 [3313] 3313 innerTail463_3 best = best := by
  rw [innerTail463_3, scan, if_neg (by decide), innerCandidates463_3_checked]
  change scan edge (search edge 3) 4 [3313] 3313 innerTail463_4
    (if 3 ≤ innerCandidates463_3.length then search edge 3 [3313, 4159] innerCandidates463_3 7472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan463_4

def innerTail463_2 : List Nat := 4111 :: innerTail463_3
def innerCandidates463_2 : List Nat := [6637]
set_option maxRecDepth 4096 in
theorem innerCandidates463_2_checked : candidates edge 4 3313 4111 innerTail463_3 best.bound = innerCandidates463_2 := by
  unfold candidates
  have row : edge 4111 = SearchData.row4111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan463_2 : scan edge (search edge 3) 4 [3313] 3313 innerTail463_2 best = best := by
  rw [innerTail463_2, scan, if_neg (by decide), innerCandidates463_2_checked]
  change scan edge (search edge 3) 4 [3313] 3313 innerTail463_3
    (if 3 ≤ innerCandidates463_2.length then search edge 3 [3313, 4111] innerCandidates463_2 7424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan463_3

def innerTail463_1 : List Nat := 3733 :: innerTail463_2
def innerCandidates463_1 : List Nat := [6421, 8089]
set_option maxRecDepth 4096 in
theorem innerCandidates463_1_checked : candidates edge 4 3313 3733 innerTail463_2 best.bound = innerCandidates463_1 := by
  unfold candidates
  have row : edge 3733 = SearchData.row3733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan463_1 : scan edge (search edge 3) 4 [3313] 3313 innerTail463_1 best = best := by
  rw [innerTail463_1, scan, if_neg (by decide), innerCandidates463_1_checked]
  change scan edge (search edge 3) 4 [3313] 3313 innerTail463_2
    (if 3 ≤ innerCandidates463_1.length then search edge 3 [3313, 3733] innerCandidates463_1 7046 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan463_2

def innerTail463_0 : List Nat := 3673 :: innerTail463_1
def innerCandidates463_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates463_0_checked : candidates edge 4 3313 3673 innerTail463_1 best.bound = innerCandidates463_0 := by
  unfold candidates
  have row : edge 3673 = SearchData.row3673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan463_0 : scan edge (search edge 3) 4 [3313] 3313 innerTail463_0 best = best := by
  rw [innerTail463_0, scan, if_neg (by decide), innerCandidates463_0_checked]
  change scan edge (search edge 3) 4 [3313] 3313 innerTail463_1
    (if 3 ≤ innerCandidates463_0.length then search edge 3 [3313, 3673] innerCandidates463_0 6986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan463_1

theorem rootBranch463 : search edge 4 [3313] rootCandidates463 3313 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3313] 3313 innerTail463_0 best = best
  exact innerScan463_0

def rootCandidates462 : List Nat := [3331, 3583, 4519, 5443, 5737, 5953, 6151, 6367, 6619, 6661, 7039, 7321, 7411]
set_option maxRecDepth 4096 in
theorem rootCandidates462_checked : candidates edge 5 0 3307 tail463 best.bound = rootCandidates462 := by
  unfold candidates
  have row : edge 3307 = SearchData.row3307.contains := by funext q; rfl
  rw [row]
  decide

def innerTail462_13 : List Nat := []
theorem innerScan462_13 : scan edge (search edge 3) 4 [3307] 3307 innerTail462_13 best = best := rfl

def innerTail462_12 : List Nat := 7411 :: innerTail462_13
def innerCandidates462_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates462_12_checked : candidates edge 4 3307 7411 innerTail462_13 best.bound = innerCandidates462_12 := by
  unfold candidates
  have row : edge 7411 = SearchData.row7411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan462_12 : scan edge (search edge 3) 4 [3307] 3307 innerTail462_12 best = best := by
  rw [innerTail462_12, scan, if_neg (by decide), innerCandidates462_12_checked]
  change scan edge (search edge 3) 4 [3307] 3307 innerTail462_13
    (if 3 ≤ innerCandidates462_12.length then search edge 3 [3307, 7411] innerCandidates462_12 10718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan462_13

def innerTail462_11 : List Nat := 7321 :: innerTail462_12
def innerCandidates462_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates462_11_checked : candidates edge 4 3307 7321 innerTail462_12 best.bound = innerCandidates462_11 := by
  unfold candidates
  have row : edge 7321 = SearchData.row7321.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan462_11 : scan edge (search edge 3) 4 [3307] 3307 innerTail462_11 best = best := by
  rw [innerTail462_11, scan, if_neg (by decide), innerCandidates462_11_checked]
  change scan edge (search edge 3) 4 [3307] 3307 innerTail462_12
    (if 3 ≤ innerCandidates462_11.length then search edge 3 [3307, 7321] innerCandidates462_11 10628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan462_12

def innerTail462_10 : List Nat := 7039 :: innerTail462_11
def innerCandidates462_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates462_10_checked : candidates edge 4 3307 7039 innerTail462_11 best.bound = innerCandidates462_10 := by
  unfold candidates
  have row : edge 7039 = SearchData.row7039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan462_10 : scan edge (search edge 3) 4 [3307] 3307 innerTail462_10 best = best := by
  rw [innerTail462_10, scan, if_neg (by decide), innerCandidates462_10_checked]
  change scan edge (search edge 3) 4 [3307] 3307 innerTail462_11
    (if 3 ≤ innerCandidates462_10.length then search edge 3 [3307, 7039] innerCandidates462_10 10346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan462_11

def innerTail462_9 : List Nat := 6661 :: innerTail462_10
def innerCandidates462_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates462_9_checked : candidates edge 4 3307 6661 innerTail462_10 best.bound = innerCandidates462_9 := by
  unfold candidates
  have row : edge 6661 = SearchData.row6661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan462_9 : scan edge (search edge 3) 4 [3307] 3307 innerTail462_9 best = best := by
  rw [innerTail462_9, scan, if_neg (by decide), innerCandidates462_9_checked]
  change scan edge (search edge 3) 4 [3307] 3307 innerTail462_10
    (if 3 ≤ innerCandidates462_9.length then search edge 3 [3307, 6661] innerCandidates462_9 9968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan462_10

def innerTail462_8 : List Nat := 6619 :: innerTail462_9
def innerCandidates462_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates462_8_checked : candidates edge 4 3307 6619 innerTail462_9 best.bound = innerCandidates462_8 := by
  unfold candidates
  have row : edge 6619 = SearchData.row6619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan462_8 : scan edge (search edge 3) 4 [3307] 3307 innerTail462_8 best = best := by
  rw [innerTail462_8, scan, if_neg (by decide), innerCandidates462_8_checked]
  change scan edge (search edge 3) 4 [3307] 3307 innerTail462_9
    (if 3 ≤ innerCandidates462_8.length then search edge 3 [3307, 6619] innerCandidates462_8 9926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan462_9

def innerTail462_7 : List Nat := 6367 :: innerTail462_8
def innerCandidates462_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates462_7_checked : candidates edge 4 3307 6367 innerTail462_8 best.bound = innerCandidates462_7 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan462_7 : scan edge (search edge 3) 4 [3307] 3307 innerTail462_7 best = best := by
  rw [innerTail462_7, scan, if_neg (by decide), innerCandidates462_7_checked]
  change scan edge (search edge 3) 4 [3307] 3307 innerTail462_8
    (if 3 ≤ innerCandidates462_7.length then search edge 3 [3307, 6367] innerCandidates462_7 9674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan462_8

def innerTail462_6 : List Nat := 6151 :: innerTail462_7
def innerCandidates462_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates462_6_checked : candidates edge 4 3307 6151 innerTail462_7 best.bound = innerCandidates462_6 := by
  unfold candidates
  have row : edge 6151 = SearchData.row6151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan462_6 : scan edge (search edge 3) 4 [3307] 3307 innerTail462_6 best = best := by
  rw [innerTail462_6, scan, if_neg (by decide), innerCandidates462_6_checked]
  change scan edge (search edge 3) 4 [3307] 3307 innerTail462_7
    (if 3 ≤ innerCandidates462_6.length then search edge 3 [3307, 6151] innerCandidates462_6 9458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan462_7

def innerTail462_5 : List Nat := 5953 :: innerTail462_6
def innerCandidates462_5 : List Nat := [7039]
set_option maxRecDepth 4096 in
theorem innerCandidates462_5_checked : candidates edge 4 3307 5953 innerTail462_6 best.bound = innerCandidates462_5 := by
  unfold candidates
  have row : edge 5953 = SearchData.row5953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan462_5 : scan edge (search edge 3) 4 [3307] 3307 innerTail462_5 best = best := by
  rw [innerTail462_5, scan, if_neg (by decide), innerCandidates462_5_checked]
  change scan edge (search edge 3) 4 [3307] 3307 innerTail462_6
    (if 3 ≤ innerCandidates462_5.length then search edge 3 [3307, 5953] innerCandidates462_5 9260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan462_6

def innerTail462_4 : List Nat := 5737 :: innerTail462_5
def innerCandidates462_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates462_4_checked : candidates edge 4 3307 5737 innerTail462_5 best.bound = innerCandidates462_4 := by
  unfold candidates
  have row : edge 5737 = SearchData.row5737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan462_4 : scan edge (search edge 3) 4 [3307] 3307 innerTail462_4 best = best := by
  rw [innerTail462_4, scan, if_neg (by decide), innerCandidates462_4_checked]
  change scan edge (search edge 3) 4 [3307] 3307 innerTail462_5
    (if 3 ≤ innerCandidates462_4.length then search edge 3 [3307, 5737] innerCandidates462_4 9044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan462_5

def innerTail462_3 : List Nat := 5443 :: innerTail462_4
def innerCandidates462_3 : List Nat := [7039]
set_option maxRecDepth 4096 in
theorem innerCandidates462_3_checked : candidates edge 4 3307 5443 innerTail462_4 best.bound = innerCandidates462_3 := by
  unfold candidates
  have row : edge 5443 = SearchData.row5443.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan462_3 : scan edge (search edge 3) 4 [3307] 3307 innerTail462_3 best = best := by
  rw [innerTail462_3, scan, if_neg (by decide), innerCandidates462_3_checked]
  change scan edge (search edge 3) 4 [3307] 3307 innerTail462_4
    (if 3 ≤ innerCandidates462_3.length then search edge 3 [3307, 5443] innerCandidates462_3 8750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan462_4

def innerTail462_2 : List Nat := 4519 :: innerTail462_3
def innerCandidates462_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates462_2_checked : candidates edge 4 3307 4519 innerTail462_3 best.bound = innerCandidates462_2 := by
  unfold candidates
  have row : edge 4519 = SearchData.row4519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan462_2 : scan edge (search edge 3) 4 [3307] 3307 innerTail462_2 best = best := by
  rw [innerTail462_2, scan, if_neg (by decide), innerCandidates462_2_checked]
  change scan edge (search edge 3) 4 [3307] 3307 innerTail462_3
    (if 3 ≤ innerCandidates462_2.length then search edge 3 [3307, 4519] innerCandidates462_2 7826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan462_3

def innerTail462_1 : List Nat := 3583 :: innerTail462_2
def innerCandidates462_1 : List Nat := [5953]
set_option maxRecDepth 4096 in
theorem innerCandidates462_1_checked : candidates edge 4 3307 3583 innerTail462_2 best.bound = innerCandidates462_1 := by
  unfold candidates
  have row : edge 3583 = SearchData.row3583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan462_1 : scan edge (search edge 3) 4 [3307] 3307 innerTail462_1 best = best := by
  rw [innerTail462_1, scan, if_neg (by decide), innerCandidates462_1_checked]
  change scan edge (search edge 3) 4 [3307] 3307 innerTail462_2
    (if 3 ≤ innerCandidates462_1.length then search edge 3 [3307, 3583] innerCandidates462_1 6890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan462_2

def innerTail462_0 : List Nat := 3331 :: innerTail462_1
def innerCandidates462_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates462_0_checked : candidates edge 4 3307 3331 innerTail462_1 best.bound = innerCandidates462_0 := by
  unfold candidates
  have row : edge 3331 = SearchData.row3331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan462_0 : scan edge (search edge 3) 4 [3307] 3307 innerTail462_0 best = best := by
  rw [innerTail462_0, scan, if_neg (by decide), innerCandidates462_0_checked]
  change scan edge (search edge 3) 4 [3307] 3307 innerTail462_1
    (if 3 ≤ innerCandidates462_0.length then search edge 3 [3307, 3331] innerCandidates462_0 6638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan462_1

theorem rootBranch462 : search edge 4 [3307] rootCandidates462 3307 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3307] 3307 innerTail462_0 best = best
  exact innerScan462_0

end ProjectEulerSolutions.PrimePairUpper.Search
