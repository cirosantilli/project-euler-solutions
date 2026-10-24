import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates429 : List Nat := [3023, 3329, 4493, 4817, 6329, 6353, 6833, 7127, 7349, 7883, 7937, 8009, 8147, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates429_checked : candidates edge 5 0 3011 tail430 best.bound = rootCandidates429 := by
  unfold candidates
  have row : edge 3011 = SearchData.row3011.contains := by funext q; rfl
  rw [row]
  decide

def innerTail429_14 : List Nat := []
theorem innerScan429_14 : scan edge (search edge 3) 4 [3011] 3011 innerTail429_14 best = best := rfl

def innerTail429_13 : List Nat := 8369 :: innerTail429_14
def innerCandidates429_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates429_13_checked : candidates edge 4 3011 8369 innerTail429_14 best.bound = innerCandidates429_13 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan429_13 : scan edge (search edge 3) 4 [3011] 3011 innerTail429_13 best = best := by
  rw [innerTail429_13, scan, if_neg (by decide), innerCandidates429_13_checked]
  change scan edge (search edge 3) 4 [3011] 3011 innerTail429_14
    (if 3 ≤ innerCandidates429_13.length then search edge 3 [3011, 8369] innerCandidates429_13 11380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan429_14

def innerTail429_12 : List Nat := 8147 :: innerTail429_13
def innerCandidates429_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates429_12_checked : candidates edge 4 3011 8147 innerTail429_13 best.bound = innerCandidates429_12 := by
  unfold candidates
  have row : edge 8147 = SearchData.row8147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan429_12 : scan edge (search edge 3) 4 [3011] 3011 innerTail429_12 best = best := by
  rw [innerTail429_12, scan, if_neg (by decide), innerCandidates429_12_checked]
  change scan edge (search edge 3) 4 [3011] 3011 innerTail429_13
    (if 3 ≤ innerCandidates429_12.length then search edge 3 [3011, 8147] innerCandidates429_12 11158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan429_13

def innerTail429_11 : List Nat := 8009 :: innerTail429_12
def innerCandidates429_11 : List Nat := [8147]
set_option maxRecDepth 4096 in
theorem innerCandidates429_11_checked : candidates edge 4 3011 8009 innerTail429_12 best.bound = innerCandidates429_11 := by
  unfold candidates
  have row : edge 8009 = SearchData.row8009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan429_11 : scan edge (search edge 3) 4 [3011] 3011 innerTail429_11 best = best := by
  rw [innerTail429_11, scan, if_neg (by decide), innerCandidates429_11_checked]
  change scan edge (search edge 3) 4 [3011] 3011 innerTail429_12
    (if 3 ≤ innerCandidates429_11.length then search edge 3 [3011, 8009] innerCandidates429_11 11020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan429_12

def innerTail429_10 : List Nat := 7937 :: innerTail429_11
def innerCandidates429_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates429_10_checked : candidates edge 4 3011 7937 innerTail429_11 best.bound = innerCandidates429_10 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan429_10 : scan edge (search edge 3) 4 [3011] 3011 innerTail429_10 best = best := by
  rw [innerTail429_10, scan, if_neg (by decide), innerCandidates429_10_checked]
  change scan edge (search edge 3) 4 [3011] 3011 innerTail429_11
    (if 3 ≤ innerCandidates429_10.length then search edge 3 [3011, 7937] innerCandidates429_10 10948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan429_11

def innerTail429_9 : List Nat := 7883 :: innerTail429_10
def innerCandidates429_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates429_9_checked : candidates edge 4 3011 7883 innerTail429_10 best.bound = innerCandidates429_9 := by
  unfold candidates
  have row : edge 7883 = SearchData.row7883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan429_9 : scan edge (search edge 3) 4 [3011] 3011 innerTail429_9 best = best := by
  rw [innerTail429_9, scan, if_neg (by decide), innerCandidates429_9_checked]
  change scan edge (search edge 3) 4 [3011] 3011 innerTail429_10
    (if 3 ≤ innerCandidates429_9.length then search edge 3 [3011, 7883] innerCandidates429_9 10894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan429_10

def innerTail429_8 : List Nat := 7349 :: innerTail429_9
def innerCandidates429_8 : List Nat := [8147]
set_option maxRecDepth 4096 in
theorem innerCandidates429_8_checked : candidates edge 4 3011 7349 innerTail429_9 best.bound = innerCandidates429_8 := by
  unfold candidates
  have row : edge 7349 = SearchData.row7349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan429_8 : scan edge (search edge 3) 4 [3011] 3011 innerTail429_8 best = best := by
  rw [innerTail429_8, scan, if_neg (by decide), innerCandidates429_8_checked]
  change scan edge (search edge 3) 4 [3011] 3011 innerTail429_9
    (if 3 ≤ innerCandidates429_8.length then search edge 3 [3011, 7349] innerCandidates429_8 10360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan429_9

def innerTail429_7 : List Nat := 7127 :: innerTail429_8
def innerCandidates429_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates429_7_checked : candidates edge 4 3011 7127 innerTail429_8 best.bound = innerCandidates429_7 := by
  unfold candidates
  have row : edge 7127 = SearchData.row7127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan429_7 : scan edge (search edge 3) 4 [3011] 3011 innerTail429_7 best = best := by
  rw [innerTail429_7, scan, if_neg (by decide), innerCandidates429_7_checked]
  change scan edge (search edge 3) 4 [3011] 3011 innerTail429_8
    (if 3 ≤ innerCandidates429_7.length then search edge 3 [3011, 7127] innerCandidates429_7 10138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan429_8

def innerTail429_6 : List Nat := 6833 :: innerTail429_7
def innerCandidates429_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates429_6_checked : candidates edge 4 3011 6833 innerTail429_7 best.bound = innerCandidates429_6 := by
  unfold candidates
  have row : edge 6833 = SearchData.row6833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan429_6 : scan edge (search edge 3) 4 [3011] 3011 innerTail429_6 best = best := by
  rw [innerTail429_6, scan, if_neg (by decide), innerCandidates429_6_checked]
  change scan edge (search edge 3) 4 [3011] 3011 innerTail429_7
    (if 3 ≤ innerCandidates429_6.length then search edge 3 [3011, 6833] innerCandidates429_6 9844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan429_7

def innerTail429_5 : List Nat := 6353 :: innerTail429_6
def innerCandidates429_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates429_5_checked : candidates edge 4 3011 6353 innerTail429_6 best.bound = innerCandidates429_5 := by
  unfold candidates
  have row : edge 6353 = SearchData.row6353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan429_5 : scan edge (search edge 3) 4 [3011] 3011 innerTail429_5 best = best := by
  rw [innerTail429_5, scan, if_neg (by decide), innerCandidates429_5_checked]
  change scan edge (search edge 3) 4 [3011] 3011 innerTail429_6
    (if 3 ≤ innerCandidates429_5.length then search edge 3 [3011, 6353] innerCandidates429_5 9364 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan429_6

def innerTail429_4 : List Nat := 6329 :: innerTail429_5
def innerCandidates429_4 : List Nat := [7937]
set_option maxRecDepth 4096 in
theorem innerCandidates429_4_checked : candidates edge 4 3011 6329 innerTail429_5 best.bound = innerCandidates429_4 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan429_4 : scan edge (search edge 3) 4 [3011] 3011 innerTail429_4 best = best := by
  rw [innerTail429_4, scan, if_neg (by decide), innerCandidates429_4_checked]
  change scan edge (search edge 3) 4 [3011] 3011 innerTail429_5
    (if 3 ≤ innerCandidates429_4.length then search edge 3 [3011, 6329] innerCandidates429_4 9340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan429_5

def innerTail429_3 : List Nat := 4817 :: innerTail429_4
def innerCandidates429_3 : List Nat := [7127]
set_option maxRecDepth 4096 in
theorem innerCandidates429_3_checked : candidates edge 4 3011 4817 innerTail429_4 best.bound = innerCandidates429_3 := by
  unfold candidates
  have row : edge 4817 = SearchData.row4817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan429_3 : scan edge (search edge 3) 4 [3011] 3011 innerTail429_3 best = best := by
  rw [innerTail429_3, scan, if_neg (by decide), innerCandidates429_3_checked]
  change scan edge (search edge 3) 4 [3011] 3011 innerTail429_4
    (if 3 ≤ innerCandidates429_3.length then search edge 3 [3011, 4817] innerCandidates429_3 7828 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan429_4

def innerTail429_2 : List Nat := 4493 :: innerTail429_3
def innerCandidates429_2 : List Nat := [7349]
set_option maxRecDepth 4096 in
theorem innerCandidates429_2_checked : candidates edge 4 3011 4493 innerTail429_3 best.bound = innerCandidates429_2 := by
  unfold candidates
  have row : edge 4493 = SearchData.row4493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan429_2 : scan edge (search edge 3) 4 [3011] 3011 innerTail429_2 best = best := by
  rw [innerTail429_2, scan, if_neg (by decide), innerCandidates429_2_checked]
  change scan edge (search edge 3) 4 [3011] 3011 innerTail429_3
    (if 3 ≤ innerCandidates429_2.length then search edge 3 [3011, 4493] innerCandidates429_2 7504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan429_3

def innerTail429_1 : List Nat := 3329 :: innerTail429_2
def innerCandidates429_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates429_1_checked : candidates edge 4 3011 3329 innerTail429_2 best.bound = innerCandidates429_1 := by
  unfold candidates
  have row : edge 3329 = SearchData.row3329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan429_1 : scan edge (search edge 3) 4 [3011] 3011 innerTail429_1 best = best := by
  rw [innerTail429_1, scan, if_neg (by decide), innerCandidates429_1_checked]
  change scan edge (search edge 3) 4 [3011] 3011 innerTail429_2
    (if 3 ≤ innerCandidates429_1.length then search edge 3 [3011, 3329] innerCandidates429_1 6340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan429_2

def innerTail429_0 : List Nat := 3023 :: innerTail429_1
def innerCandidates429_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates429_0_checked : candidates edge 4 3011 3023 innerTail429_1 best.bound = innerCandidates429_0 := by
  unfold candidates
  have row : edge 3023 = SearchData.row3023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan429_0 : scan edge (search edge 3) 4 [3011] 3011 innerTail429_0 best = best := by
  rw [innerTail429_0, scan, if_neg (by decide), innerCandidates429_0_checked]
  change scan edge (search edge 3) 4 [3011] 3011 innerTail429_1
    (if 3 ≤ innerCandidates429_0.length then search edge 3 [3011, 3023] innerCandidates429_0 6034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan429_1

theorem rootBranch429 : search edge 4 [3011] rootCandidates429 3011 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3011] 3011 innerTail429_0 best = best
  exact innerScan429_0

def rootCandidates428 : List Nat := [3163, 3391, 3499, 3727, 3967, 4021, 4219, 4591, 4909, 4951, 4969, 5749, 6121, 6607, 6733, 7069, 7573, 7621, 7789]
set_option maxRecDepth 4096 in
theorem rootCandidates428_checked : candidates edge 5 0 3001 tail429 best.bound = rootCandidates428 := by
  unfold candidates
  have row : edge 3001 = SearchData.row3001.contains := by funext q; rfl
  rw [row]
  decide

def innerTail428_19 : List Nat := []
theorem innerScan428_19 : scan edge (search edge 3) 4 [3001] 3001 innerTail428_19 best = best := rfl

def innerTail428_18 : List Nat := 7789 :: innerTail428_19
def innerCandidates428_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates428_18_checked : candidates edge 4 3001 7789 innerTail428_19 best.bound = innerCandidates428_18 := by
  unfold candidates
  have row : edge 7789 = SearchData.row7789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan428_18 : scan edge (search edge 3) 4 [3001] 3001 innerTail428_18 best = best := by
  rw [innerTail428_18, scan, if_neg (by decide), innerCandidates428_18_checked]
  change scan edge (search edge 3) 4 [3001] 3001 innerTail428_19
    (if 3 ≤ innerCandidates428_18.length then search edge 3 [3001, 7789] innerCandidates428_18 10790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan428_19

def innerTail428_17 : List Nat := 7621 :: innerTail428_18
def innerCandidates428_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates428_17_checked : candidates edge 4 3001 7621 innerTail428_18 best.bound = innerCandidates428_17 := by
  unfold candidates
  have row : edge 7621 = SearchData.row7621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan428_17 : scan edge (search edge 3) 4 [3001] 3001 innerTail428_17 best = best := by
  rw [innerTail428_17, scan, if_neg (by decide), innerCandidates428_17_checked]
  change scan edge (search edge 3) 4 [3001] 3001 innerTail428_18
    (if 3 ≤ innerCandidates428_17.length then search edge 3 [3001, 7621] innerCandidates428_17 10622 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan428_18

def innerTail428_16 : List Nat := 7573 :: innerTail428_17
def innerCandidates428_16 : List Nat := [7621]
set_option maxRecDepth 4096 in
theorem innerCandidates428_16_checked : candidates edge 4 3001 7573 innerTail428_17 best.bound = innerCandidates428_16 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan428_16 : scan edge (search edge 3) 4 [3001] 3001 innerTail428_16 best = best := by
  rw [innerTail428_16, scan, if_neg (by decide), innerCandidates428_16_checked]
  change scan edge (search edge 3) 4 [3001] 3001 innerTail428_17
    (if 3 ≤ innerCandidates428_16.length then search edge 3 [3001, 7573] innerCandidates428_16 10574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan428_17

def innerTail428_15 : List Nat := 7069 :: innerTail428_16
def innerCandidates428_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates428_15_checked : candidates edge 4 3001 7069 innerTail428_16 best.bound = innerCandidates428_15 := by
  unfold candidates
  have row : edge 7069 = SearchData.row7069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan428_15 : scan edge (search edge 3) 4 [3001] 3001 innerTail428_15 best = best := by
  rw [innerTail428_15, scan, if_neg (by decide), innerCandidates428_15_checked]
  change scan edge (search edge 3) 4 [3001] 3001 innerTail428_16
    (if 3 ≤ innerCandidates428_15.length then search edge 3 [3001, 7069] innerCandidates428_15 10070 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan428_16

def innerTail428_14 : List Nat := 6733 :: innerTail428_15
def innerCandidates428_14 : List Nat := [7789]
set_option maxRecDepth 4096 in
theorem innerCandidates428_14_checked : candidates edge 4 3001 6733 innerTail428_15 best.bound = innerCandidates428_14 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan428_14 : scan edge (search edge 3) 4 [3001] 3001 innerTail428_14 best = best := by
  rw [innerTail428_14, scan, if_neg (by decide), innerCandidates428_14_checked]
  change scan edge (search edge 3) 4 [3001] 3001 innerTail428_15
    (if 3 ≤ innerCandidates428_14.length then search edge 3 [3001, 6733] innerCandidates428_14 9734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan428_15

def innerTail428_13 : List Nat := 6607 :: innerTail428_14
def innerCandidates428_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates428_13_checked : candidates edge 4 3001 6607 innerTail428_14 best.bound = innerCandidates428_13 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan428_13 : scan edge (search edge 3) 4 [3001] 3001 innerTail428_13 best = best := by
  rw [innerTail428_13, scan, if_neg (by decide), innerCandidates428_13_checked]
  change scan edge (search edge 3) 4 [3001] 3001 innerTail428_14
    (if 3 ≤ innerCandidates428_13.length then search edge 3 [3001, 6607] innerCandidates428_13 9608 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan428_14

def innerTail428_12 : List Nat := 6121 :: innerTail428_13
def innerCandidates428_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates428_12_checked : candidates edge 4 3001 6121 innerTail428_13 best.bound = innerCandidates428_12 := by
  unfold candidates
  have row : edge 6121 = SearchData.row6121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan428_12 : scan edge (search edge 3) 4 [3001] 3001 innerTail428_12 best = best := by
  rw [innerTail428_12, scan, if_neg (by decide), innerCandidates428_12_checked]
  change scan edge (search edge 3) 4 [3001] 3001 innerTail428_13
    (if 3 ≤ innerCandidates428_12.length then search edge 3 [3001, 6121] innerCandidates428_12 9122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan428_13

def innerTail428_11 : List Nat := 5749 :: innerTail428_12
def innerCandidates428_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates428_11_checked : candidates edge 4 3001 5749 innerTail428_12 best.bound = innerCandidates428_11 := by
  unfold candidates
  have row : edge 5749 = SearchData.row5749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan428_11 : scan edge (search edge 3) 4 [3001] 3001 innerTail428_11 best = best := by
  rw [innerTail428_11, scan, if_neg (by decide), innerCandidates428_11_checked]
  change scan edge (search edge 3) 4 [3001] 3001 innerTail428_12
    (if 3 ≤ innerCandidates428_11.length then search edge 3 [3001, 5749] innerCandidates428_11 8750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan428_12

def innerTail428_10 : List Nat := 4969 :: innerTail428_11
def innerCandidates428_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates428_10_checked : candidates edge 4 3001 4969 innerTail428_11 best.bound = innerCandidates428_10 := by
  unfold candidates
  have row : edge 4969 = SearchData.row4969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan428_10 : scan edge (search edge 3) 4 [3001] 3001 innerTail428_10 best = best := by
  rw [innerTail428_10, scan, if_neg (by decide), innerCandidates428_10_checked]
  change scan edge (search edge 3) 4 [3001] 3001 innerTail428_11
    (if 3 ≤ innerCandidates428_10.length then search edge 3 [3001, 4969] innerCandidates428_10 7970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan428_11

def innerTail428_9 : List Nat := 4951 :: innerTail428_10
def innerCandidates428_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates428_9_checked : candidates edge 4 3001 4951 innerTail428_10 best.bound = innerCandidates428_9 := by
  unfold candidates
  have row : edge 4951 = SearchData.row4951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan428_9 : scan edge (search edge 3) 4 [3001] 3001 innerTail428_9 best = best := by
  rw [innerTail428_9, scan, if_neg (by decide), innerCandidates428_9_checked]
  change scan edge (search edge 3) 4 [3001] 3001 innerTail428_10
    (if 3 ≤ innerCandidates428_9.length then search edge 3 [3001, 4951] innerCandidates428_9 7952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan428_10

def innerTail428_8 : List Nat := 4909 :: innerTail428_9
def innerCandidates428_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates428_8_checked : candidates edge 4 3001 4909 innerTail428_9 best.bound = innerCandidates428_8 := by
  unfold candidates
  have row : edge 4909 = SearchData.row4909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan428_8 : scan edge (search edge 3) 4 [3001] 3001 innerTail428_8 best = best := by
  rw [innerTail428_8, scan, if_neg (by decide), innerCandidates428_8_checked]
  change scan edge (search edge 3) 4 [3001] 3001 innerTail428_9
    (if 3 ≤ innerCandidates428_8.length then search edge 3 [3001, 4909] innerCandidates428_8 7910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan428_9

def innerTail428_7 : List Nat := 4591 :: innerTail428_8
def innerCandidates428_7 : List Nat := [6121, 7621]
set_option maxRecDepth 4096 in
theorem innerCandidates428_7_checked : candidates edge 4 3001 4591 innerTail428_8 best.bound = innerCandidates428_7 := by
  unfold candidates
  have row : edge 4591 = SearchData.row4591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan428_7 : scan edge (search edge 3) 4 [3001] 3001 innerTail428_7 best = best := by
  rw [innerTail428_7, scan, if_neg (by decide), innerCandidates428_7_checked]
  change scan edge (search edge 3) 4 [3001] 3001 innerTail428_8
    (if 3 ≤ innerCandidates428_7.length then search edge 3 [3001, 4591] innerCandidates428_7 7592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan428_8

def innerTail428_6 : List Nat := 4219 :: innerTail428_7
def innerCandidates428_6 : List Nat := [5749]
set_option maxRecDepth 4096 in
theorem innerCandidates428_6_checked : candidates edge 4 3001 4219 innerTail428_7 best.bound = innerCandidates428_6 := by
  unfold candidates
  have row : edge 4219 = SearchData.row4219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan428_6 : scan edge (search edge 3) 4 [3001] 3001 innerTail428_6 best = best := by
  rw [innerTail428_6, scan, if_neg (by decide), innerCandidates428_6_checked]
  change scan edge (search edge 3) 4 [3001] 3001 innerTail428_7
    (if 3 ≤ innerCandidates428_6.length then search edge 3 [3001, 4219] innerCandidates428_6 7220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan428_7

def innerTail428_5 : List Nat := 4021 :: innerTail428_6
def innerCandidates428_5 : List Nat := [6733]
set_option maxRecDepth 4096 in
theorem innerCandidates428_5_checked : candidates edge 4 3001 4021 innerTail428_6 best.bound = innerCandidates428_5 := by
  unfold candidates
  have row : edge 4021 = SearchData.row4021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan428_5 : scan edge (search edge 3) 4 [3001] 3001 innerTail428_5 best = best := by
  rw [innerTail428_5, scan, if_neg (by decide), innerCandidates428_5_checked]
  change scan edge (search edge 3) 4 [3001] 3001 innerTail428_6
    (if 3 ≤ innerCandidates428_5.length then search edge 3 [3001, 4021] innerCandidates428_5 7022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan428_6

def innerTail428_4 : List Nat := 3967 :: innerTail428_5
def innerCandidates428_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates428_4_checked : candidates edge 4 3001 3967 innerTail428_5 best.bound = innerCandidates428_4 := by
  unfold candidates
  have row : edge 3967 = SearchData.row3967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan428_4 : scan edge (search edge 3) 4 [3001] 3001 innerTail428_4 best = best := by
  rw [innerTail428_4, scan, if_neg (by decide), innerCandidates428_4_checked]
  change scan edge (search edge 3) 4 [3001] 3001 innerTail428_5
    (if 3 ≤ innerCandidates428_4.length then search edge 3 [3001, 3967] innerCandidates428_4 6968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan428_5

def innerTail428_3 : List Nat := 3727 :: innerTail428_4
def innerCandidates428_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates428_3_checked : candidates edge 4 3001 3727 innerTail428_4 best.bound = innerCandidates428_3 := by
  unfold candidates
  have row : edge 3727 = SearchData.row3727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan428_3 : scan edge (search edge 3) 4 [3001] 3001 innerTail428_3 best = best := by
  rw [innerTail428_3, scan, if_neg (by decide), innerCandidates428_3_checked]
  change scan edge (search edge 3) 4 [3001] 3001 innerTail428_4
    (if 3 ≤ innerCandidates428_3.length then search edge 3 [3001, 3727] innerCandidates428_3 6728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan428_4

def innerTail428_2 : List Nat := 3499 :: innerTail428_3
def innerCandidates428_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates428_2_checked : candidates edge 4 3001 3499 innerTail428_3 best.bound = innerCandidates428_2 := by
  unfold candidates
  have row : edge 3499 = SearchData.row3499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan428_2 : scan edge (search edge 3) 4 [3001] 3001 innerTail428_2 best = best := by
  rw [innerTail428_2, scan, if_neg (by decide), innerCandidates428_2_checked]
  change scan edge (search edge 3) 4 [3001] 3001 innerTail428_3
    (if 3 ≤ innerCandidates428_2.length then search edge 3 [3001, 3499] innerCandidates428_2 6500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan428_3

def innerTail428_1 : List Nat := 3391 :: innerTail428_2
def innerCandidates428_1 : List Nat := [6607, 6733]
set_option maxRecDepth 4096 in
theorem innerCandidates428_1_checked : candidates edge 4 3001 3391 innerTail428_2 best.bound = innerCandidates428_1 := by
  unfold candidates
  have row : edge 3391 = SearchData.row3391.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan428_1 : scan edge (search edge 3) 4 [3001] 3001 innerTail428_1 best = best := by
  rw [innerTail428_1, scan, if_neg (by decide), innerCandidates428_1_checked]
  change scan edge (search edge 3) 4 [3001] 3001 innerTail428_2
    (if 3 ≤ innerCandidates428_1.length then search edge 3 [3001, 3391] innerCandidates428_1 6392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan428_2

def innerTail428_0 : List Nat := 3163 :: innerTail428_1
def innerCandidates428_0 : List Nat := [3967, 6733]
set_option maxRecDepth 4096 in
theorem innerCandidates428_0_checked : candidates edge 4 3001 3163 innerTail428_1 best.bound = innerCandidates428_0 := by
  unfold candidates
  have row : edge 3163 = SearchData.row3163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan428_0 : scan edge (search edge 3) 4 [3001] 3001 innerTail428_0 best = best := by
  rw [innerTail428_0, scan, if_neg (by decide), innerCandidates428_0_checked]
  change scan edge (search edge 3) 4 [3001] 3001 innerTail428_1
    (if 3 ≤ innerCandidates428_0.length then search edge 3 [3001, 3163] innerCandidates428_0 6164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan428_1

theorem rootBranch428 : search edge 4 [3001] rootCandidates428 3001 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3001] 3001 innerTail428_0 best = best
  exact innerScan428_0

def rootCandidates427 : List Nat := [3371, 3413, 3881, 4073, 4259, 4373, 4871, 5003, 6311, 7349, 7481, 7577, 7907]
set_option maxRecDepth 4096 in
theorem rootCandidates427_checked : candidates edge 5 0 2999 tail428 best.bound = rootCandidates427 := by
  unfold candidates
  have row : edge 2999 = SearchData.row2999.contains := by funext q; rfl
  rw [row]
  decide

def innerTail427_13 : List Nat := []
theorem innerScan427_13 : scan edge (search edge 3) 4 [2999] 2999 innerTail427_13 best = best := rfl

def innerTail427_12 : List Nat := 7907 :: innerTail427_13
def innerCandidates427_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates427_12_checked : candidates edge 4 2999 7907 innerTail427_13 best.bound = innerCandidates427_12 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan427_12 : scan edge (search edge 3) 4 [2999] 2999 innerTail427_12 best = best := by
  rw [innerTail427_12, scan, if_neg (by decide), innerCandidates427_12_checked]
  change scan edge (search edge 3) 4 [2999] 2999 innerTail427_13
    (if 3 ≤ innerCandidates427_12.length then search edge 3 [2999, 7907] innerCandidates427_12 10906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan427_13

def innerTail427_11 : List Nat := 7577 :: innerTail427_12
def innerCandidates427_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates427_11_checked : candidates edge 4 2999 7577 innerTail427_12 best.bound = innerCandidates427_11 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan427_11 : scan edge (search edge 3) 4 [2999] 2999 innerTail427_11 best = best := by
  rw [innerTail427_11, scan, if_neg (by decide), innerCandidates427_11_checked]
  change scan edge (search edge 3) 4 [2999] 2999 innerTail427_12
    (if 3 ≤ innerCandidates427_11.length then search edge 3 [2999, 7577] innerCandidates427_11 10576 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan427_12

def innerTail427_10 : List Nat := 7481 :: innerTail427_11
def innerCandidates427_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates427_10_checked : candidates edge 4 2999 7481 innerTail427_11 best.bound = innerCandidates427_10 := by
  unfold candidates
  have row : edge 7481 = SearchData.row7481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan427_10 : scan edge (search edge 3) 4 [2999] 2999 innerTail427_10 best = best := by
  rw [innerTail427_10, scan, if_neg (by decide), innerCandidates427_10_checked]
  change scan edge (search edge 3) 4 [2999] 2999 innerTail427_11
    (if 3 ≤ innerCandidates427_10.length then search edge 3 [2999, 7481] innerCandidates427_10 10480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan427_11

def innerTail427_9 : List Nat := 7349 :: innerTail427_10
def innerCandidates427_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates427_9_checked : candidates edge 4 2999 7349 innerTail427_10 best.bound = innerCandidates427_9 := by
  unfold candidates
  have row : edge 7349 = SearchData.row7349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan427_9 : scan edge (search edge 3) 4 [2999] 2999 innerTail427_9 best = best := by
  rw [innerTail427_9, scan, if_neg (by decide), innerCandidates427_9_checked]
  change scan edge (search edge 3) 4 [2999] 2999 innerTail427_10
    (if 3 ≤ innerCandidates427_9.length then search edge 3 [2999, 7349] innerCandidates427_9 10348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan427_10

def innerTail427_8 : List Nat := 6311 :: innerTail427_9
def innerCandidates427_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates427_8_checked : candidates edge 4 2999 6311 innerTail427_9 best.bound = innerCandidates427_8 := by
  unfold candidates
  have row : edge 6311 = SearchData.row6311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan427_8 : scan edge (search edge 3) 4 [2999] 2999 innerTail427_8 best = best := by
  rw [innerTail427_8, scan, if_neg (by decide), innerCandidates427_8_checked]
  change scan edge (search edge 3) 4 [2999] 2999 innerTail427_9
    (if 3 ≤ innerCandidates427_8.length then search edge 3 [2999, 6311] innerCandidates427_8 9310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan427_9

def innerTail427_7 : List Nat := 5003 :: innerTail427_8
def innerCandidates427_7 : List Nat := [7349]
set_option maxRecDepth 4096 in
theorem innerCandidates427_7_checked : candidates edge 4 2999 5003 innerTail427_8 best.bound = innerCandidates427_7 := by
  unfold candidates
  have row : edge 5003 = SearchData.row5003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan427_7 : scan edge (search edge 3) 4 [2999] 2999 innerTail427_7 best = best := by
  rw [innerTail427_7, scan, if_neg (by decide), innerCandidates427_7_checked]
  change scan edge (search edge 3) 4 [2999] 2999 innerTail427_8
    (if 3 ≤ innerCandidates427_7.length then search edge 3 [2999, 5003] innerCandidates427_7 8002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan427_8

def innerTail427_6 : List Nat := 4871 :: innerTail427_7
def innerCandidates427_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates427_6_checked : candidates edge 4 2999 4871 innerTail427_7 best.bound = innerCandidates427_6 := by
  unfold candidates
  have row : edge 4871 = SearchData.row4871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan427_6 : scan edge (search edge 3) 4 [2999] 2999 innerTail427_6 best = best := by
  rw [innerTail427_6, scan, if_neg (by decide), innerCandidates427_6_checked]
  change scan edge (search edge 3) 4 [2999] 2999 innerTail427_7
    (if 3 ≤ innerCandidates427_6.length then search edge 3 [2999, 4871] innerCandidates427_6 7870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan427_7

def innerTail427_5 : List Nat := 4373 :: innerTail427_6
def innerCandidates427_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates427_5_checked : candidates edge 4 2999 4373 innerTail427_6 best.bound = innerCandidates427_5 := by
  unfold candidates
  have row : edge 4373 = SearchData.row4373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan427_5 : scan edge (search edge 3) 4 [2999] 2999 innerTail427_5 best = best := by
  rw [innerTail427_5, scan, if_neg (by decide), innerCandidates427_5_checked]
  change scan edge (search edge 3) 4 [2999] 2999 innerTail427_6
    (if 3 ≤ innerCandidates427_5.length then search edge 3 [2999, 4373] innerCandidates427_5 7372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan427_6

def innerTail427_4 : List Nat := 4259 :: innerTail427_5
def innerCandidates427_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates427_4_checked : candidates edge 4 2999 4259 innerTail427_5 best.bound = innerCandidates427_4 := by
  unfold candidates
  have row : edge 4259 = SearchData.row4259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan427_4 : scan edge (search edge 3) 4 [2999] 2999 innerTail427_4 best = best := by
  rw [innerTail427_4, scan, if_neg (by decide), innerCandidates427_4_checked]
  change scan edge (search edge 3) 4 [2999] 2999 innerTail427_5
    (if 3 ≤ innerCandidates427_4.length then search edge 3 [2999, 4259] innerCandidates427_4 7258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan427_5

def innerTail427_3 : List Nat := 4073 :: innerTail427_4
def innerCandidates427_3 : List Nat := [5003]
set_option maxRecDepth 4096 in
theorem innerCandidates427_3_checked : candidates edge 4 2999 4073 innerTail427_4 best.bound = innerCandidates427_3 := by
  unfold candidates
  have row : edge 4073 = SearchData.row4073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan427_3 : scan edge (search edge 3) 4 [2999] 2999 innerTail427_3 best = best := by
  rw [innerTail427_3, scan, if_neg (by decide), innerCandidates427_3_checked]
  change scan edge (search edge 3) 4 [2999] 2999 innerTail427_4
    (if 3 ≤ innerCandidates427_3.length then search edge 3 [2999, 4073] innerCandidates427_3 7072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan427_4

def innerTail427_2 : List Nat := 3881 :: innerTail427_3
def innerCandidates427_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates427_2_checked : candidates edge 4 2999 3881 innerTail427_3 best.bound = innerCandidates427_2 := by
  unfold candidates
  have row : edge 3881 = SearchData.row3881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan427_2 : scan edge (search edge 3) 4 [2999] 2999 innerTail427_2 best = best := by
  rw [innerTail427_2, scan, if_neg (by decide), innerCandidates427_2_checked]
  change scan edge (search edge 3) 4 [2999] 2999 innerTail427_3
    (if 3 ≤ innerCandidates427_2.length then search edge 3 [2999, 3881] innerCandidates427_2 6880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan427_3

def innerTail427_1 : List Nat := 3413 :: innerTail427_2
def innerCandidates427_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates427_1_checked : candidates edge 4 2999 3413 innerTail427_2 best.bound = innerCandidates427_1 := by
  unfold candidates
  have row : edge 3413 = SearchData.row3413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan427_1 : scan edge (search edge 3) 4 [2999] 2999 innerTail427_1 best = best := by
  rw [innerTail427_1, scan, if_neg (by decide), innerCandidates427_1_checked]
  change scan edge (search edge 3) 4 [2999] 2999 innerTail427_2
    (if 3 ≤ innerCandidates427_1.length then search edge 3 [2999, 3413] innerCandidates427_1 6412 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan427_2

def innerTail427_0 : List Nat := 3371 :: innerTail427_1
def innerCandidates427_0 : List Nat := [3413]
set_option maxRecDepth 4096 in
theorem innerCandidates427_0_checked : candidates edge 4 2999 3371 innerTail427_1 best.bound = innerCandidates427_0 := by
  unfold candidates
  have row : edge 3371 = SearchData.row3371.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan427_0 : scan edge (search edge 3) 4 [2999] 2999 innerTail427_0 best = best := by
  rw [innerTail427_0, scan, if_neg (by decide), innerCandidates427_0_checked]
  change scan edge (search edge 3) 4 [2999] 2999 innerTail427_1
    (if 3 ≤ innerCandidates427_0.length then search edge 3 [2999, 3371] innerCandidates427_0 6370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan427_1

theorem rootBranch427 : search edge 4 [2999] rootCandidates427 2999 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2999] 2999 innerTail427_0 best = best
  exact innerScan427_0

def rootCandidates426 : List Nat := [3037, 3709, 4057, 4273, 4639, 5557, 5953, 6067, 6871, 6961, 7459, 7573]
set_option maxRecDepth 4096 in
theorem rootCandidates426_checked : candidates edge 5 0 2971 tail427 best.bound = rootCandidates426 := by
  unfold candidates
  have row : edge 2971 = SearchData.row2971.contains := by funext q; rfl
  rw [row]
  decide

def innerTail426_12 : List Nat := []
theorem innerScan426_12 : scan edge (search edge 3) 4 [2971] 2971 innerTail426_12 best = best := rfl

def innerTail426_11 : List Nat := 7573 :: innerTail426_12
def innerCandidates426_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates426_11_checked : candidates edge 4 2971 7573 innerTail426_12 best.bound = innerCandidates426_11 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan426_11 : scan edge (search edge 3) 4 [2971] 2971 innerTail426_11 best = best := by
  rw [innerTail426_11, scan, if_neg (by decide), innerCandidates426_11_checked]
  change scan edge (search edge 3) 4 [2971] 2971 innerTail426_12
    (if 3 ≤ innerCandidates426_11.length then search edge 3 [2971, 7573] innerCandidates426_11 10544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan426_12

def innerTail426_10 : List Nat := 7459 :: innerTail426_11
def innerCandidates426_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates426_10_checked : candidates edge 4 2971 7459 innerTail426_11 best.bound = innerCandidates426_10 := by
  unfold candidates
  have row : edge 7459 = SearchData.row7459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan426_10 : scan edge (search edge 3) 4 [2971] 2971 innerTail426_10 best = best := by
  rw [innerTail426_10, scan, if_neg (by decide), innerCandidates426_10_checked]
  change scan edge (search edge 3) 4 [2971] 2971 innerTail426_11
    (if 3 ≤ innerCandidates426_10.length then search edge 3 [2971, 7459] innerCandidates426_10 10430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan426_11

def innerTail426_9 : List Nat := 6961 :: innerTail426_10
def innerCandidates426_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates426_9_checked : candidates edge 4 2971 6961 innerTail426_10 best.bound = innerCandidates426_9 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan426_9 : scan edge (search edge 3) 4 [2971] 2971 innerTail426_9 best = best := by
  rw [innerTail426_9, scan, if_neg (by decide), innerCandidates426_9_checked]
  change scan edge (search edge 3) 4 [2971] 2971 innerTail426_10
    (if 3 ≤ innerCandidates426_9.length then search edge 3 [2971, 6961] innerCandidates426_9 9932 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan426_10

def innerTail426_8 : List Nat := 6871 :: innerTail426_9
def innerCandidates426_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates426_8_checked : candidates edge 4 2971 6871 innerTail426_9 best.bound = innerCandidates426_8 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan426_8 : scan edge (search edge 3) 4 [2971] 2971 innerTail426_8 best = best := by
  rw [innerTail426_8, scan, if_neg (by decide), innerCandidates426_8_checked]
  change scan edge (search edge 3) 4 [2971] 2971 innerTail426_9
    (if 3 ≤ innerCandidates426_8.length then search edge 3 [2971, 6871] innerCandidates426_8 9842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan426_9

def innerTail426_7 : List Nat := 6067 :: innerTail426_8
def innerCandidates426_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates426_7_checked : candidates edge 4 2971 6067 innerTail426_8 best.bound = innerCandidates426_7 := by
  unfold candidates
  have row : edge 6067 = SearchData.row6067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan426_7 : scan edge (search edge 3) 4 [2971] 2971 innerTail426_7 best = best := by
  rw [innerTail426_7, scan, if_neg (by decide), innerCandidates426_7_checked]
  change scan edge (search edge 3) 4 [2971] 2971 innerTail426_8
    (if 3 ≤ innerCandidates426_7.length then search edge 3 [2971, 6067] innerCandidates426_7 9038 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan426_8

def innerTail426_6 : List Nat := 5953 :: innerTail426_7
def innerCandidates426_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates426_6_checked : candidates edge 4 2971 5953 innerTail426_7 best.bound = innerCandidates426_6 := by
  unfold candidates
  have row : edge 5953 = SearchData.row5953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan426_6 : scan edge (search edge 3) 4 [2971] 2971 innerTail426_6 best = best := by
  rw [innerTail426_6, scan, if_neg (by decide), innerCandidates426_6_checked]
  change scan edge (search edge 3) 4 [2971] 2971 innerTail426_7
    (if 3 ≤ innerCandidates426_6.length then search edge 3 [2971, 5953] innerCandidates426_6 8924 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan426_7

def innerTail426_5 : List Nat := 5557 :: innerTail426_6
def innerCandidates426_5 : List Nat := [6067]
set_option maxRecDepth 4096 in
theorem innerCandidates426_5_checked : candidates edge 4 2971 5557 innerTail426_6 best.bound = innerCandidates426_5 := by
  unfold candidates
  have row : edge 5557 = SearchData.row5557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan426_5 : scan edge (search edge 3) 4 [2971] 2971 innerTail426_5 best = best := by
  rw [innerTail426_5, scan, if_neg (by decide), innerCandidates426_5_checked]
  change scan edge (search edge 3) 4 [2971] 2971 innerTail426_6
    (if 3 ≤ innerCandidates426_5.length then search edge 3 [2971, 5557] innerCandidates426_5 8528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan426_6

def innerTail426_4 : List Nat := 4639 :: innerTail426_5
def innerCandidates426_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates426_4_checked : candidates edge 4 2971 4639 innerTail426_5 best.bound = innerCandidates426_4 := by
  unfold candidates
  have row : edge 4639 = SearchData.row4639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan426_4 : scan edge (search edge 3) 4 [2971] 2971 innerTail426_4 best = best := by
  rw [innerTail426_4, scan, if_neg (by decide), innerCandidates426_4_checked]
  change scan edge (search edge 3) 4 [2971] 2971 innerTail426_5
    (if 3 ≤ innerCandidates426_4.length then search edge 3 [2971, 4639] innerCandidates426_4 7610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan426_5

def innerTail426_3 : List Nat := 4273 :: innerTail426_4
def innerCandidates426_3 : List Nat := [7573]
set_option maxRecDepth 4096 in
theorem innerCandidates426_3_checked : candidates edge 4 2971 4273 innerTail426_4 best.bound = innerCandidates426_3 := by
  unfold candidates
  have row : edge 4273 = SearchData.row4273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan426_3 : scan edge (search edge 3) 4 [2971] 2971 innerTail426_3 best = best := by
  rw [innerTail426_3, scan, if_neg (by decide), innerCandidates426_3_checked]
  change scan edge (search edge 3) 4 [2971] 2971 innerTail426_4
    (if 3 ≤ innerCandidates426_3.length then search edge 3 [2971, 4273] innerCandidates426_3 7244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan426_4

def innerTail426_2 : List Nat := 4057 :: innerTail426_3
def innerCandidates426_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates426_2_checked : candidates edge 4 2971 4057 innerTail426_3 best.bound = innerCandidates426_2 := by
  unfold candidates
  have row : edge 4057 = SearchData.row4057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan426_2 : scan edge (search edge 3) 4 [2971] 2971 innerTail426_2 best = best := by
  rw [innerTail426_2, scan, if_neg (by decide), innerCandidates426_2_checked]
  change scan edge (search edge 3) 4 [2971] 2971 innerTail426_3
    (if 3 ≤ innerCandidates426_2.length then search edge 3 [2971, 4057] innerCandidates426_2 7028 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan426_3

def innerTail426_1 : List Nat := 3709 :: innerTail426_2
def innerCandidates426_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates426_1_checked : candidates edge 4 2971 3709 innerTail426_2 best.bound = innerCandidates426_1 := by
  unfold candidates
  have row : edge 3709 = SearchData.row3709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan426_1 : scan edge (search edge 3) 4 [2971] 2971 innerTail426_1 best = best := by
  rw [innerTail426_1, scan, if_neg (by decide), innerCandidates426_1_checked]
  change scan edge (search edge 3) 4 [2971] 2971 innerTail426_2
    (if 3 ≤ innerCandidates426_1.length then search edge 3 [2971, 3709] innerCandidates426_1 6680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan426_2

def innerTail426_0 : List Nat := 3037 :: innerTail426_1
def innerCandidates426_0 : List Nat := [3709]
set_option maxRecDepth 4096 in
theorem innerCandidates426_0_checked : candidates edge 4 2971 3037 innerTail426_1 best.bound = innerCandidates426_0 := by
  unfold candidates
  have row : edge 3037 = SearchData.row3037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan426_0 : scan edge (search edge 3) 4 [2971] 2971 innerTail426_0 best = best := by
  rw [innerTail426_0, scan, if_neg (by decide), innerCandidates426_0_checked]
  change scan edge (search edge 3) 4 [2971] 2971 innerTail426_1
    (if 3 ≤ innerCandidates426_0.length then search edge 3 [2971, 3037] innerCandidates426_0 6008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan426_1

theorem rootBranch426 : search edge 4 [2971] rootCandidates426 2971 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2971] 2971 innerTail426_0 best = best
  exact innerScan426_0

end ProjectEulerSolutions.PrimePairUpper.Search
