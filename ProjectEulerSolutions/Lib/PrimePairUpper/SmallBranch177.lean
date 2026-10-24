import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates154 : List Nat := [937, 1069, 1201, 1279, 1627, 2017, 2749, 3463, 3643, 4231, 4441, 4567, 4597, 5167, 5821, 5839, 5881, 6679, 6961, 7039, 7207]
set_option maxRecDepth 4096 in
theorem rootCandidates154_checked : candidates edge 5 0 919 tail155 best.bound = rootCandidates154 := by
  unfold candidates
  have row : edge 919 = PrimePairCache.smallRow919.contains := by funext q; rfl
  rw [row]
  decide

def innerTail154_21 : List Nat := []
theorem innerScan154_21 : scan edge (search edge 3) 4 [919] 919 innerTail154_21 best = best := rfl

def innerTail154_20 : List Nat := 7207 :: innerTail154_21
def innerCandidates154_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates154_20_checked : candidates edge 4 919 7207 innerTail154_21 best.bound = innerCandidates154_20 := by
  unfold candidates
  have row : edge 7207 = SearchData.row7207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan154_20 : scan edge (search edge 3) 4 [919] 919 innerTail154_20 best = best := by
  rw [innerTail154_20, scan, if_neg (by decide), innerCandidates154_20_checked]
  change scan edge (search edge 3) 4 [919] 919 innerTail154_21
    (if 3 ≤ innerCandidates154_20.length then search edge 3 [919, 7207] innerCandidates154_20 8126 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan154_21

def innerTail154_19 : List Nat := 7039 :: innerTail154_20
def innerCandidates154_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates154_19_checked : candidates edge 4 919 7039 innerTail154_20 best.bound = innerCandidates154_19 := by
  unfold candidates
  have row : edge 7039 = SearchData.row7039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan154_19 : scan edge (search edge 3) 4 [919] 919 innerTail154_19 best = best := by
  rw [innerTail154_19, scan, if_neg (by decide), innerCandidates154_19_checked]
  change scan edge (search edge 3) 4 [919] 919 innerTail154_20
    (if 3 ≤ innerCandidates154_19.length then search edge 3 [919, 7039] innerCandidates154_19 7958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan154_20

def innerTail154_18 : List Nat := 6961 :: innerTail154_19
def innerCandidates154_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates154_18_checked : candidates edge 4 919 6961 innerTail154_19 best.bound = innerCandidates154_18 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan154_18 : scan edge (search edge 3) 4 [919] 919 innerTail154_18 best = best := by
  rw [innerTail154_18, scan, if_neg (by decide), innerCandidates154_18_checked]
  change scan edge (search edge 3) 4 [919] 919 innerTail154_19
    (if 3 ≤ innerCandidates154_18.length then search edge 3 [919, 6961] innerCandidates154_18 7880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan154_19

def innerTail154_17 : List Nat := 6679 :: innerTail154_18
def innerCandidates154_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates154_17_checked : candidates edge 4 919 6679 innerTail154_18 best.bound = innerCandidates154_17 := by
  unfold candidates
  have row : edge 6679 = SearchData.row6679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan154_17 : scan edge (search edge 3) 4 [919] 919 innerTail154_17 best = best := by
  rw [innerTail154_17, scan, if_neg (by decide), innerCandidates154_17_checked]
  change scan edge (search edge 3) 4 [919] 919 innerTail154_18
    (if 3 ≤ innerCandidates154_17.length then search edge 3 [919, 6679] innerCandidates154_17 7598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan154_18

def innerTail154_16 : List Nat := 5881 :: innerTail154_17
def innerCandidates154_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates154_16_checked : candidates edge 4 919 5881 innerTail154_17 best.bound = innerCandidates154_16 := by
  unfold candidates
  have row : edge 5881 = SearchData.row5881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan154_16 : scan edge (search edge 3) 4 [919] 919 innerTail154_16 best = best := by
  rw [innerTail154_16, scan, if_neg (by decide), innerCandidates154_16_checked]
  change scan edge (search edge 3) 4 [919] 919 innerTail154_17
    (if 3 ≤ innerCandidates154_16.length then search edge 3 [919, 5881] innerCandidates154_16 6800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan154_17

def innerTail154_15 : List Nat := 5839 :: innerTail154_16
def innerCandidates154_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates154_15_checked : candidates edge 4 919 5839 innerTail154_16 best.bound = innerCandidates154_15 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan154_15 : scan edge (search edge 3) 4 [919] 919 innerTail154_15 best = best := by
  rw [innerTail154_15, scan, if_neg (by decide), innerCandidates154_15_checked]
  change scan edge (search edge 3) 4 [919] 919 innerTail154_16
    (if 3 ≤ innerCandidates154_15.length then search edge 3 [919, 5839] innerCandidates154_15 6758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan154_16

def innerTail154_14 : List Nat := 5821 :: innerTail154_15
def innerCandidates154_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates154_14_checked : candidates edge 4 919 5821 innerTail154_15 best.bound = innerCandidates154_14 := by
  unfold candidates
  have row : edge 5821 = SearchData.row5821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan154_14 : scan edge (search edge 3) 4 [919] 919 innerTail154_14 best = best := by
  rw [innerTail154_14, scan, if_neg (by decide), innerCandidates154_14_checked]
  change scan edge (search edge 3) 4 [919] 919 innerTail154_15
    (if 3 ≤ innerCandidates154_14.length then search edge 3 [919, 5821] innerCandidates154_14 6740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan154_15

def innerTail154_13 : List Nat := 5167 :: innerTail154_14
def innerCandidates154_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates154_13_checked : candidates edge 4 919 5167 innerTail154_14 best.bound = innerCandidates154_13 := by
  unfold candidates
  have row : edge 5167 = SearchData.row5167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan154_13 : scan edge (search edge 3) 4 [919] 919 innerTail154_13 best = best := by
  rw [innerTail154_13, scan, if_neg (by decide), innerCandidates154_13_checked]
  change scan edge (search edge 3) 4 [919] 919 innerTail154_14
    (if 3 ≤ innerCandidates154_13.length then search edge 3 [919, 5167] innerCandidates154_13 6086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan154_14

def innerTail154_12 : List Nat := 4597 :: innerTail154_13
def innerCandidates154_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates154_12_checked : candidates edge 4 919 4597 innerTail154_13 best.bound = innerCandidates154_12 := by
  unfold candidates
  have row : edge 4597 = SearchData.row4597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan154_12 : scan edge (search edge 3) 4 [919] 919 innerTail154_12 best = best := by
  rw [innerTail154_12, scan, if_neg (by decide), innerCandidates154_12_checked]
  change scan edge (search edge 3) 4 [919] 919 innerTail154_13
    (if 3 ≤ innerCandidates154_12.length then search edge 3 [919, 4597] innerCandidates154_12 5516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan154_13

def innerTail154_11 : List Nat := 4567 :: innerTail154_12
def innerCandidates154_11 : List Nat := [7039]
set_option maxRecDepth 4096 in
theorem innerCandidates154_11_checked : candidates edge 4 919 4567 innerTail154_12 best.bound = innerCandidates154_11 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan154_11 : scan edge (search edge 3) 4 [919] 919 innerTail154_11 best = best := by
  rw [innerTail154_11, scan, if_neg (by decide), innerCandidates154_11_checked]
  change scan edge (search edge 3) 4 [919] 919 innerTail154_12
    (if 3 ≤ innerCandidates154_11.length then search edge 3 [919, 4567] innerCandidates154_11 5486 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan154_12

def innerTail154_10 : List Nat := 4441 :: innerTail154_11
def innerCandidates154_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates154_10_checked : candidates edge 4 919 4441 innerTail154_11 best.bound = innerCandidates154_10 := by
  unfold candidates
  have row : edge 4441 = SearchData.row4441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan154_10 : scan edge (search edge 3) 4 [919] 919 innerTail154_10 best = best := by
  rw [innerTail154_10, scan, if_neg (by decide), innerCandidates154_10_checked]
  change scan edge (search edge 3) 4 [919] 919 innerTail154_11
    (if 3 ≤ innerCandidates154_10.length then search edge 3 [919, 4441] innerCandidates154_10 5360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan154_11

def innerTail154_9 : List Nat := 4231 :: innerTail154_10
def innerCandidates154_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates154_9_checked : candidates edge 4 919 4231 innerTail154_10 best.bound = innerCandidates154_9 := by
  unfold candidates
  have row : edge 4231 = SearchData.row4231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan154_9 : scan edge (search edge 3) 4 [919] 919 innerTail154_9 best = best := by
  rw [innerTail154_9, scan, if_neg (by decide), innerCandidates154_9_checked]
  change scan edge (search edge 3) 4 [919] 919 innerTail154_10
    (if 3 ≤ innerCandidates154_9.length then search edge 3 [919, 4231] innerCandidates154_9 5150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan154_10

def innerTail154_8 : List Nat := 3643 :: innerTail154_9
def innerCandidates154_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates154_8_checked : candidates edge 4 919 3643 innerTail154_9 best.bound = innerCandidates154_8 := by
  unfold candidates
  have row : edge 3643 = SearchData.row3643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan154_8 : scan edge (search edge 3) 4 [919] 919 innerTail154_8 best = best := by
  rw [innerTail154_8, scan, if_neg (by decide), innerCandidates154_8_checked]
  change scan edge (search edge 3) 4 [919] 919 innerTail154_9
    (if 3 ≤ innerCandidates154_8.length then search edge 3 [919, 3643] innerCandidates154_8 4562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan154_9

def innerTail154_7 : List Nat := 3463 :: innerTail154_8
def innerCandidates154_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates154_7_checked : candidates edge 4 919 3463 innerTail154_8 best.bound = innerCandidates154_7 := by
  unfold candidates
  have row : edge 3463 = SearchData.row3463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan154_7 : scan edge (search edge 3) 4 [919] 919 innerTail154_7 best = best := by
  rw [innerTail154_7, scan, if_neg (by decide), innerCandidates154_7_checked]
  change scan edge (search edge 3) 4 [919] 919 innerTail154_8
    (if 3 ≤ innerCandidates154_7.length then search edge 3 [919, 3463] innerCandidates154_7 4382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan154_8

def innerTail154_6 : List Nat := 2749 :: innerTail154_7
def innerCandidates154_6 : List Nat := [3643]
set_option maxRecDepth 4096 in
theorem innerCandidates154_6_checked : candidates edge 4 919 2749 innerTail154_7 best.bound = innerCandidates154_6 := by
  unfold candidates
  have row : edge 2749 = SearchData.row2749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan154_6 : scan edge (search edge 3) 4 [919] 919 innerTail154_6 best = best := by
  rw [innerTail154_6, scan, if_neg (by decide), innerCandidates154_6_checked]
  change scan edge (search edge 3) 4 [919] 919 innerTail154_7
    (if 3 ≤ innerCandidates154_6.length then search edge 3 [919, 2749] innerCandidates154_6 3668 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan154_7

def innerTail154_5 : List Nat := 2017 :: innerTail154_6
def innerCandidates154_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates154_5_checked : candidates edge 4 919 2017 innerTail154_6 best.bound = innerCandidates154_5 := by
  unfold candidates
  have row : edge 2017 = PrimePairCache.smallRow2017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan154_5 : scan edge (search edge 3) 4 [919] 919 innerTail154_5 best = best := by
  rw [innerTail154_5, scan, if_neg (by decide), innerCandidates154_5_checked]
  change scan edge (search edge 3) 4 [919] 919 innerTail154_6
    (if 3 ≤ innerCandidates154_5.length then search edge 3 [919, 2017] innerCandidates154_5 2936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan154_6

def innerTail154_4 : List Nat := 1627 :: innerTail154_5
def innerCandidates154_4 : List Nat := [3463, 5821, 6961]
set_option maxRecDepth 4096 in
theorem innerCandidates154_4_checked : candidates edge 4 919 1627 innerTail154_5 best.bound = innerCandidates154_4 := by
  unfold candidates
  have row : edge 1627 = PrimePairCache.smallRow1627.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild154_4 : search edge 3 [919, 1627] innerCandidates154_4 2546 best = best := by decide

theorem innerScan154_4 : scan edge (search edge 3) 4 [919] 919 innerTail154_4 best = best := by
  rw [innerTail154_4, scan, if_neg (by decide), innerCandidates154_4_checked]
  change scan edge (search edge 3) 4 [919] 919 innerTail154_5
    (if 3 ≤ innerCandidates154_4.length then search edge 3 [919, 1627] innerCandidates154_4 2546 best else best) = best
  rw [if_pos (by decide), innerChild154_4]
  exact innerScan154_5

def innerTail154_3 : List Nat := 1279 :: innerTail154_4
def innerCandidates154_3 : List Nat := [3463]
set_option maxRecDepth 4096 in
theorem innerCandidates154_3_checked : candidates edge 4 919 1279 innerTail154_4 best.bound = innerCandidates154_3 := by
  unfold candidates
  have row : edge 1279 = PrimePairCache.smallRow1279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan154_3 : scan edge (search edge 3) 4 [919] 919 innerTail154_3 best = best := by
  rw [innerTail154_3, scan, if_neg (by decide), innerCandidates154_3_checked]
  change scan edge (search edge 3) 4 [919] 919 innerTail154_4
    (if 3 ≤ innerCandidates154_3.length then search edge 3 [919, 1279] innerCandidates154_3 2198 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan154_4

def innerTail154_2 : List Nat := 1201 :: innerTail154_3
def innerCandidates154_2 : List Nat := [2017, 5167, 7207]
set_option maxRecDepth 4096 in
theorem innerCandidates154_2_checked : candidates edge 4 919 1201 innerTail154_3 best.bound = innerCandidates154_2 := by
  unfold candidates
  have row : edge 1201 = PrimePairCache.smallRow1201.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild154_2 : search edge 3 [919, 1201] innerCandidates154_2 2120 best = best := by decide

theorem innerScan154_2 : scan edge (search edge 3) 4 [919] 919 innerTail154_2 best = best := by
  rw [innerTail154_2, scan, if_neg (by decide), innerCandidates154_2_checked]
  change scan edge (search edge 3) 4 [919] 919 innerTail154_3
    (if 3 ≤ innerCandidates154_2.length then search edge 3 [919, 1201] innerCandidates154_2 2120 best else best) = best
  rw [if_pos (by decide), innerChild154_2]
  exact innerScan154_3

def innerTail154_1 : List Nat := 1069 :: innerTail154_2
def innerCandidates154_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates154_1_checked : candidates edge 4 919 1069 innerTail154_2 best.bound = innerCandidates154_1 := by
  unfold candidates
  have row : edge 1069 = PrimePairCache.smallRow1069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan154_1 : scan edge (search edge 3) 4 [919] 919 innerTail154_1 best = best := by
  rw [innerTail154_1, scan, if_neg (by decide), innerCandidates154_1_checked]
  change scan edge (search edge 3) 4 [919] 919 innerTail154_2
    (if 3 ≤ innerCandidates154_1.length then search edge 3 [919, 1069] innerCandidates154_1 1988 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan154_2

def innerTail154_0 : List Nat := 937 :: innerTail154_1
def innerCandidates154_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates154_0_checked : candidates edge 4 919 937 innerTail154_1 best.bound = innerCandidates154_0 := by
  unfold candidates
  have row : edge 937 = PrimePairCache.smallRow937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan154_0 : scan edge (search edge 3) 4 [919] 919 innerTail154_0 best = best := by
  rw [innerTail154_0, scan, if_neg (by decide), innerCandidates154_0_checked]
  change scan edge (search edge 3) 4 [919] 919 innerTail154_1
    (if 3 ≤ innerCandidates154_0.length then search edge 3 [919, 937] innerCandidates154_0 1856 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan154_1

theorem rootBranch154 : search edge 4 [919] rootCandidates154 919 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [919] 919 innerTail154_0 best = best
  exact innerScan154_0

def rootCandidates153 : List Nat := [947, 1103, 1229, 1511, 2309, 2399, 2657, 3413, 3821, 4337, 4517, 4583, 4643, 5303, 5471, 6089, 6113, 6269, 6971, 7109, 8111]
set_option maxRecDepth 4096 in
theorem rootCandidates153_checked : candidates edge 5 0 911 tail154 best.bound = rootCandidates153 := by
  unfold candidates
  have row : edge 911 = PrimePairCache.smallRow911.contains := by funext q; rfl
  rw [row]
  decide

def innerTail153_21 : List Nat := []
theorem innerScan153_21 : scan edge (search edge 3) 4 [911] 911 innerTail153_21 best = best := rfl

def innerTail153_20 : List Nat := 8111 :: innerTail153_21
def innerCandidates153_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates153_20_checked : candidates edge 4 911 8111 innerTail153_21 best.bound = innerCandidates153_20 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan153_20 : scan edge (search edge 3) 4 [911] 911 innerTail153_20 best = best := by
  rw [innerTail153_20, scan, if_neg (by decide), innerCandidates153_20_checked]
  change scan edge (search edge 3) 4 [911] 911 innerTail153_21
    (if 3 ≤ innerCandidates153_20.length then search edge 3 [911, 8111] innerCandidates153_20 9022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan153_21

def innerTail153_19 : List Nat := 7109 :: innerTail153_20
def innerCandidates153_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates153_19_checked : candidates edge 4 911 7109 innerTail153_20 best.bound = innerCandidates153_19 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan153_19 : scan edge (search edge 3) 4 [911] 911 innerTail153_19 best = best := by
  rw [innerTail153_19, scan, if_neg (by decide), innerCandidates153_19_checked]
  change scan edge (search edge 3) 4 [911] 911 innerTail153_20
    (if 3 ≤ innerCandidates153_19.length then search edge 3 [911, 7109] innerCandidates153_19 8020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan153_20

def innerTail153_18 : List Nat := 6971 :: innerTail153_19
def innerCandidates153_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates153_18_checked : candidates edge 4 911 6971 innerTail153_19 best.bound = innerCandidates153_18 := by
  unfold candidates
  have row : edge 6971 = SearchData.row6971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan153_18 : scan edge (search edge 3) 4 [911] 911 innerTail153_18 best = best := by
  rw [innerTail153_18, scan, if_neg (by decide), innerCandidates153_18_checked]
  change scan edge (search edge 3) 4 [911] 911 innerTail153_19
    (if 3 ≤ innerCandidates153_18.length then search edge 3 [911, 6971] innerCandidates153_18 7882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan153_19

def innerTail153_17 : List Nat := 6269 :: innerTail153_18
def innerCandidates153_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates153_17_checked : candidates edge 4 911 6269 innerTail153_18 best.bound = innerCandidates153_17 := by
  unfold candidates
  have row : edge 6269 = SearchData.row6269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan153_17 : scan edge (search edge 3) 4 [911] 911 innerTail153_17 best = best := by
  rw [innerTail153_17, scan, if_neg (by decide), innerCandidates153_17_checked]
  change scan edge (search edge 3) 4 [911] 911 innerTail153_18
    (if 3 ≤ innerCandidates153_17.length then search edge 3 [911, 6269] innerCandidates153_17 7180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan153_18

def innerTail153_16 : List Nat := 6113 :: innerTail153_17
def innerCandidates153_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates153_16_checked : candidates edge 4 911 6113 innerTail153_17 best.bound = innerCandidates153_16 := by
  unfold candidates
  have row : edge 6113 = SearchData.row6113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan153_16 : scan edge (search edge 3) 4 [911] 911 innerTail153_16 best = best := by
  rw [innerTail153_16, scan, if_neg (by decide), innerCandidates153_16_checked]
  change scan edge (search edge 3) 4 [911] 911 innerTail153_17
    (if 3 ≤ innerCandidates153_16.length then search edge 3 [911, 6113] innerCandidates153_16 7024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan153_17

def innerTail153_15 : List Nat := 6089 :: innerTail153_16
def innerCandidates153_15 : List Nat := [8111]
set_option maxRecDepth 4096 in
theorem innerCandidates153_15_checked : candidates edge 4 911 6089 innerTail153_16 best.bound = innerCandidates153_15 := by
  unfold candidates
  have row : edge 6089 = SearchData.row6089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan153_15 : scan edge (search edge 3) 4 [911] 911 innerTail153_15 best = best := by
  rw [innerTail153_15, scan, if_neg (by decide), innerCandidates153_15_checked]
  change scan edge (search edge 3) 4 [911] 911 innerTail153_16
    (if 3 ≤ innerCandidates153_15.length then search edge 3 [911, 6089] innerCandidates153_15 7000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan153_16

def innerTail153_14 : List Nat := 5471 :: innerTail153_15
def innerCandidates153_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates153_14_checked : candidates edge 4 911 5471 innerTail153_15 best.bound = innerCandidates153_14 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan153_14 : scan edge (search edge 3) 4 [911] 911 innerTail153_14 best = best := by
  rw [innerTail153_14, scan, if_neg (by decide), innerCandidates153_14_checked]
  change scan edge (search edge 3) 4 [911] 911 innerTail153_15
    (if 3 ≤ innerCandidates153_14.length then search edge 3 [911, 5471] innerCandidates153_14 6382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan153_15

def innerTail153_13 : List Nat := 5303 :: innerTail153_14
def innerCandidates153_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates153_13_checked : candidates edge 4 911 5303 innerTail153_14 best.bound = innerCandidates153_13 := by
  unfold candidates
  have row : edge 5303 = SearchData.row5303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan153_13 : scan edge (search edge 3) 4 [911] 911 innerTail153_13 best = best := by
  rw [innerTail153_13, scan, if_neg (by decide), innerCandidates153_13_checked]
  change scan edge (search edge 3) 4 [911] 911 innerTail153_14
    (if 3 ≤ innerCandidates153_13.length then search edge 3 [911, 5303] innerCandidates153_13 6214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan153_14

def innerTail153_12 : List Nat := 4643 :: innerTail153_13
def innerCandidates153_12 : List Nat := [5471]
set_option maxRecDepth 4096 in
theorem innerCandidates153_12_checked : candidates edge 4 911 4643 innerTail153_13 best.bound = innerCandidates153_12 := by
  unfold candidates
  have row : edge 4643 = SearchData.row4643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan153_12 : scan edge (search edge 3) 4 [911] 911 innerTail153_12 best = best := by
  rw [innerTail153_12, scan, if_neg (by decide), innerCandidates153_12_checked]
  change scan edge (search edge 3) 4 [911] 911 innerTail153_13
    (if 3 ≤ innerCandidates153_12.length then search edge 3 [911, 4643] innerCandidates153_12 5554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan153_13

def innerTail153_11 : List Nat := 4583 :: innerTail153_12
def innerCandidates153_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates153_11_checked : candidates edge 4 911 4583 innerTail153_12 best.bound = innerCandidates153_11 := by
  unfold candidates
  have row : edge 4583 = SearchData.row4583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan153_11 : scan edge (search edge 3) 4 [911] 911 innerTail153_11 best = best := by
  rw [innerTail153_11, scan, if_neg (by decide), innerCandidates153_11_checked]
  change scan edge (search edge 3) 4 [911] 911 innerTail153_12
    (if 3 ≤ innerCandidates153_11.length then search edge 3 [911, 4583] innerCandidates153_11 5494 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan153_12

def innerTail153_10 : List Nat := 4517 :: innerTail153_11
def innerCandidates153_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates153_10_checked : candidates edge 4 911 4517 innerTail153_11 best.bound = innerCandidates153_10 := by
  unfold candidates
  have row : edge 4517 = SearchData.row4517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan153_10 : scan edge (search edge 3) 4 [911] 911 innerTail153_10 best = best := by
  rw [innerTail153_10, scan, if_neg (by decide), innerCandidates153_10_checked]
  change scan edge (search edge 3) 4 [911] 911 innerTail153_11
    (if 3 ≤ innerCandidates153_10.length then search edge 3 [911, 4517] innerCandidates153_10 5428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan153_11

def innerTail153_9 : List Nat := 4337 :: innerTail153_10
def innerCandidates153_9 : List Nat := [5471]
set_option maxRecDepth 4096 in
theorem innerCandidates153_9_checked : candidates edge 4 911 4337 innerTail153_10 best.bound = innerCandidates153_9 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan153_9 : scan edge (search edge 3) 4 [911] 911 innerTail153_9 best = best := by
  rw [innerTail153_9, scan, if_neg (by decide), innerCandidates153_9_checked]
  change scan edge (search edge 3) 4 [911] 911 innerTail153_10
    (if 3 ≤ innerCandidates153_9.length then search edge 3 [911, 4337] innerCandidates153_9 5248 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan153_10

def innerTail153_8 : List Nat := 3821 :: innerTail153_9
def innerCandidates153_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates153_8_checked : candidates edge 4 911 3821 innerTail153_9 best.bound = innerCandidates153_8 := by
  unfold candidates
  have row : edge 3821 = SearchData.row3821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan153_8 : scan edge (search edge 3) 4 [911] 911 innerTail153_8 best = best := by
  rw [innerTail153_8, scan, if_neg (by decide), innerCandidates153_8_checked]
  change scan edge (search edge 3) 4 [911] 911 innerTail153_9
    (if 3 ≤ innerCandidates153_8.length then search edge 3 [911, 3821] innerCandidates153_8 4732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan153_9

def innerTail153_7 : List Nat := 3413 :: innerTail153_8
def innerCandidates153_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates153_7_checked : candidates edge 4 911 3413 innerTail153_8 best.bound = innerCandidates153_7 := by
  unfold candidates
  have row : edge 3413 = SearchData.row3413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan153_7 : scan edge (search edge 3) 4 [911] 911 innerTail153_7 best = best := by
  rw [innerTail153_7, scan, if_neg (by decide), innerCandidates153_7_checked]
  change scan edge (search edge 3) 4 [911] 911 innerTail153_8
    (if 3 ≤ innerCandidates153_7.length then search edge 3 [911, 3413] innerCandidates153_7 4324 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan153_8

def innerTail153_6 : List Nat := 2657 :: innerTail153_7
def innerCandidates153_6 : List Nat := [4517]
set_option maxRecDepth 4096 in
theorem innerCandidates153_6_checked : candidates edge 4 911 2657 innerTail153_7 best.bound = innerCandidates153_6 := by
  unfold candidates
  have row : edge 2657 = PrimePairCache.smallRow2657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan153_6 : scan edge (search edge 3) 4 [911] 911 innerTail153_6 best = best := by
  rw [innerTail153_6, scan, if_neg (by decide), innerCandidates153_6_checked]
  change scan edge (search edge 3) 4 [911] 911 innerTail153_7
    (if 3 ≤ innerCandidates153_6.length then search edge 3 [911, 2657] innerCandidates153_6 3568 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan153_7

def innerTail153_5 : List Nat := 2399 :: innerTail153_6
def innerCandidates153_5 : List Nat := [3821, 4517]
set_option maxRecDepth 4096 in
theorem innerCandidates153_5_checked : candidates edge 4 911 2399 innerTail153_6 best.bound = innerCandidates153_5 := by
  unfold candidates
  have row : edge 2399 = PrimePairCache.smallRow2399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan153_5 : scan edge (search edge 3) 4 [911] 911 innerTail153_5 best = best := by
  rw [innerTail153_5, scan, if_neg (by decide), innerCandidates153_5_checked]
  change scan edge (search edge 3) 4 [911] 911 innerTail153_6
    (if 3 ≤ innerCandidates153_5.length then search edge 3 [911, 2399] innerCandidates153_5 3310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan153_6

def innerTail153_4 : List Nat := 2309 :: innerTail153_5
def innerCandidates153_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates153_4_checked : candidates edge 4 911 2309 innerTail153_5 best.bound = innerCandidates153_4 := by
  unfold candidates
  have row : edge 2309 = PrimePairCache.smallRow2309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan153_4 : scan edge (search edge 3) 4 [911] 911 innerTail153_4 best = best := by
  rw [innerTail153_4, scan, if_neg (by decide), innerCandidates153_4_checked]
  change scan edge (search edge 3) 4 [911] 911 innerTail153_5
    (if 3 ≤ innerCandidates153_4.length then search edge 3 [911, 2309] innerCandidates153_4 3220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan153_5

def innerTail153_3 : List Nat := 1511 :: innerTail153_4
def innerCandidates153_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates153_3_checked : candidates edge 4 911 1511 innerTail153_4 best.bound = innerCandidates153_3 := by
  unfold candidates
  have row : edge 1511 = PrimePairCache.smallRow1511.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan153_3 : scan edge (search edge 3) 4 [911] 911 innerTail153_3 best = best := by
  rw [innerTail153_3, scan, if_neg (by decide), innerCandidates153_3_checked]
  change scan edge (search edge 3) 4 [911] 911 innerTail153_4
    (if 3 ≤ innerCandidates153_3.length then search edge 3 [911, 1511] innerCandidates153_3 2422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan153_4

def innerTail153_2 : List Nat := 1229 :: innerTail153_3
def innerCandidates153_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates153_2_checked : candidates edge 4 911 1229 innerTail153_3 best.bound = innerCandidates153_2 := by
  unfold candidates
  have row : edge 1229 = PrimePairCache.smallRow1229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan153_2 : scan edge (search edge 3) 4 [911] 911 innerTail153_2 best = best := by
  rw [innerTail153_2, scan, if_neg (by decide), innerCandidates153_2_checked]
  change scan edge (search edge 3) 4 [911] 911 innerTail153_3
    (if 3 ≤ innerCandidates153_2.length then search edge 3 [911, 1229] innerCandidates153_2 2140 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan153_3

def innerTail153_1 : List Nat := 1103 :: innerTail153_2
def innerCandidates153_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates153_1_checked : candidates edge 4 911 1103 innerTail153_2 best.bound = innerCandidates153_1 := by
  unfold candidates
  have row : edge 1103 = PrimePairCache.smallRow1103.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan153_1 : scan edge (search edge 3) 4 [911] 911 innerTail153_1 best = best := by
  rw [innerTail153_1, scan, if_neg (by decide), innerCandidates153_1_checked]
  change scan edge (search edge 3) 4 [911] 911 innerTail153_2
    (if 3 ≤ innerCandidates153_1.length then search edge 3 [911, 1103] innerCandidates153_1 2014 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan153_2

def innerTail153_0 : List Nat := 947 :: innerTail153_1
def innerCandidates153_0 : List Nat := [2657, 8111]
set_option maxRecDepth 4096 in
theorem innerCandidates153_0_checked : candidates edge 4 911 947 innerTail153_1 best.bound = innerCandidates153_0 := by
  unfold candidates
  have row : edge 947 = PrimePairCache.smallRow947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan153_0 : scan edge (search edge 3) 4 [911] 911 innerTail153_0 best = best := by
  rw [innerTail153_0, scan, if_neg (by decide), innerCandidates153_0_checked]
  change scan edge (search edge 3) 4 [911] 911 innerTail153_1
    (if 3 ≤ innerCandidates153_0.length then search edge 3 [911, 947] innerCandidates153_0 1858 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan153_1

theorem rootBranch153 : search edge 4 [911] rootCandidates153 911 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [911] 911 innerTail153_0 best = best
  exact innerScan153_0

end ProjectEulerSolutions.PrimePairUpper.Search
