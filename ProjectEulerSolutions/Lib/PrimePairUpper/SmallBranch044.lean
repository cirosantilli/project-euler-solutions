import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates697 : List Nat := [5651, 5783, 6113, 6449, 6719, 6827, 7109, 7253, 7793, 7883, 8231]
set_option maxRecDepth 4096 in
theorem rootCandidates697_checked : candidates edge 5 0 5279 tail698 best.bound = rootCandidates697 := by
  unfold candidates
  have row : edge 5279 = SearchData.row5279.contains := by funext q; rfl
  rw [row]
  decide

def innerTail697_11 : List Nat := []
theorem innerScan697_11 : scan edge (search edge 3) 4 [5279] 5279 innerTail697_11 best = best := rfl

def innerTail697_10 : List Nat := 8231 :: innerTail697_11
def innerCandidates697_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates697_10_checked : candidates edge 4 5279 8231 innerTail697_11 best.bound = innerCandidates697_10 := by
  unfold candidates
  have row : edge 8231 = SearchData.row8231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan697_10 : scan edge (search edge 3) 4 [5279] 5279 innerTail697_10 best = best := by
  rw [innerTail697_10, scan, if_neg (by decide), innerCandidates697_10_checked]
  change scan edge (search edge 3) 4 [5279] 5279 innerTail697_11
    (if 3 ≤ innerCandidates697_10.length then search edge 3 [5279, 8231] innerCandidates697_10 13510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan697_11

def innerTail697_9 : List Nat := 7883 :: innerTail697_10
def innerCandidates697_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates697_9_checked : candidates edge 4 5279 7883 innerTail697_10 best.bound = innerCandidates697_9 := by
  unfold candidates
  have row : edge 7883 = SearchData.row7883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan697_9 : scan edge (search edge 3) 4 [5279] 5279 innerTail697_9 best = best := by
  rw [innerTail697_9, scan, if_neg (by decide), innerCandidates697_9_checked]
  change scan edge (search edge 3) 4 [5279] 5279 innerTail697_10
    (if 3 ≤ innerCandidates697_9.length then search edge 3 [5279, 7883] innerCandidates697_9 13162 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan697_10

def innerTail697_8 : List Nat := 7793 :: innerTail697_9
def innerCandidates697_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates697_8_checked : candidates edge 4 5279 7793 innerTail697_9 best.bound = innerCandidates697_8 := by
  unfold candidates
  have row : edge 7793 = SearchData.row7793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan697_8 : scan edge (search edge 3) 4 [5279] 5279 innerTail697_8 best = best := by
  rw [innerTail697_8, scan, if_neg (by decide), innerCandidates697_8_checked]
  change scan edge (search edge 3) 4 [5279] 5279 innerTail697_9
    (if 3 ≤ innerCandidates697_8.length then search edge 3 [5279, 7793] innerCandidates697_8 13072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan697_9

def innerTail697_7 : List Nat := 7253 :: innerTail697_8
def innerCandidates697_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates697_7_checked : candidates edge 4 5279 7253 innerTail697_8 best.bound = innerCandidates697_7 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan697_7 : scan edge (search edge 3) 4 [5279] 5279 innerTail697_7 best = best := by
  rw [innerTail697_7, scan, if_neg (by decide), innerCandidates697_7_checked]
  change scan edge (search edge 3) 4 [5279] 5279 innerTail697_8
    (if 3 ≤ innerCandidates697_7.length then search edge 3 [5279, 7253] innerCandidates697_7 12532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan697_8

def innerTail697_6 : List Nat := 7109 :: innerTail697_7
def innerCandidates697_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates697_6_checked : candidates edge 4 5279 7109 innerTail697_7 best.bound = innerCandidates697_6 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan697_6 : scan edge (search edge 3) 4 [5279] 5279 innerTail697_6 best = best := by
  rw [innerTail697_6, scan, if_neg (by decide), innerCandidates697_6_checked]
  change scan edge (search edge 3) 4 [5279] 5279 innerTail697_7
    (if 3 ≤ innerCandidates697_6.length then search edge 3 [5279, 7109] innerCandidates697_6 12388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan697_7

def innerTail697_5 : List Nat := 6827 :: innerTail697_6
def innerCandidates697_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates697_5_checked : candidates edge 4 5279 6827 innerTail697_6 best.bound = innerCandidates697_5 := by
  unfold candidates
  have row : edge 6827 = SearchData.row6827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan697_5 : scan edge (search edge 3) 4 [5279] 5279 innerTail697_5 best = best := by
  rw [innerTail697_5, scan, if_neg (by decide), innerCandidates697_5_checked]
  change scan edge (search edge 3) 4 [5279] 5279 innerTail697_6
    (if 3 ≤ innerCandidates697_5.length then search edge 3 [5279, 6827] innerCandidates697_5 12106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan697_6

def innerTail697_4 : List Nat := 6719 :: innerTail697_5
def innerCandidates697_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates697_4_checked : candidates edge 4 5279 6719 innerTail697_5 best.bound = innerCandidates697_4 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan697_4 : scan edge (search edge 3) 4 [5279] 5279 innerTail697_4 best = best := by
  rw [innerTail697_4, scan, if_neg (by decide), innerCandidates697_4_checked]
  change scan edge (search edge 3) 4 [5279] 5279 innerTail697_5
    (if 3 ≤ innerCandidates697_4.length then search edge 3 [5279, 6719] innerCandidates697_4 11998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan697_5

def innerTail697_3 : List Nat := 6449 :: innerTail697_4
def innerCandidates697_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates697_3_checked : candidates edge 4 5279 6449 innerTail697_4 best.bound = innerCandidates697_3 := by
  unfold candidates
  have row : edge 6449 = SearchData.row6449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan697_3 : scan edge (search edge 3) 4 [5279] 5279 innerTail697_3 best = best := by
  rw [innerTail697_3, scan, if_neg (by decide), innerCandidates697_3_checked]
  change scan edge (search edge 3) 4 [5279] 5279 innerTail697_4
    (if 3 ≤ innerCandidates697_3.length then search edge 3 [5279, 6449] innerCandidates697_3 11728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan697_4

def innerTail697_2 : List Nat := 6113 :: innerTail697_3
def innerCandidates697_2 : List Nat := [7793]
set_option maxRecDepth 4096 in
theorem innerCandidates697_2_checked : candidates edge 4 5279 6113 innerTail697_3 best.bound = innerCandidates697_2 := by
  unfold candidates
  have row : edge 6113 = SearchData.row6113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan697_2 : scan edge (search edge 3) 4 [5279] 5279 innerTail697_2 best = best := by
  rw [innerTail697_2, scan, if_neg (by decide), innerCandidates697_2_checked]
  change scan edge (search edge 3) 4 [5279] 5279 innerTail697_3
    (if 3 ≤ innerCandidates697_2.length then search edge 3 [5279, 6113] innerCandidates697_2 11392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan697_3

def innerTail697_1 : List Nat := 5783 :: innerTail697_2
def innerCandidates697_1 : List Nat := [6719]
set_option maxRecDepth 4096 in
theorem innerCandidates697_1_checked : candidates edge 4 5279 5783 innerTail697_2 best.bound = innerCandidates697_1 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan697_1 : scan edge (search edge 3) 4 [5279] 5279 innerTail697_1 best = best := by
  rw [innerTail697_1, scan, if_neg (by decide), innerCandidates697_1_checked]
  change scan edge (search edge 3) 4 [5279] 5279 innerTail697_2
    (if 3 ≤ innerCandidates697_1.length then search edge 3 [5279, 5783] innerCandidates697_1 11062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan697_2

def innerTail697_0 : List Nat := 5651 :: innerTail697_1
def innerCandidates697_0 : List Nat := [5783]
set_option maxRecDepth 4096 in
theorem innerCandidates697_0_checked : candidates edge 4 5279 5651 innerTail697_1 best.bound = innerCandidates697_0 := by
  unfold candidates
  have row : edge 5651 = SearchData.row5651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan697_0 : scan edge (search edge 3) 4 [5279] 5279 innerTail697_0 best = best := by
  rw [innerTail697_0, scan, if_neg (by decide), innerCandidates697_0_checked]
  change scan edge (search edge 3) 4 [5279] 5279 innerTail697_1
    (if 3 ≤ innerCandidates697_0.length then search edge 3 [5279, 5651] innerCandidates697_0 10930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan697_1

theorem rootBranch697 : search edge 4 [5279] rootCandidates697 5279 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5279] 5279 innerTail697_0 best = best
  exact innerScan697_0

def rootCandidates696 : List Nat := [5381, 5651, 6833, 7127, 7211, 8093]
set_option maxRecDepth 4096 in
theorem rootCandidates696_checked : candidates edge 5 0 5273 tail697 best.bound = rootCandidates696 := by
  unfold candidates
  have row : edge 5273 = SearchData.row5273.contains := by funext q; rfl
  rw [row]
  decide

def innerTail696_6 : List Nat := []
theorem innerScan696_6 : scan edge (search edge 3) 4 [5273] 5273 innerTail696_6 best = best := rfl

def innerTail696_5 : List Nat := 8093 :: innerTail696_6
def innerCandidates696_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates696_5_checked : candidates edge 4 5273 8093 innerTail696_6 best.bound = innerCandidates696_5 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan696_5 : scan edge (search edge 3) 4 [5273] 5273 innerTail696_5 best = best := by
  rw [innerTail696_5, scan, if_neg (by decide), innerCandidates696_5_checked]
  change scan edge (search edge 3) 4 [5273] 5273 innerTail696_6
    (if 3 ≤ innerCandidates696_5.length then search edge 3 [5273, 8093] innerCandidates696_5 13366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan696_6

def innerTail696_4 : List Nat := 7211 :: innerTail696_5
def innerCandidates696_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates696_4_checked : candidates edge 4 5273 7211 innerTail696_5 best.bound = innerCandidates696_4 := by
  unfold candidates
  have row : edge 7211 = SearchData.row7211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan696_4 : scan edge (search edge 3) 4 [5273] 5273 innerTail696_4 best = best := by
  rw [innerTail696_4, scan, if_neg (by decide), innerCandidates696_4_checked]
  change scan edge (search edge 3) 4 [5273] 5273 innerTail696_5
    (if 3 ≤ innerCandidates696_4.length then search edge 3 [5273, 7211] innerCandidates696_4 12484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan696_5

def innerTail696_3 : List Nat := 7127 :: innerTail696_4
def innerCandidates696_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates696_3_checked : candidates edge 4 5273 7127 innerTail696_4 best.bound = innerCandidates696_3 := by
  unfold candidates
  have row : edge 7127 = SearchData.row7127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan696_3 : scan edge (search edge 3) 4 [5273] 5273 innerTail696_3 best = best := by
  rw [innerTail696_3, scan, if_neg (by decide), innerCandidates696_3_checked]
  change scan edge (search edge 3) 4 [5273] 5273 innerTail696_4
    (if 3 ≤ innerCandidates696_3.length then search edge 3 [5273, 7127] innerCandidates696_3 12400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan696_4

def innerTail696_2 : List Nat := 6833 :: innerTail696_3
def innerCandidates696_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates696_2_checked : candidates edge 4 5273 6833 innerTail696_3 best.bound = innerCandidates696_2 := by
  unfold candidates
  have row : edge 6833 = SearchData.row6833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan696_2 : scan edge (search edge 3) 4 [5273] 5273 innerTail696_2 best = best := by
  rw [innerTail696_2, scan, if_neg (by decide), innerCandidates696_2_checked]
  change scan edge (search edge 3) 4 [5273] 5273 innerTail696_3
    (if 3 ≤ innerCandidates696_2.length then search edge 3 [5273, 6833] innerCandidates696_2 12106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan696_3

def innerTail696_1 : List Nat := 5651 :: innerTail696_2
def innerCandidates696_1 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates696_1_checked : candidates edge 4 5273 5651 innerTail696_2 best.bound = innerCandidates696_1 := by
  unfold candidates
  have row : edge 5651 = SearchData.row5651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan696_1 : scan edge (search edge 3) 4 [5273] 5273 innerTail696_1 best = best := by
  rw [innerTail696_1, scan, if_neg (by decide), innerCandidates696_1_checked]
  change scan edge (search edge 3) 4 [5273] 5273 innerTail696_2
    (if 3 ≤ innerCandidates696_1.length then search edge 3 [5273, 5651] innerCandidates696_1 10924 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan696_2

def innerTail696_0 : List Nat := 5381 :: innerTail696_1
def innerCandidates696_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates696_0_checked : candidates edge 4 5273 5381 innerTail696_1 best.bound = innerCandidates696_0 := by
  unfold candidates
  have row : edge 5381 = SearchData.row5381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan696_0 : scan edge (search edge 3) 4 [5273] 5273 innerTail696_0 best = best := by
  rw [innerTail696_0, scan, if_neg (by decide), innerCandidates696_0_checked]
  change scan edge (search edge 3) 4 [5273] 5273 innerTail696_1
    (if 3 ≤ innerCandidates696_0.length then search edge 3 [5273, 5381] innerCandidates696_0 10654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan696_1

theorem rootBranch696 : search edge 4 [5273] rootCandidates696 5273 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5273] 5273 innerTail696_0 best = best
  exact innerScan696_0

def rootCandidates695 : List Nat := [5573, 5987, 6329, 6779, 6983, 7247, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates695_checked : candidates edge 5 0 5261 tail696 best.bound = rootCandidates695 := by
  unfold candidates
  have row : edge 5261 = SearchData.row5261.contains := by funext q; rfl
  rw [row]
  decide

def innerTail695_7 : List Nat := []
theorem innerScan695_7 : scan edge (search edge 3) 4 [5261] 5261 innerTail695_7 best = best := rfl

def innerTail695_6 : List Nat := 8387 :: innerTail695_7
def innerCandidates695_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates695_6_checked : candidates edge 4 5261 8387 innerTail695_7 best.bound = innerCandidates695_6 := by
  unfold candidates
  have row : edge 8387 = SearchData.row8387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan695_6 : scan edge (search edge 3) 4 [5261] 5261 innerTail695_6 best = best := by
  rw [innerTail695_6, scan, if_neg (by decide), innerCandidates695_6_checked]
  change scan edge (search edge 3) 4 [5261] 5261 innerTail695_7
    (if 3 ≤ innerCandidates695_6.length then search edge 3 [5261, 8387] innerCandidates695_6 13648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan695_7

def innerTail695_5 : List Nat := 7247 :: innerTail695_6
def innerCandidates695_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates695_5_checked : candidates edge 4 5261 7247 innerTail695_6 best.bound = innerCandidates695_5 := by
  unfold candidates
  have row : edge 7247 = SearchData.row7247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan695_5 : scan edge (search edge 3) 4 [5261] 5261 innerTail695_5 best = best := by
  rw [innerTail695_5, scan, if_neg (by decide), innerCandidates695_5_checked]
  change scan edge (search edge 3) 4 [5261] 5261 innerTail695_6
    (if 3 ≤ innerCandidates695_5.length then search edge 3 [5261, 7247] innerCandidates695_5 12508 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan695_6

def innerTail695_4 : List Nat := 6983 :: innerTail695_5
def innerCandidates695_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates695_4_checked : candidates edge 4 5261 6983 innerTail695_5 best.bound = innerCandidates695_4 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan695_4 : scan edge (search edge 3) 4 [5261] 5261 innerTail695_4 best = best := by
  rw [innerTail695_4, scan, if_neg (by decide), innerCandidates695_4_checked]
  change scan edge (search edge 3) 4 [5261] 5261 innerTail695_5
    (if 3 ≤ innerCandidates695_4.length then search edge 3 [5261, 6983] innerCandidates695_4 12244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan695_5

def innerTail695_3 : List Nat := 6779 :: innerTail695_4
def innerCandidates695_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates695_3_checked : candidates edge 4 5261 6779 innerTail695_4 best.bound = innerCandidates695_3 := by
  unfold candidates
  have row : edge 6779 = SearchData.row6779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan695_3 : scan edge (search edge 3) 4 [5261] 5261 innerTail695_3 best = best := by
  rw [innerTail695_3, scan, if_neg (by decide), innerCandidates695_3_checked]
  change scan edge (search edge 3) 4 [5261] 5261 innerTail695_4
    (if 3 ≤ innerCandidates695_3.length then search edge 3 [5261, 6779] innerCandidates695_3 12040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan695_4

def innerTail695_2 : List Nat := 6329 :: innerTail695_3
def innerCandidates695_2 : List Nat := [6983]
set_option maxRecDepth 4096 in
theorem innerCandidates695_2_checked : candidates edge 4 5261 6329 innerTail695_3 best.bound = innerCandidates695_2 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan695_2 : scan edge (search edge 3) 4 [5261] 5261 innerTail695_2 best = best := by
  rw [innerTail695_2, scan, if_neg (by decide), innerCandidates695_2_checked]
  change scan edge (search edge 3) 4 [5261] 5261 innerTail695_3
    (if 3 ≤ innerCandidates695_2.length then search edge 3 [5261, 6329] innerCandidates695_2 11590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan695_3

def innerTail695_1 : List Nat := 5987 :: innerTail695_2
def innerCandidates695_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates695_1_checked : candidates edge 4 5261 5987 innerTail695_2 best.bound = innerCandidates695_1 := by
  unfold candidates
  have row : edge 5987 = SearchData.row5987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan695_1 : scan edge (search edge 3) 4 [5261] 5261 innerTail695_1 best = best := by
  rw [innerTail695_1, scan, if_neg (by decide), innerCandidates695_1_checked]
  change scan edge (search edge 3) 4 [5261] 5261 innerTail695_2
    (if 3 ≤ innerCandidates695_1.length then search edge 3 [5261, 5987] innerCandidates695_1 11248 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan695_2

def innerTail695_0 : List Nat := 5573 :: innerTail695_1
def innerCandidates695_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates695_0_checked : candidates edge 4 5261 5573 innerTail695_1 best.bound = innerCandidates695_0 := by
  unfold candidates
  have row : edge 5573 = SearchData.row5573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan695_0 : scan edge (search edge 3) 4 [5261] 5261 innerTail695_0 best = best := by
  rw [innerTail695_0, scan, if_neg (by decide), innerCandidates695_0_checked]
  change scan edge (search edge 3) 4 [5261] 5261 innerTail695_1
    (if 3 ≤ innerCandidates695_0.length then search edge 3 [5261, 5573] innerCandidates695_0 10834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan695_1

theorem rootBranch695 : search edge 4 [5261] rootCandidates695 5261 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5261] 5261 innerTail695_0 best = best
  exact innerScan695_0

def rootCandidates694 : List Nat := [5471, 5693, 5783, 6563, 6761, 7043, 7151, 7349, 7487, 8009, 8243]
set_option maxRecDepth 4096 in
theorem rootCandidates694_checked : candidates edge 5 0 5237 tail695 best.bound = rootCandidates694 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext q; rfl
  rw [row]
  decide

def innerTail694_11 : List Nat := []
theorem innerScan694_11 : scan edge (search edge 3) 4 [5237] 5237 innerTail694_11 best = best := rfl

def innerTail694_10 : List Nat := 8243 :: innerTail694_11
def innerCandidates694_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates694_10_checked : candidates edge 4 5237 8243 innerTail694_11 best.bound = innerCandidates694_10 := by
  unfold candidates
  have row : edge 8243 = SearchData.row8243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan694_10 : scan edge (search edge 3) 4 [5237] 5237 innerTail694_10 best = best := by
  rw [innerTail694_10, scan, if_neg (by decide), innerCandidates694_10_checked]
  change scan edge (search edge 3) 4 [5237] 5237 innerTail694_11
    (if 3 ≤ innerCandidates694_10.length then search edge 3 [5237, 8243] innerCandidates694_10 13480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan694_11

def innerTail694_9 : List Nat := 8009 :: innerTail694_10
def innerCandidates694_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates694_9_checked : candidates edge 4 5237 8009 innerTail694_10 best.bound = innerCandidates694_9 := by
  unfold candidates
  have row : edge 8009 = SearchData.row8009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan694_9 : scan edge (search edge 3) 4 [5237] 5237 innerTail694_9 best = best := by
  rw [innerTail694_9, scan, if_neg (by decide), innerCandidates694_9_checked]
  change scan edge (search edge 3) 4 [5237] 5237 innerTail694_10
    (if 3 ≤ innerCandidates694_9.length then search edge 3 [5237, 8009] innerCandidates694_9 13246 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan694_10

def innerTail694_8 : List Nat := 7487 :: innerTail694_9
def innerCandidates694_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates694_8_checked : candidates edge 4 5237 7487 innerTail694_9 best.bound = innerCandidates694_8 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan694_8 : scan edge (search edge 3) 4 [5237] 5237 innerTail694_8 best = best := by
  rw [innerTail694_8, scan, if_neg (by decide), innerCandidates694_8_checked]
  change scan edge (search edge 3) 4 [5237] 5237 innerTail694_9
    (if 3 ≤ innerCandidates694_8.length then search edge 3 [5237, 7487] innerCandidates694_8 12724 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan694_9

def innerTail694_7 : List Nat := 7349 :: innerTail694_8
def innerCandidates694_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates694_7_checked : candidates edge 4 5237 7349 innerTail694_8 best.bound = innerCandidates694_7 := by
  unfold candidates
  have row : edge 7349 = SearchData.row7349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan694_7 : scan edge (search edge 3) 4 [5237] 5237 innerTail694_7 best = best := by
  rw [innerTail694_7, scan, if_neg (by decide), innerCandidates694_7_checked]
  change scan edge (search edge 3) 4 [5237] 5237 innerTail694_8
    (if 3 ≤ innerCandidates694_7.length then search edge 3 [5237, 7349] innerCandidates694_7 12586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan694_8

def innerTail694_6 : List Nat := 7151 :: innerTail694_7
def innerCandidates694_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates694_6_checked : candidates edge 4 5237 7151 innerTail694_7 best.bound = innerCandidates694_6 := by
  unfold candidates
  have row : edge 7151 = SearchData.row7151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan694_6 : scan edge (search edge 3) 4 [5237] 5237 innerTail694_6 best = best := by
  rw [innerTail694_6, scan, if_neg (by decide), innerCandidates694_6_checked]
  change scan edge (search edge 3) 4 [5237] 5237 innerTail694_7
    (if 3 ≤ innerCandidates694_6.length then search edge 3 [5237, 7151] innerCandidates694_6 12388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan694_7

def innerTail694_5 : List Nat := 7043 :: innerTail694_6
def innerCandidates694_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates694_5_checked : candidates edge 4 5237 7043 innerTail694_6 best.bound = innerCandidates694_5 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan694_5 : scan edge (search edge 3) 4 [5237] 5237 innerTail694_5 best = best := by
  rw [innerTail694_5, scan, if_neg (by decide), innerCandidates694_5_checked]
  change scan edge (search edge 3) 4 [5237] 5237 innerTail694_6
    (if 3 ≤ innerCandidates694_5.length then search edge 3 [5237, 7043] innerCandidates694_5 12280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan694_6

def innerTail694_4 : List Nat := 6761 :: innerTail694_5
def innerCandidates694_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates694_4_checked : candidates edge 4 5237 6761 innerTail694_5 best.bound = innerCandidates694_4 := by
  unfold candidates
  have row : edge 6761 = SearchData.row6761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan694_4 : scan edge (search edge 3) 4 [5237] 5237 innerTail694_4 best = best := by
  rw [innerTail694_4, scan, if_neg (by decide), innerCandidates694_4_checked]
  change scan edge (search edge 3) 4 [5237] 5237 innerTail694_5
    (if 3 ≤ innerCandidates694_4.length then search edge 3 [5237, 6761] innerCandidates694_4 11998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan694_5

def innerTail694_3 : List Nat := 6563 :: innerTail694_4
def innerCandidates694_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates694_3_checked : candidates edge 4 5237 6563 innerTail694_4 best.bound = innerCandidates694_3 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan694_3 : scan edge (search edge 3) 4 [5237] 5237 innerTail694_3 best = best := by
  rw [innerTail694_3, scan, if_neg (by decide), innerCandidates694_3_checked]
  change scan edge (search edge 3) 4 [5237] 5237 innerTail694_4
    (if 3 ≤ innerCandidates694_3.length then search edge 3 [5237, 6563] innerCandidates694_3 11800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan694_4

def innerTail694_2 : List Nat := 5783 :: innerTail694_3
def innerCandidates694_2 : List Nat := [8009]
set_option maxRecDepth 4096 in
theorem innerCandidates694_2_checked : candidates edge 4 5237 5783 innerTail694_3 best.bound = innerCandidates694_2 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan694_2 : scan edge (search edge 3) 4 [5237] 5237 innerTail694_2 best = best := by
  rw [innerTail694_2, scan, if_neg (by decide), innerCandidates694_2_checked]
  change scan edge (search edge 3) 4 [5237] 5237 innerTail694_3
    (if 3 ≤ innerCandidates694_2.length then search edge 3 [5237, 5783] innerCandidates694_2 11020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan694_3

def innerTail694_1 : List Nat := 5693 :: innerTail694_2
def innerCandidates694_1 : List Nat := [6563]
set_option maxRecDepth 4096 in
theorem innerCandidates694_1_checked : candidates edge 4 5237 5693 innerTail694_2 best.bound = innerCandidates694_1 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan694_1 : scan edge (search edge 3) 4 [5237] 5237 innerTail694_1 best = best := by
  rw [innerTail694_1, scan, if_neg (by decide), innerCandidates694_1_checked]
  change scan edge (search edge 3) 4 [5237] 5237 innerTail694_2
    (if 3 ≤ innerCandidates694_1.length then search edge 3 [5237, 5693] innerCandidates694_1 10930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan694_2

def innerTail694_0 : List Nat := 5471 :: innerTail694_1
def innerCandidates694_0 : List Nat := [5783]
set_option maxRecDepth 4096 in
theorem innerCandidates694_0_checked : candidates edge 4 5237 5471 innerTail694_1 best.bound = innerCandidates694_0 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan694_0 : scan edge (search edge 3) 4 [5237] 5237 innerTail694_0 best = best := by
  rw [innerTail694_0, scan, if_neg (by decide), innerCandidates694_0_checked]
  change scan edge (search edge 3) 4 [5237] 5237 innerTail694_1
    (if 3 ≤ innerCandidates694_0.length then search edge 3 [5237, 5471] innerCandidates694_0 10708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan694_1

theorem rootBranch694 : search edge 4 [5237] rootCandidates694 5237 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5237] 5237 innerTail694_0 best = best
  exact innerScan694_0

def rootCandidates693 : List Nat := [5323, 5779, 6343, 6367, 6373, 6451, 7309, 7639, 8089]
set_option maxRecDepth 4096 in
theorem rootCandidates693_checked : candidates edge 5 0 5233 tail694 best.bound = rootCandidates693 := by
  unfold candidates
  have row : edge 5233 = SearchData.row5233.contains := by funext q; rfl
  rw [row]
  decide

def innerTail693_9 : List Nat := []
theorem innerScan693_9 : scan edge (search edge 3) 4 [5233] 5233 innerTail693_9 best = best := rfl

def innerTail693_8 : List Nat := 8089 :: innerTail693_9
def innerCandidates693_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates693_8_checked : candidates edge 4 5233 8089 innerTail693_9 best.bound = innerCandidates693_8 := by
  unfold candidates
  have row : edge 8089 = SearchData.row8089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan693_8 : scan edge (search edge 3) 4 [5233] 5233 innerTail693_8 best = best := by
  rw [innerTail693_8, scan, if_neg (by decide), innerCandidates693_8_checked]
  change scan edge (search edge 3) 4 [5233] 5233 innerTail693_9
    (if 3 ≤ innerCandidates693_8.length then search edge 3 [5233, 8089] innerCandidates693_8 13322 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan693_9

def innerTail693_7 : List Nat := 7639 :: innerTail693_8
def innerCandidates693_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates693_7_checked : candidates edge 4 5233 7639 innerTail693_8 best.bound = innerCandidates693_7 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan693_7 : scan edge (search edge 3) 4 [5233] 5233 innerTail693_7 best = best := by
  rw [innerTail693_7, scan, if_neg (by decide), innerCandidates693_7_checked]
  change scan edge (search edge 3) 4 [5233] 5233 innerTail693_8
    (if 3 ≤ innerCandidates693_7.length then search edge 3 [5233, 7639] innerCandidates693_7 12872 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan693_8

def innerTail693_6 : List Nat := 7309 :: innerTail693_7
def innerCandidates693_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates693_6_checked : candidates edge 4 5233 7309 innerTail693_7 best.bound = innerCandidates693_6 := by
  unfold candidates
  have row : edge 7309 = SearchData.row7309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan693_6 : scan edge (search edge 3) 4 [5233] 5233 innerTail693_6 best = best := by
  rw [innerTail693_6, scan, if_neg (by decide), innerCandidates693_6_checked]
  change scan edge (search edge 3) 4 [5233] 5233 innerTail693_7
    (if 3 ≤ innerCandidates693_6.length then search edge 3 [5233, 7309] innerCandidates693_6 12542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan693_7

def innerTail693_5 : List Nat := 6451 :: innerTail693_6
def innerCandidates693_5 : List Nat := [7309]
set_option maxRecDepth 4096 in
theorem innerCandidates693_5_checked : candidates edge 4 5233 6451 innerTail693_6 best.bound = innerCandidates693_5 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan693_5 : scan edge (search edge 3) 4 [5233] 5233 innerTail693_5 best = best := by
  rw [innerTail693_5, scan, if_neg (by decide), innerCandidates693_5_checked]
  change scan edge (search edge 3) 4 [5233] 5233 innerTail693_6
    (if 3 ≤ innerCandidates693_5.length then search edge 3 [5233, 6451] innerCandidates693_5 11684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan693_6

def innerTail693_4 : List Nat := 6373 :: innerTail693_5
def innerCandidates693_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates693_4_checked : candidates edge 4 5233 6373 innerTail693_5 best.bound = innerCandidates693_4 := by
  unfold candidates
  have row : edge 6373 = SearchData.row6373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan693_4 : scan edge (search edge 3) 4 [5233] 5233 innerTail693_4 best = best := by
  rw [innerTail693_4, scan, if_neg (by decide), innerCandidates693_4_checked]
  change scan edge (search edge 3) 4 [5233] 5233 innerTail693_5
    (if 3 ≤ innerCandidates693_4.length then search edge 3 [5233, 6373] innerCandidates693_4 11606 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan693_5

def innerTail693_3 : List Nat := 6367 :: innerTail693_4
def innerCandidates693_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates693_3_checked : candidates edge 4 5233 6367 innerTail693_4 best.bound = innerCandidates693_3 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan693_3 : scan edge (search edge 3) 4 [5233] 5233 innerTail693_3 best = best := by
  rw [innerTail693_3, scan, if_neg (by decide), innerCandidates693_3_checked]
  change scan edge (search edge 3) 4 [5233] 5233 innerTail693_4
    (if 3 ≤ innerCandidates693_3.length then search edge 3 [5233, 6367] innerCandidates693_3 11600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan693_4

def innerTail693_2 : List Nat := 6343 :: innerTail693_3
def innerCandidates693_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates693_2_checked : candidates edge 4 5233 6343 innerTail693_3 best.bound = innerCandidates693_2 := by
  unfold candidates
  have row : edge 6343 = SearchData.row6343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan693_2 : scan edge (search edge 3) 4 [5233] 5233 innerTail693_2 best = best := by
  rw [innerTail693_2, scan, if_neg (by decide), innerCandidates693_2_checked]
  change scan edge (search edge 3) 4 [5233] 5233 innerTail693_3
    (if 3 ≤ innerCandidates693_2.length then search edge 3 [5233, 6343] innerCandidates693_2 11576 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan693_3

def innerTail693_1 : List Nat := 5779 :: innerTail693_2
def innerCandidates693_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates693_1_checked : candidates edge 4 5233 5779 innerTail693_2 best.bound = innerCandidates693_1 := by
  unfold candidates
  have row : edge 5779 = SearchData.row5779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan693_1 : scan edge (search edge 3) 4 [5233] 5233 innerTail693_1 best = best := by
  rw [innerTail693_1, scan, if_neg (by decide), innerCandidates693_1_checked]
  change scan edge (search edge 3) 4 [5233] 5233 innerTail693_2
    (if 3 ≤ innerCandidates693_1.length then search edge 3 [5233, 5779] innerCandidates693_1 11012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan693_2

def innerTail693_0 : List Nat := 5323 :: innerTail693_1
def innerCandidates693_0 : List Nat := [5779, 8089]
set_option maxRecDepth 4096 in
theorem innerCandidates693_0_checked : candidates edge 4 5233 5323 innerTail693_1 best.bound = innerCandidates693_0 := by
  unfold candidates
  have row : edge 5323 = SearchData.row5323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan693_0 : scan edge (search edge 3) 4 [5233] 5233 innerTail693_0 best = best := by
  rw [innerTail693_0, scan, if_neg (by decide), innerCandidates693_0_checked]
  change scan edge (search edge 3) 4 [5233] 5233 innerTail693_1
    (if 3 ≤ innerCandidates693_0.length then search edge 3 [5233, 5323] innerCandidates693_0 10556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan693_1

theorem rootBranch693 : search edge 4 [5233] rootCandidates693 5233 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5233] 5233 innerTail693_0 best = best
  exact innerScan693_0

def rootCandidates692 : List Nat := [5387, 5843, 6581, 7589, 7949]
set_option maxRecDepth 4096 in
theorem rootCandidates692_checked : candidates edge 5 0 5231 tail693 best.bound = rootCandidates692 := by
  unfold candidates
  have row : edge 5231 = SearchData.row5231.contains := by funext q; rfl
  rw [row]
  decide

def innerTail692_5 : List Nat := []
theorem innerScan692_5 : scan edge (search edge 3) 4 [5231] 5231 innerTail692_5 best = best := rfl

def innerTail692_4 : List Nat := 7949 :: innerTail692_5
def innerCandidates692_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates692_4_checked : candidates edge 4 5231 7949 innerTail692_5 best.bound = innerCandidates692_4 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan692_4 : scan edge (search edge 3) 4 [5231] 5231 innerTail692_4 best = best := by
  rw [innerTail692_4, scan, if_neg (by decide), innerCandidates692_4_checked]
  change scan edge (search edge 3) 4 [5231] 5231 innerTail692_5
    (if 3 ≤ innerCandidates692_4.length then search edge 3 [5231, 7949] innerCandidates692_4 13180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan692_5

def innerTail692_3 : List Nat := 7589 :: innerTail692_4
def innerCandidates692_3 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates692_3_checked : candidates edge 4 5231 7589 innerTail692_4 best.bound = innerCandidates692_3 := by
  unfold candidates
  have row : edge 7589 = SearchData.row7589.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan692_3 : scan edge (search edge 3) 4 [5231] 5231 innerTail692_3 best = best := by
  rw [innerTail692_3, scan, if_neg (by decide), innerCandidates692_3_checked]
  change scan edge (search edge 3) 4 [5231] 5231 innerTail692_4
    (if 3 ≤ innerCandidates692_3.length then search edge 3 [5231, 7589] innerCandidates692_3 12820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan692_4

def innerTail692_2 : List Nat := 6581 :: innerTail692_3
def innerCandidates692_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates692_2_checked : candidates edge 4 5231 6581 innerTail692_3 best.bound = innerCandidates692_2 := by
  unfold candidates
  have row : edge 6581 = SearchData.row6581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan692_2 : scan edge (search edge 3) 4 [5231] 5231 innerTail692_2 best = best := by
  rw [innerTail692_2, scan, if_neg (by decide), innerCandidates692_2_checked]
  change scan edge (search edge 3) 4 [5231] 5231 innerTail692_3
    (if 3 ≤ innerCandidates692_2.length then search edge 3 [5231, 6581] innerCandidates692_2 11812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan692_3

def innerTail692_1 : List Nat := 5843 :: innerTail692_2
def innerCandidates692_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates692_1_checked : candidates edge 4 5231 5843 innerTail692_2 best.bound = innerCandidates692_1 := by
  unfold candidates
  have row : edge 5843 = SearchData.row5843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan692_1 : scan edge (search edge 3) 4 [5231] 5231 innerTail692_1 best = best := by
  rw [innerTail692_1, scan, if_neg (by decide), innerCandidates692_1_checked]
  change scan edge (search edge 3) 4 [5231] 5231 innerTail692_2
    (if 3 ≤ innerCandidates692_1.length then search edge 3 [5231, 5843] innerCandidates692_1 11074 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan692_2

def innerTail692_0 : List Nat := 5387 :: innerTail692_1
def innerCandidates692_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates692_0_checked : candidates edge 4 5231 5387 innerTail692_1 best.bound = innerCandidates692_0 := by
  unfold candidates
  have row : edge 5387 = SearchData.row5387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan692_0 : scan edge (search edge 3) 4 [5231] 5231 innerTail692_0 best = best := by
  rw [innerTail692_0, scan, if_neg (by decide), innerCandidates692_0_checked]
  change scan edge (search edge 3) 4 [5231] 5231 innerTail692_1
    (if 3 ≤ innerCandidates692_0.length then search edge 3 [5231, 5387] innerCandidates692_0 10618 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan692_1

theorem rootBranch692 : search edge 4 [5231] rootCandidates692 5231 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5231] 5231 innerTail692_0 best = best
  exact innerScan692_0

def rootCandidates691 : List Nat := [5683, 6229, 6379, 6481, 7243, 7591, 8353]
set_option maxRecDepth 4096 in
theorem rootCandidates691_checked : candidates edge 5 0 5227 tail692 best.bound = rootCandidates691 := by
  unfold candidates
  have row : edge 5227 = SearchData.row5227.contains := by funext q; rfl
  rw [row]
  decide

def innerTail691_7 : List Nat := []
theorem innerScan691_7 : scan edge (search edge 3) 4 [5227] 5227 innerTail691_7 best = best := rfl

def innerTail691_6 : List Nat := 8353 :: innerTail691_7
def innerCandidates691_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates691_6_checked : candidates edge 4 5227 8353 innerTail691_7 best.bound = innerCandidates691_6 := by
  unfold candidates
  have row : edge 8353 = SearchData.row8353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan691_6 : scan edge (search edge 3) 4 [5227] 5227 innerTail691_6 best = best := by
  rw [innerTail691_6, scan, if_neg (by decide), innerCandidates691_6_checked]
  change scan edge (search edge 3) 4 [5227] 5227 innerTail691_7
    (if 3 ≤ innerCandidates691_6.length then search edge 3 [5227, 8353] innerCandidates691_6 13580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan691_7

def innerTail691_5 : List Nat := 7591 :: innerTail691_6
def innerCandidates691_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates691_5_checked : candidates edge 4 5227 7591 innerTail691_6 best.bound = innerCandidates691_5 := by
  unfold candidates
  have row : edge 7591 = SearchData.row7591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan691_5 : scan edge (search edge 3) 4 [5227] 5227 innerTail691_5 best = best := by
  rw [innerTail691_5, scan, if_neg (by decide), innerCandidates691_5_checked]
  change scan edge (search edge 3) 4 [5227] 5227 innerTail691_6
    (if 3 ≤ innerCandidates691_5.length then search edge 3 [5227, 7591] innerCandidates691_5 12818 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan691_6

def innerTail691_4 : List Nat := 7243 :: innerTail691_5
def innerCandidates691_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates691_4_checked : candidates edge 4 5227 7243 innerTail691_5 best.bound = innerCandidates691_4 := by
  unfold candidates
  have row : edge 7243 = SearchData.row7243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan691_4 : scan edge (search edge 3) 4 [5227] 5227 innerTail691_4 best = best := by
  rw [innerTail691_4, scan, if_neg (by decide), innerCandidates691_4_checked]
  change scan edge (search edge 3) 4 [5227] 5227 innerTail691_5
    (if 3 ≤ innerCandidates691_4.length then search edge 3 [5227, 7243] innerCandidates691_4 12470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan691_5

def innerTail691_3 : List Nat := 6481 :: innerTail691_4
def innerCandidates691_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates691_3_checked : candidates edge 4 5227 6481 innerTail691_4 best.bound = innerCandidates691_3 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan691_3 : scan edge (search edge 3) 4 [5227] 5227 innerTail691_3 best = best := by
  rw [innerTail691_3, scan, if_neg (by decide), innerCandidates691_3_checked]
  change scan edge (search edge 3) 4 [5227] 5227 innerTail691_4
    (if 3 ≤ innerCandidates691_3.length then search edge 3 [5227, 6481] innerCandidates691_3 11708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan691_4

def innerTail691_2 : List Nat := 6379 :: innerTail691_3
def innerCandidates691_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates691_2_checked : candidates edge 4 5227 6379 innerTail691_3 best.bound = innerCandidates691_2 := by
  unfold candidates
  have row : edge 6379 = SearchData.row6379.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan691_2 : scan edge (search edge 3) 4 [5227] 5227 innerTail691_2 best = best := by
  rw [innerTail691_2, scan, if_neg (by decide), innerCandidates691_2_checked]
  change scan edge (search edge 3) 4 [5227] 5227 innerTail691_3
    (if 3 ≤ innerCandidates691_2.length then search edge 3 [5227, 6379] innerCandidates691_2 11606 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan691_3

def innerTail691_1 : List Nat := 6229 :: innerTail691_2
def innerCandidates691_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates691_1_checked : candidates edge 4 5227 6229 innerTail691_2 best.bound = innerCandidates691_1 := by
  unfold candidates
  have row : edge 6229 = SearchData.row6229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan691_1 : scan edge (search edge 3) 4 [5227] 5227 innerTail691_1 best = best := by
  rw [innerTail691_1, scan, if_neg (by decide), innerCandidates691_1_checked]
  change scan edge (search edge 3) 4 [5227] 5227 innerTail691_2
    (if 3 ≤ innerCandidates691_1.length then search edge 3 [5227, 6229] innerCandidates691_1 11456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan691_2

def innerTail691_0 : List Nat := 5683 :: innerTail691_1
def innerCandidates691_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates691_0_checked : candidates edge 4 5227 5683 innerTail691_1 best.bound = innerCandidates691_0 := by
  unfold candidates
  have row : edge 5683 = SearchData.row5683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan691_0 : scan edge (search edge 3) 4 [5227] 5227 innerTail691_0 best = best := by
  rw [innerTail691_0, scan, if_neg (by decide), innerCandidates691_0_checked]
  change scan edge (search edge 3) 4 [5227] 5227 innerTail691_1
    (if 3 ≤ innerCandidates691_0.length then search edge 3 [5227, 5683] innerCandidates691_0 10910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan691_1

theorem rootBranch691 : search edge 4 [5227] rootCandidates691 5227 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5227] 5227 innerTail691_0 best = best
  exact innerScan691_0

def rootCandidates690 : List Nat := [5419, 6397, 7057, 7207, 7417, 7879, 7963, 8191]
set_option maxRecDepth 4096 in
theorem rootCandidates690_checked : candidates edge 5 0 5209 tail691 best.bound = rootCandidates690 := by
  unfold candidates
  have row : edge 5209 = SearchData.row5209.contains := by funext q; rfl
  rw [row]
  decide

def innerTail690_8 : List Nat := []
theorem innerScan690_8 : scan edge (search edge 3) 4 [5209] 5209 innerTail690_8 best = best := rfl

def innerTail690_7 : List Nat := 8191 :: innerTail690_8
def innerCandidates690_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates690_7_checked : candidates edge 4 5209 8191 innerTail690_8 best.bound = innerCandidates690_7 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan690_7 : scan edge (search edge 3) 4 [5209] 5209 innerTail690_7 best = best := by
  rw [innerTail690_7, scan, if_neg (by decide), innerCandidates690_7_checked]
  change scan edge (search edge 3) 4 [5209] 5209 innerTail690_8
    (if 3 ≤ innerCandidates690_7.length then search edge 3 [5209, 8191] innerCandidates690_7 13400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan690_8

def innerTail690_6 : List Nat := 7963 :: innerTail690_7
def innerCandidates690_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates690_6_checked : candidates edge 4 5209 7963 innerTail690_7 best.bound = innerCandidates690_6 := by
  unfold candidates
  have row : edge 7963 = SearchData.row7963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan690_6 : scan edge (search edge 3) 4 [5209] 5209 innerTail690_6 best = best := by
  rw [innerTail690_6, scan, if_neg (by decide), innerCandidates690_6_checked]
  change scan edge (search edge 3) 4 [5209] 5209 innerTail690_7
    (if 3 ≤ innerCandidates690_6.length then search edge 3 [5209, 7963] innerCandidates690_6 13172 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan690_7

def innerTail690_5 : List Nat := 7879 :: innerTail690_6
def innerCandidates690_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates690_5_checked : candidates edge 4 5209 7879 innerTail690_6 best.bound = innerCandidates690_5 := by
  unfold candidates
  have row : edge 7879 = SearchData.row7879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan690_5 : scan edge (search edge 3) 4 [5209] 5209 innerTail690_5 best = best := by
  rw [innerTail690_5, scan, if_neg (by decide), innerCandidates690_5_checked]
  change scan edge (search edge 3) 4 [5209] 5209 innerTail690_6
    (if 3 ≤ innerCandidates690_5.length then search edge 3 [5209, 7879] innerCandidates690_5 13088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan690_6

def innerTail690_4 : List Nat := 7417 :: innerTail690_5
def innerCandidates690_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates690_4_checked : candidates edge 4 5209 7417 innerTail690_5 best.bound = innerCandidates690_4 := by
  unfold candidates
  have row : edge 7417 = SearchData.row7417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan690_4 : scan edge (search edge 3) 4 [5209] 5209 innerTail690_4 best = best := by
  rw [innerTail690_4, scan, if_neg (by decide), innerCandidates690_4_checked]
  change scan edge (search edge 3) 4 [5209] 5209 innerTail690_5
    (if 3 ≤ innerCandidates690_4.length then search edge 3 [5209, 7417] innerCandidates690_4 12626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan690_5

def innerTail690_3 : List Nat := 7207 :: innerTail690_4
def innerCandidates690_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates690_3_checked : candidates edge 4 5209 7207 innerTail690_4 best.bound = innerCandidates690_3 := by
  unfold candidates
  have row : edge 7207 = SearchData.row7207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan690_3 : scan edge (search edge 3) 4 [5209] 5209 innerTail690_3 best = best := by
  rw [innerTail690_3, scan, if_neg (by decide), innerCandidates690_3_checked]
  change scan edge (search edge 3) 4 [5209] 5209 innerTail690_4
    (if 3 ≤ innerCandidates690_3.length then search edge 3 [5209, 7207] innerCandidates690_3 12416 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan690_4

def innerTail690_2 : List Nat := 7057 :: innerTail690_3
def innerCandidates690_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates690_2_checked : candidates edge 4 5209 7057 innerTail690_3 best.bound = innerCandidates690_2 := by
  unfold candidates
  have row : edge 7057 = SearchData.row7057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan690_2 : scan edge (search edge 3) 4 [5209] 5209 innerTail690_2 best = best := by
  rw [innerTail690_2, scan, if_neg (by decide), innerCandidates690_2_checked]
  change scan edge (search edge 3) 4 [5209] 5209 innerTail690_3
    (if 3 ≤ innerCandidates690_2.length then search edge 3 [5209, 7057] innerCandidates690_2 12266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan690_3

def innerTail690_1 : List Nat := 6397 :: innerTail690_2
def innerCandidates690_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates690_1_checked : candidates edge 4 5209 6397 innerTail690_2 best.bound = innerCandidates690_1 := by
  unfold candidates
  have row : edge 6397 = SearchData.row6397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan690_1 : scan edge (search edge 3) 4 [5209] 5209 innerTail690_1 best = best := by
  rw [innerTail690_1, scan, if_neg (by decide), innerCandidates690_1_checked]
  change scan edge (search edge 3) 4 [5209] 5209 innerTail690_2
    (if 3 ≤ innerCandidates690_1.length then search edge 3 [5209, 6397] innerCandidates690_1 11606 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan690_2

def innerTail690_0 : List Nat := 5419 :: innerTail690_1
def innerCandidates690_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates690_0_checked : candidates edge 4 5209 5419 innerTail690_1 best.bound = innerCandidates690_0 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan690_0 : scan edge (search edge 3) 4 [5209] 5209 innerTail690_0 best = best := by
  rw [innerTail690_0, scan, if_neg (by decide), innerCandidates690_0_checked]
  change scan edge (search edge 3) 4 [5209] 5209 innerTail690_1
    (if 3 ≤ innerCandidates690_0.length then search edge 3 [5209, 5419] innerCandidates690_0 10628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan690_1

theorem rootBranch690 : search edge 4 [5209] rootCandidates690 5209 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5209] 5209 innerTail690_0 best = best
  exact innerScan690_0

end ProjectEulerSolutions.PrimePairUpper.Search
