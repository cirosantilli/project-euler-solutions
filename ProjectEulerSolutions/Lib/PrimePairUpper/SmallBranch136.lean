import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates267 : List Nat := [2411, 2657, 2663, 2927, 3347, 3593, 3917, 4001, 4133, 4817, 5051, 5273, 5867, 5897, 6101, 6323, 6521, 6959, 7043, 7757, 7949]
set_option maxRecDepth 4096 in
theorem rootCandidates267_checked : candidates edge 5 0 1733 tail268 best.bound = rootCandidates267 := by
  unfold candidates
  have row : edge 1733 = PrimePairCache.smallRow1733.contains := by funext q; rfl
  rw [row]
  decide

def innerTail267_21 : List Nat := []
theorem innerScan267_21 : scan edge (search edge 3) 4 [1733] 1733 innerTail267_21 best = best := rfl

def innerTail267_20 : List Nat := 7949 :: innerTail267_21
def innerCandidates267_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates267_20_checked : candidates edge 4 1733 7949 innerTail267_21 best.bound = innerCandidates267_20 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan267_20 : scan edge (search edge 3) 4 [1733] 1733 innerTail267_20 best = best := by
  rw [innerTail267_20, scan, if_neg (by decide), innerCandidates267_20_checked]
  change scan edge (search edge 3) 4 [1733] 1733 innerTail267_21
    (if 3 ≤ innerCandidates267_20.length then search edge 3 [1733, 7949] innerCandidates267_20 9682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan267_21

def innerTail267_19 : List Nat := 7757 :: innerTail267_20
def innerCandidates267_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates267_19_checked : candidates edge 4 1733 7757 innerTail267_20 best.bound = innerCandidates267_19 := by
  unfold candidates
  have row : edge 7757 = SearchData.row7757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan267_19 : scan edge (search edge 3) 4 [1733] 1733 innerTail267_19 best = best := by
  rw [innerTail267_19, scan, if_neg (by decide), innerCandidates267_19_checked]
  change scan edge (search edge 3) 4 [1733] 1733 innerTail267_20
    (if 3 ≤ innerCandidates267_19.length then search edge 3 [1733, 7757] innerCandidates267_19 9490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan267_20

def innerTail267_18 : List Nat := 7043 :: innerTail267_19
def innerCandidates267_18 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates267_18_checked : candidates edge 4 1733 7043 innerTail267_19 best.bound = innerCandidates267_18 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan267_18 : scan edge (search edge 3) 4 [1733] 1733 innerTail267_18 best = best := by
  rw [innerTail267_18, scan, if_neg (by decide), innerCandidates267_18_checked]
  change scan edge (search edge 3) 4 [1733] 1733 innerTail267_19
    (if 3 ≤ innerCandidates267_18.length then search edge 3 [1733, 7043] innerCandidates267_18 8776 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan267_19

def innerTail267_17 : List Nat := 6959 :: innerTail267_18
def innerCandidates267_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates267_17_checked : candidates edge 4 1733 6959 innerTail267_18 best.bound = innerCandidates267_17 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan267_17 : scan edge (search edge 3) 4 [1733] 1733 innerTail267_17 best = best := by
  rw [innerTail267_17, scan, if_neg (by decide), innerCandidates267_17_checked]
  change scan edge (search edge 3) 4 [1733] 1733 innerTail267_18
    (if 3 ≤ innerCandidates267_17.length then search edge 3 [1733, 6959] innerCandidates267_17 8692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan267_18

def innerTail267_16 : List Nat := 6521 :: innerTail267_17
def innerCandidates267_16 : List Nat := [7043]
set_option maxRecDepth 4096 in
theorem innerCandidates267_16_checked : candidates edge 4 1733 6521 innerTail267_17 best.bound = innerCandidates267_16 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan267_16 : scan edge (search edge 3) 4 [1733] 1733 innerTail267_16 best = best := by
  rw [innerTail267_16, scan, if_neg (by decide), innerCandidates267_16_checked]
  change scan edge (search edge 3) 4 [1733] 1733 innerTail267_17
    (if 3 ≤ innerCandidates267_16.length then search edge 3 [1733, 6521] innerCandidates267_16 8254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan267_17

def innerTail267_15 : List Nat := 6323 :: innerTail267_16
def innerCandidates267_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates267_15_checked : candidates edge 4 1733 6323 innerTail267_16 best.bound = innerCandidates267_15 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan267_15 : scan edge (search edge 3) 4 [1733] 1733 innerTail267_15 best = best := by
  rw [innerTail267_15, scan, if_neg (by decide), innerCandidates267_15_checked]
  change scan edge (search edge 3) 4 [1733] 1733 innerTail267_16
    (if 3 ≤ innerCandidates267_15.length then search edge 3 [1733, 6323] innerCandidates267_15 8056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan267_16

def innerTail267_14 : List Nat := 6101 :: innerTail267_15
def innerCandidates267_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates267_14_checked : candidates edge 4 1733 6101 innerTail267_15 best.bound = innerCandidates267_14 := by
  unfold candidates
  have row : edge 6101 = SearchData.row6101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan267_14 : scan edge (search edge 3) 4 [1733] 1733 innerTail267_14 best = best := by
  rw [innerTail267_14, scan, if_neg (by decide), innerCandidates267_14_checked]
  change scan edge (search edge 3) 4 [1733] 1733 innerTail267_15
    (if 3 ≤ innerCandidates267_14.length then search edge 3 [1733, 6101] innerCandidates267_14 7834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan267_15

def innerTail267_13 : List Nat := 5897 :: innerTail267_14
def innerCandidates267_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates267_13_checked : candidates edge 4 1733 5897 innerTail267_14 best.bound = innerCandidates267_13 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan267_13 : scan edge (search edge 3) 4 [1733] 1733 innerTail267_13 best = best := by
  rw [innerTail267_13, scan, if_neg (by decide), innerCandidates267_13_checked]
  change scan edge (search edge 3) 4 [1733] 1733 innerTail267_14
    (if 3 ≤ innerCandidates267_13.length then search edge 3 [1733, 5897] innerCandidates267_13 7630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan267_14

def innerTail267_12 : List Nat := 5867 :: innerTail267_13
def innerCandidates267_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates267_12_checked : candidates edge 4 1733 5867 innerTail267_13 best.bound = innerCandidates267_12 := by
  unfold candidates
  have row : edge 5867 = SearchData.row5867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan267_12 : scan edge (search edge 3) 4 [1733] 1733 innerTail267_12 best = best := by
  rw [innerTail267_12, scan, if_neg (by decide), innerCandidates267_12_checked]
  change scan edge (search edge 3) 4 [1733] 1733 innerTail267_13
    (if 3 ≤ innerCandidates267_12.length then search edge 3 [1733, 5867] innerCandidates267_12 7600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan267_13

def innerTail267_11 : List Nat := 5273 :: innerTail267_12
def innerCandidates267_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates267_11_checked : candidates edge 4 1733 5273 innerTail267_12 best.bound = innerCandidates267_11 := by
  unfold candidates
  have row : edge 5273 = SearchData.row5273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan267_11 : scan edge (search edge 3) 4 [1733] 1733 innerTail267_11 best = best := by
  rw [innerTail267_11, scan, if_neg (by decide), innerCandidates267_11_checked]
  change scan edge (search edge 3) 4 [1733] 1733 innerTail267_12
    (if 3 ≤ innerCandidates267_11.length then search edge 3 [1733, 5273] innerCandidates267_11 7006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan267_12

def innerTail267_10 : List Nat := 5051 :: innerTail267_11
def innerCandidates267_10 : List Nat := [5867, 6101, 7949]
set_option maxRecDepth 4096 in
theorem innerCandidates267_10_checked : candidates edge 4 1733 5051 innerTail267_11 best.bound = innerCandidates267_10 := by
  unfold candidates
  have row : edge 5051 = SearchData.row5051.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild267_10 : search edge 3 [1733, 5051] innerCandidates267_10 6784 best = best := by decide

theorem innerScan267_10 : scan edge (search edge 3) 4 [1733] 1733 innerTail267_10 best = best := by
  rw [innerTail267_10, scan, if_neg (by decide), innerCandidates267_10_checked]
  change scan edge (search edge 3) 4 [1733] 1733 innerTail267_11
    (if 3 ≤ innerCandidates267_10.length then search edge 3 [1733, 5051] innerCandidates267_10 6784 best else best) = best
  rw [if_pos (by decide), innerChild267_10]
  exact innerScan267_11

def innerTail267_9 : List Nat := 4817 :: innerTail267_10
def innerCandidates267_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates267_9_checked : candidates edge 4 1733 4817 innerTail267_10 best.bound = innerCandidates267_9 := by
  unfold candidates
  have row : edge 4817 = SearchData.row4817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan267_9 : scan edge (search edge 3) 4 [1733] 1733 innerTail267_9 best = best := by
  rw [innerTail267_9, scan, if_neg (by decide), innerCandidates267_9_checked]
  change scan edge (search edge 3) 4 [1733] 1733 innerTail267_10
    (if 3 ≤ innerCandidates267_9.length then search edge 3 [1733, 4817] innerCandidates267_9 6550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan267_10

def innerTail267_8 : List Nat := 4133 :: innerTail267_9
def innerCandidates267_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates267_8_checked : candidates edge 4 1733 4133 innerTail267_9 best.bound = innerCandidates267_8 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan267_8 : scan edge (search edge 3) 4 [1733] 1733 innerTail267_8 best = best := by
  rw [innerTail267_8, scan, if_neg (by decide), innerCandidates267_8_checked]
  change scan edge (search edge 3) 4 [1733] 1733 innerTail267_9
    (if 3 ≤ innerCandidates267_8.length then search edge 3 [1733, 4133] innerCandidates267_8 5866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan267_9

def innerTail267_7 : List Nat := 4001 :: innerTail267_8
def innerCandidates267_7 : List Nat := [6323]
set_option maxRecDepth 4096 in
theorem innerCandidates267_7_checked : candidates edge 4 1733 4001 innerTail267_8 best.bound = innerCandidates267_7 := by
  unfold candidates
  have row : edge 4001 = SearchData.row4001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan267_7 : scan edge (search edge 3) 4 [1733] 1733 innerTail267_7 best = best := by
  rw [innerTail267_7, scan, if_neg (by decide), innerCandidates267_7_checked]
  change scan edge (search edge 3) 4 [1733] 1733 innerTail267_8
    (if 3 ≤ innerCandidates267_7.length then search edge 3 [1733, 4001] innerCandidates267_7 5734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan267_8

def innerTail267_6 : List Nat := 3917 :: innerTail267_7
def innerCandidates267_6 : List Nat := [6323]
set_option maxRecDepth 4096 in
theorem innerCandidates267_6_checked : candidates edge 4 1733 3917 innerTail267_7 best.bound = innerCandidates267_6 := by
  unfold candidates
  have row : edge 3917 = SearchData.row3917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan267_6 : scan edge (search edge 3) 4 [1733] 1733 innerTail267_6 best = best := by
  rw [innerTail267_6, scan, if_neg (by decide), innerCandidates267_6_checked]
  change scan edge (search edge 3) 4 [1733] 1733 innerTail267_7
    (if 3 ≤ innerCandidates267_6.length then search edge 3 [1733, 3917] innerCandidates267_6 5650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan267_7

def innerTail267_5 : List Nat := 3593 :: innerTail267_6
def innerCandidates267_5 : List Nat := [4001]
set_option maxRecDepth 4096 in
theorem innerCandidates267_5_checked : candidates edge 4 1733 3593 innerTail267_6 best.bound = innerCandidates267_5 := by
  unfold candidates
  have row : edge 3593 = SearchData.row3593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan267_5 : scan edge (search edge 3) 4 [1733] 1733 innerTail267_5 best = best := by
  rw [innerTail267_5, scan, if_neg (by decide), innerCandidates267_5_checked]
  change scan edge (search edge 3) 4 [1733] 1733 innerTail267_6
    (if 3 ≤ innerCandidates267_5.length then search edge 3 [1733, 3593] innerCandidates267_5 5326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan267_6

def innerTail267_4 : List Nat := 3347 :: innerTail267_5
def innerCandidates267_4 : List Nat := [6101, 6959, 7043]
set_option maxRecDepth 4096 in
theorem innerCandidates267_4_checked : candidates edge 4 1733 3347 innerTail267_5 best.bound = innerCandidates267_4 := by
  unfold candidates
  have row : edge 3347 = SearchData.row3347.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild267_4 : search edge 3 [1733, 3347] innerCandidates267_4 5080 best = best := by decide

theorem innerScan267_4 : scan edge (search edge 3) 4 [1733] 1733 innerTail267_4 best = best := by
  rw [innerTail267_4, scan, if_neg (by decide), innerCandidates267_4_checked]
  change scan edge (search edge 3) 4 [1733] 1733 innerTail267_5
    (if 3 ≤ innerCandidates267_4.length then search edge 3 [1733, 3347] innerCandidates267_4 5080 best else best) = best
  rw [if_pos (by decide), innerChild267_4]
  exact innerScan267_5

def innerTail267_3 : List Nat := 2927 :: innerTail267_4
def innerCandidates267_3 : List Nat := [3593]
set_option maxRecDepth 4096 in
theorem innerCandidates267_3_checked : candidates edge 4 1733 2927 innerTail267_4 best.bound = innerCandidates267_3 := by
  unfold candidates
  have row : edge 2927 = SearchData.row2927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan267_3 : scan edge (search edge 3) 4 [1733] 1733 innerTail267_3 best = best := by
  rw [innerTail267_3, scan, if_neg (by decide), innerCandidates267_3_checked]
  change scan edge (search edge 3) 4 [1733] 1733 innerTail267_4
    (if 3 ≤ innerCandidates267_3.length then search edge 3 [1733, 2927] innerCandidates267_3 4660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan267_4

def innerTail267_2 : List Nat := 2663 :: innerTail267_3
def innerCandidates267_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates267_2_checked : candidates edge 4 1733 2663 innerTail267_3 best.bound = innerCandidates267_2 := by
  unfold candidates
  have row : edge 2663 = PrimePairCache.smallRow2663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan267_2 : scan edge (search edge 3) 4 [1733] 1733 innerTail267_2 best = best := by
  rw [innerTail267_2, scan, if_neg (by decide), innerCandidates267_2_checked]
  change scan edge (search edge 3) 4 [1733] 1733 innerTail267_3
    (if 3 ≤ innerCandidates267_2.length then search edge 3 [1733, 2663] innerCandidates267_2 4396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan267_3

def innerTail267_1 : List Nat := 2657 :: innerTail267_2
def innerCandidates267_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates267_1_checked : candidates edge 4 1733 2657 innerTail267_2 best.bound = innerCandidates267_1 := by
  unfold candidates
  have row : edge 2657 = PrimePairCache.smallRow2657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan267_1 : scan edge (search edge 3) 4 [1733] 1733 innerTail267_1 best = best := by
  rw [innerTail267_1, scan, if_neg (by decide), innerCandidates267_1_checked]
  change scan edge (search edge 3) 4 [1733] 1733 innerTail267_2
    (if 3 ≤ innerCandidates267_1.length then search edge 3 [1733, 2657] innerCandidates267_1 4390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan267_2

def innerTail267_0 : List Nat := 2411 :: innerTail267_1
def innerCandidates267_0 : List Nat := [3917]
set_option maxRecDepth 4096 in
theorem innerCandidates267_0_checked : candidates edge 4 1733 2411 innerTail267_1 best.bound = innerCandidates267_0 := by
  unfold candidates
  have row : edge 2411 = PrimePairCache.smallRow2411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan267_0 : scan edge (search edge 3) 4 [1733] 1733 innerTail267_0 best = best := by
  rw [innerTail267_0, scan, if_neg (by decide), innerCandidates267_0_checked]
  change scan edge (search edge 3) 4 [1733] 1733 innerTail267_1
    (if 3 ≤ innerCandidates267_0.length then search edge 3 [1733, 2411] innerCandidates267_0 4144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan267_1

theorem rootBranch267 : search edge 4 [1733] rootCandidates267 1733 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1733] 1733 innerTail267_0 best = best
  exact innerScan267_0

def rootCandidates266 : List Nat := [1831, 2179, 3319, 3457, 4057, 4363, 4423, 4621, 4783, 4813, 4993, 4999, 5419, 5431, 6397, 6547, 7951, 7993, 8233, 8329]
set_option maxRecDepth 4096 in
theorem rootCandidates266_checked : candidates edge 5 0 1723 tail267 best.bound = rootCandidates266 := by
  unfold candidates
  have row : edge 1723 = PrimePairCache.smallRow1723.contains := by funext q; rfl
  rw [row]
  decide

def innerTail266_20 : List Nat := []
theorem innerScan266_20 : scan edge (search edge 3) 4 [1723] 1723 innerTail266_20 best = best := rfl

def innerTail266_19 : List Nat := 8329 :: innerTail266_20
def innerCandidates266_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates266_19_checked : candidates edge 4 1723 8329 innerTail266_20 best.bound = innerCandidates266_19 := by
  unfold candidates
  have row : edge 8329 = SearchData.row8329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan266_19 : scan edge (search edge 3) 4 [1723] 1723 innerTail266_19 best = best := by
  rw [innerTail266_19, scan, if_neg (by decide), innerCandidates266_19_checked]
  change scan edge (search edge 3) 4 [1723] 1723 innerTail266_20
    (if 3 ≤ innerCandidates266_19.length then search edge 3 [1723, 8329] innerCandidates266_19 10052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan266_20

def innerTail266_18 : List Nat := 8233 :: innerTail266_19
def innerCandidates266_18 : List Nat := [8329]
set_option maxRecDepth 4096 in
theorem innerCandidates266_18_checked : candidates edge 4 1723 8233 innerTail266_19 best.bound = innerCandidates266_18 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan266_18 : scan edge (search edge 3) 4 [1723] 1723 innerTail266_18 best = best := by
  rw [innerTail266_18, scan, if_neg (by decide), innerCandidates266_18_checked]
  change scan edge (search edge 3) 4 [1723] 1723 innerTail266_19
    (if 3 ≤ innerCandidates266_18.length then search edge 3 [1723, 8233] innerCandidates266_18 9956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan266_19

def innerTail266_17 : List Nat := 7993 :: innerTail266_18
def innerCandidates266_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates266_17_checked : candidates edge 4 1723 7993 innerTail266_18 best.bound = innerCandidates266_17 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan266_17 : scan edge (search edge 3) 4 [1723] 1723 innerTail266_17 best = best := by
  rw [innerTail266_17, scan, if_neg (by decide), innerCandidates266_17_checked]
  change scan edge (search edge 3) 4 [1723] 1723 innerTail266_18
    (if 3 ≤ innerCandidates266_17.length then search edge 3 [1723, 7993] innerCandidates266_17 9716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan266_18

def innerTail266_16 : List Nat := 7951 :: innerTail266_17
def innerCandidates266_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates266_16_checked : candidates edge 4 1723 7951 innerTail266_17 best.bound = innerCandidates266_16 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan266_16 : scan edge (search edge 3) 4 [1723] 1723 innerTail266_16 best = best := by
  rw [innerTail266_16, scan, if_neg (by decide), innerCandidates266_16_checked]
  change scan edge (search edge 3) 4 [1723] 1723 innerTail266_17
    (if 3 ≤ innerCandidates266_16.length then search edge 3 [1723, 7951] innerCandidates266_16 9674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan266_17

def innerTail266_15 : List Nat := 6547 :: innerTail266_16
def innerCandidates266_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates266_15_checked : candidates edge 4 1723 6547 innerTail266_16 best.bound = innerCandidates266_15 := by
  unfold candidates
  have row : edge 6547 = SearchData.row6547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan266_15 : scan edge (search edge 3) 4 [1723] 1723 innerTail266_15 best = best := by
  rw [innerTail266_15, scan, if_neg (by decide), innerCandidates266_15_checked]
  change scan edge (search edge 3) 4 [1723] 1723 innerTail266_16
    (if 3 ≤ innerCandidates266_15.length then search edge 3 [1723, 6547] innerCandidates266_15 8270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan266_16

def innerTail266_14 : List Nat := 6397 :: innerTail266_15
def innerCandidates266_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates266_14_checked : candidates edge 4 1723 6397 innerTail266_15 best.bound = innerCandidates266_14 := by
  unfold candidates
  have row : edge 6397 = SearchData.row6397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan266_14 : scan edge (search edge 3) 4 [1723] 1723 innerTail266_14 best = best := by
  rw [innerTail266_14, scan, if_neg (by decide), innerCandidates266_14_checked]
  change scan edge (search edge 3) 4 [1723] 1723 innerTail266_15
    (if 3 ≤ innerCandidates266_14.length then search edge 3 [1723, 6397] innerCandidates266_14 8120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan266_15

def innerTail266_13 : List Nat := 5431 :: innerTail266_14
def innerCandidates266_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates266_13_checked : candidates edge 4 1723 5431 innerTail266_14 best.bound = innerCandidates266_13 := by
  unfold candidates
  have row : edge 5431 = SearchData.row5431.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan266_13 : scan edge (search edge 3) 4 [1723] 1723 innerTail266_13 best = best := by
  rw [innerTail266_13, scan, if_neg (by decide), innerCandidates266_13_checked]
  change scan edge (search edge 3) 4 [1723] 1723 innerTail266_14
    (if 3 ≤ innerCandidates266_13.length then search edge 3 [1723, 5431] innerCandidates266_13 7154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan266_14

def innerTail266_12 : List Nat := 5419 :: innerTail266_13
def innerCandidates266_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates266_12_checked : candidates edge 4 1723 5419 innerTail266_13 best.bound = innerCandidates266_12 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan266_12 : scan edge (search edge 3) 4 [1723] 1723 innerTail266_12 best = best := by
  rw [innerTail266_12, scan, if_neg (by decide), innerCandidates266_12_checked]
  change scan edge (search edge 3) 4 [1723] 1723 innerTail266_13
    (if 3 ≤ innerCandidates266_12.length then search edge 3 [1723, 5419] innerCandidates266_12 7142 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan266_13

def innerTail266_11 : List Nat := 4999 :: innerTail266_12
def innerCandidates266_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates266_11_checked : candidates edge 4 1723 4999 innerTail266_12 best.bound = innerCandidates266_11 := by
  unfold candidates
  have row : edge 4999 = SearchData.row4999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan266_11 : scan edge (search edge 3) 4 [1723] 1723 innerTail266_11 best = best := by
  rw [innerTail266_11, scan, if_neg (by decide), innerCandidates266_11_checked]
  change scan edge (search edge 3) 4 [1723] 1723 innerTail266_12
    (if 3 ≤ innerCandidates266_11.length then search edge 3 [1723, 4999] innerCandidates266_11 6722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan266_12

def innerTail266_10 : List Nat := 4993 :: innerTail266_11
def innerCandidates266_10 : List Nat := [4999]
set_option maxRecDepth 4096 in
theorem innerCandidates266_10_checked : candidates edge 4 1723 4993 innerTail266_11 best.bound = innerCandidates266_10 := by
  unfold candidates
  have row : edge 4993 = SearchData.row4993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan266_10 : scan edge (search edge 3) 4 [1723] 1723 innerTail266_10 best = best := by
  rw [innerTail266_10, scan, if_neg (by decide), innerCandidates266_10_checked]
  change scan edge (search edge 3) 4 [1723] 1723 innerTail266_11
    (if 3 ≤ innerCandidates266_10.length then search edge 3 [1723, 4993] innerCandidates266_10 6716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan266_11

def innerTail266_9 : List Nat := 4813 :: innerTail266_10
def innerCandidates266_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates266_9_checked : candidates edge 4 1723 4813 innerTail266_10 best.bound = innerCandidates266_9 := by
  unfold candidates
  have row : edge 4813 = SearchData.row4813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan266_9 : scan edge (search edge 3) 4 [1723] 1723 innerTail266_9 best = best := by
  rw [innerTail266_9, scan, if_neg (by decide), innerCandidates266_9_checked]
  change scan edge (search edge 3) 4 [1723] 1723 innerTail266_10
    (if 3 ≤ innerCandidates266_9.length then search edge 3 [1723, 4813] innerCandidates266_9 6536 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan266_10

def innerTail266_8 : List Nat := 4783 :: innerTail266_9
def innerCandidates266_8 : List Nat := [4999]
set_option maxRecDepth 4096 in
theorem innerCandidates266_8_checked : candidates edge 4 1723 4783 innerTail266_9 best.bound = innerCandidates266_8 := by
  unfold candidates
  have row : edge 4783 = SearchData.row4783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan266_8 : scan edge (search edge 3) 4 [1723] 1723 innerTail266_8 best = best := by
  rw [innerTail266_8, scan, if_neg (by decide), innerCandidates266_8_checked]
  change scan edge (search edge 3) 4 [1723] 1723 innerTail266_9
    (if 3 ≤ innerCandidates266_8.length then search edge 3 [1723, 4783] innerCandidates266_8 6506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan266_9

def innerTail266_7 : List Nat := 4621 :: innerTail266_8
def innerCandidates266_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates266_7_checked : candidates edge 4 1723 4621 innerTail266_8 best.bound = innerCandidates266_7 := by
  unfold candidates
  have row : edge 4621 = SearchData.row4621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan266_7 : scan edge (search edge 3) 4 [1723] 1723 innerTail266_7 best = best := by
  rw [innerTail266_7, scan, if_neg (by decide), innerCandidates266_7_checked]
  change scan edge (search edge 3) 4 [1723] 1723 innerTail266_8
    (if 3 ≤ innerCandidates266_7.length then search edge 3 [1723, 4621] innerCandidates266_7 6344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan266_8

def innerTail266_6 : List Nat := 4423 :: innerTail266_7
def innerCandidates266_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates266_6_checked : candidates edge 4 1723 4423 innerTail266_7 best.bound = innerCandidates266_6 := by
  unfold candidates
  have row : edge 4423 = SearchData.row4423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan266_6 : scan edge (search edge 3) 4 [1723] 1723 innerTail266_6 best = best := by
  rw [innerTail266_6, scan, if_neg (by decide), innerCandidates266_6_checked]
  change scan edge (search edge 3) 4 [1723] 1723 innerTail266_7
    (if 3 ≤ innerCandidates266_6.length then search edge 3 [1723, 4423] innerCandidates266_6 6146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan266_7

def innerTail266_5 : List Nat := 4363 :: innerTail266_6
def innerCandidates266_5 : List Nat := [4813]
set_option maxRecDepth 4096 in
theorem innerCandidates266_5_checked : candidates edge 4 1723 4363 innerTail266_6 best.bound = innerCandidates266_5 := by
  unfold candidates
  have row : edge 4363 = SearchData.row4363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan266_5 : scan edge (search edge 3) 4 [1723] 1723 innerTail266_5 best = best := by
  rw [innerTail266_5, scan, if_neg (by decide), innerCandidates266_5_checked]
  change scan edge (search edge 3) 4 [1723] 1723 innerTail266_6
    (if 3 ≤ innerCandidates266_5.length then search edge 3 [1723, 4363] innerCandidates266_5 6086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan266_6

def innerTail266_4 : List Nat := 4057 :: innerTail266_5
def innerCandidates266_4 : List Nat := [5419]
set_option maxRecDepth 4096 in
theorem innerCandidates266_4_checked : candidates edge 4 1723 4057 innerTail266_5 best.bound = innerCandidates266_4 := by
  unfold candidates
  have row : edge 4057 = SearchData.row4057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan266_4 : scan edge (search edge 3) 4 [1723] 1723 innerTail266_4 best = best := by
  rw [innerTail266_4, scan, if_neg (by decide), innerCandidates266_4_checked]
  change scan edge (search edge 3) 4 [1723] 1723 innerTail266_5
    (if 3 ≤ innerCandidates266_4.length then search edge 3 [1723, 4057] innerCandidates266_4 5780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan266_5

def innerTail266_3 : List Nat := 3457 :: innerTail266_4
def innerCandidates266_3 : List Nat := [5419]
set_option maxRecDepth 4096 in
theorem innerCandidates266_3_checked : candidates edge 4 1723 3457 innerTail266_4 best.bound = innerCandidates266_3 := by
  unfold candidates
  have row : edge 3457 = SearchData.row3457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan266_3 : scan edge (search edge 3) 4 [1723] 1723 innerTail266_3 best = best := by
  rw [innerTail266_3, scan, if_neg (by decide), innerCandidates266_3_checked]
  change scan edge (search edge 3) 4 [1723] 1723 innerTail266_4
    (if 3 ≤ innerCandidates266_3.length then search edge 3 [1723, 3457] innerCandidates266_3 5180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan266_4

def innerTail266_2 : List Nat := 3319 :: innerTail266_3
def innerCandidates266_2 : List Nat := [4363, 5419, 7951]
set_option maxRecDepth 4096 in
theorem innerCandidates266_2_checked : candidates edge 4 1723 3319 innerTail266_3 best.bound = innerCandidates266_2 := by
  unfold candidates
  have row : edge 3319 = SearchData.row3319.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild266_2 : search edge 3 [1723, 3319] innerCandidates266_2 5042 best = best := by decide

theorem innerScan266_2 : scan edge (search edge 3) 4 [1723] 1723 innerTail266_2 best = best := by
  rw [innerTail266_2, scan, if_neg (by decide), innerCandidates266_2_checked]
  change scan edge (search edge 3) 4 [1723] 1723 innerTail266_3
    (if 3 ≤ innerCandidates266_2.length then search edge 3 [1723, 3319] innerCandidates266_2 5042 best else best) = best
  rw [if_pos (by decide), innerChild266_2]
  exact innerScan266_3

def innerTail266_1 : List Nat := 2179 :: innerTail266_2
def innerCandidates266_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates266_1_checked : candidates edge 4 1723 2179 innerTail266_2 best.bound = innerCandidates266_1 := by
  unfold candidates
  have row : edge 2179 = PrimePairCache.smallRow2179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan266_1 : scan edge (search edge 3) 4 [1723] 1723 innerTail266_1 best = best := by
  rw [innerTail266_1, scan, if_neg (by decide), innerCandidates266_1_checked]
  change scan edge (search edge 3) 4 [1723] 1723 innerTail266_2
    (if 3 ≤ innerCandidates266_1.length then search edge 3 [1723, 2179] innerCandidates266_1 3902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan266_2

def innerTail266_0 : List Nat := 1831 :: innerTail266_1
def innerCandidates266_0 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates266_0_checked : candidates edge 4 1723 1831 innerTail266_1 best.bound = innerCandidates266_0 := by
  unfold candidates
  have row : edge 1831 = PrimePairCache.smallRow1831.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan266_0 : scan edge (search edge 3) 4 [1723] 1723 innerTail266_0 best = best := by
  rw [innerTail266_0, scan, if_neg (by decide), innerCandidates266_0_checked]
  change scan edge (search edge 3) 4 [1723] 1723 innerTail266_1
    (if 3 ≤ innerCandidates266_0.length then search edge 3 [1723, 1831] innerCandidates266_0 3554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan266_1

theorem rootBranch266 : search edge 4 [1723] rootCandidates266 1723 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1723] 1723 innerTail266_0 best = best
  exact innerScan266_0

def rootCandidates265 : List Nat := [2213, 2297, 2411, 2621, 3347, 3359, 3557, 4679, 6389, 6779, 7283, 7589, 7829, 8039, 8219, 8231, 8273, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates265_checked : candidates edge 5 0 1721 tail266 best.bound = rootCandidates265 := by
  unfold candidates
  have row : edge 1721 = PrimePairCache.smallRow1721.contains := by funext q; rfl
  rw [row]
  decide

def innerTail265_18 : List Nat := []
theorem innerScan265_18 : scan edge (search edge 3) 4 [1721] 1721 innerTail265_18 best = best := rfl

def innerTail265_17 : List Nat := 8387 :: innerTail265_18
def innerCandidates265_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates265_17_checked : candidates edge 4 1721 8387 innerTail265_18 best.bound = innerCandidates265_17 := by
  unfold candidates
  have row : edge 8387 = SearchData.row8387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan265_17 : scan edge (search edge 3) 4 [1721] 1721 innerTail265_17 best = best := by
  rw [innerTail265_17, scan, if_neg (by decide), innerCandidates265_17_checked]
  change scan edge (search edge 3) 4 [1721] 1721 innerTail265_18
    (if 3 ≤ innerCandidates265_17.length then search edge 3 [1721, 8387] innerCandidates265_17 10108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan265_18

def innerTail265_16 : List Nat := 8273 :: innerTail265_17
def innerCandidates265_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates265_16_checked : candidates edge 4 1721 8273 innerTail265_17 best.bound = innerCandidates265_16 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan265_16 : scan edge (search edge 3) 4 [1721] 1721 innerTail265_16 best = best := by
  rw [innerTail265_16, scan, if_neg (by decide), innerCandidates265_16_checked]
  change scan edge (search edge 3) 4 [1721] 1721 innerTail265_17
    (if 3 ≤ innerCandidates265_16.length then search edge 3 [1721, 8273] innerCandidates265_16 9994 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan265_17

def innerTail265_15 : List Nat := 8231 :: innerTail265_16
def innerCandidates265_15 : List Nat := [8273]
set_option maxRecDepth 4096 in
theorem innerCandidates265_15_checked : candidates edge 4 1721 8231 innerTail265_16 best.bound = innerCandidates265_15 := by
  unfold candidates
  have row : edge 8231 = SearchData.row8231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan265_15 : scan edge (search edge 3) 4 [1721] 1721 innerTail265_15 best = best := by
  rw [innerTail265_15, scan, if_neg (by decide), innerCandidates265_15_checked]
  change scan edge (search edge 3) 4 [1721] 1721 innerTail265_16
    (if 3 ≤ innerCandidates265_15.length then search edge 3 [1721, 8231] innerCandidates265_15 9952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan265_16

def innerTail265_14 : List Nat := 8219 :: innerTail265_15
def innerCandidates265_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates265_14_checked : candidates edge 4 1721 8219 innerTail265_15 best.bound = innerCandidates265_14 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan265_14 : scan edge (search edge 3) 4 [1721] 1721 innerTail265_14 best = best := by
  rw [innerTail265_14, scan, if_neg (by decide), innerCandidates265_14_checked]
  change scan edge (search edge 3) 4 [1721] 1721 innerTail265_15
    (if 3 ≤ innerCandidates265_14.length then search edge 3 [1721, 8219] innerCandidates265_14 9940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan265_15

def innerTail265_13 : List Nat := 8039 :: innerTail265_14
def innerCandidates265_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates265_13_checked : candidates edge 4 1721 8039 innerTail265_14 best.bound = innerCandidates265_13 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan265_13 : scan edge (search edge 3) 4 [1721] 1721 innerTail265_13 best = best := by
  rw [innerTail265_13, scan, if_neg (by decide), innerCandidates265_13_checked]
  change scan edge (search edge 3) 4 [1721] 1721 innerTail265_14
    (if 3 ≤ innerCandidates265_13.length then search edge 3 [1721, 8039] innerCandidates265_13 9760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan265_14

def innerTail265_12 : List Nat := 7829 :: innerTail265_13
def innerCandidates265_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates265_12_checked : candidates edge 4 1721 7829 innerTail265_13 best.bound = innerCandidates265_12 := by
  unfold candidates
  have row : edge 7829 = SearchData.row7829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan265_12 : scan edge (search edge 3) 4 [1721] 1721 innerTail265_12 best = best := by
  rw [innerTail265_12, scan, if_neg (by decide), innerCandidates265_12_checked]
  change scan edge (search edge 3) 4 [1721] 1721 innerTail265_13
    (if 3 ≤ innerCandidates265_12.length then search edge 3 [1721, 7829] innerCandidates265_12 9550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan265_13

def innerTail265_11 : List Nat := 7589 :: innerTail265_12
def innerCandidates265_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates265_11_checked : candidates edge 4 1721 7589 innerTail265_12 best.bound = innerCandidates265_11 := by
  unfold candidates
  have row : edge 7589 = SearchData.row7589.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan265_11 : scan edge (search edge 3) 4 [1721] 1721 innerTail265_11 best = best := by
  rw [innerTail265_11, scan, if_neg (by decide), innerCandidates265_11_checked]
  change scan edge (search edge 3) 4 [1721] 1721 innerTail265_12
    (if 3 ≤ innerCandidates265_11.length then search edge 3 [1721, 7589] innerCandidates265_11 9310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan265_12

def innerTail265_10 : List Nat := 7283 :: innerTail265_11
def innerCandidates265_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates265_10_checked : candidates edge 4 1721 7283 innerTail265_11 best.bound = innerCandidates265_10 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan265_10 : scan edge (search edge 3) 4 [1721] 1721 innerTail265_10 best = best := by
  rw [innerTail265_10, scan, if_neg (by decide), innerCandidates265_10_checked]
  change scan edge (search edge 3) 4 [1721] 1721 innerTail265_11
    (if 3 ≤ innerCandidates265_10.length then search edge 3 [1721, 7283] innerCandidates265_10 9004 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan265_11

def innerTail265_9 : List Nat := 6779 :: innerTail265_10
def innerCandidates265_9 : List Nat := [8231]
set_option maxRecDepth 4096 in
theorem innerCandidates265_9_checked : candidates edge 4 1721 6779 innerTail265_10 best.bound = innerCandidates265_9 := by
  unfold candidates
  have row : edge 6779 = SearchData.row6779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan265_9 : scan edge (search edge 3) 4 [1721] 1721 innerTail265_9 best = best := by
  rw [innerTail265_9, scan, if_neg (by decide), innerCandidates265_9_checked]
  change scan edge (search edge 3) 4 [1721] 1721 innerTail265_10
    (if 3 ≤ innerCandidates265_9.length then search edge 3 [1721, 6779] innerCandidates265_9 8500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan265_10

def innerTail265_8 : List Nat := 6389 :: innerTail265_9
def innerCandidates265_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates265_8_checked : candidates edge 4 1721 6389 innerTail265_9 best.bound = innerCandidates265_8 := by
  unfold candidates
  have row : edge 6389 = SearchData.row6389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan265_8 : scan edge (search edge 3) 4 [1721] 1721 innerTail265_8 best = best := by
  rw [innerTail265_8, scan, if_neg (by decide), innerCandidates265_8_checked]
  change scan edge (search edge 3) 4 [1721] 1721 innerTail265_9
    (if 3 ≤ innerCandidates265_8.length then search edge 3 [1721, 6389] innerCandidates265_8 8110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan265_9

def innerTail265_7 : List Nat := 4679 :: innerTail265_8
def innerCandidates265_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates265_7_checked : candidates edge 4 1721 4679 innerTail265_8 best.bound = innerCandidates265_7 := by
  unfold candidates
  have row : edge 4679 = SearchData.row4679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan265_7 : scan edge (search edge 3) 4 [1721] 1721 innerTail265_7 best = best := by
  rw [innerTail265_7, scan, if_neg (by decide), innerCandidates265_7_checked]
  change scan edge (search edge 3) 4 [1721] 1721 innerTail265_8
    (if 3 ≤ innerCandidates265_7.length then search edge 3 [1721, 4679] innerCandidates265_7 6400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan265_8

def innerTail265_6 : List Nat := 3557 :: innerTail265_7
def innerCandidates265_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates265_6_checked : candidates edge 4 1721 3557 innerTail265_7 best.bound = innerCandidates265_6 := by
  unfold candidates
  have row : edge 3557 = SearchData.row3557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan265_6 : scan edge (search edge 3) 4 [1721] 1721 innerTail265_6 best = best := by
  rw [innerTail265_6, scan, if_neg (by decide), innerCandidates265_6_checked]
  change scan edge (search edge 3) 4 [1721] 1721 innerTail265_7
    (if 3 ≤ innerCandidates265_6.length then search edge 3 [1721, 3557] innerCandidates265_6 5278 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan265_7

def innerTail265_5 : List Nat := 3359 :: innerTail265_6
def innerCandidates265_5 : List Nat := [6779]
set_option maxRecDepth 4096 in
theorem innerCandidates265_5_checked : candidates edge 4 1721 3359 innerTail265_6 best.bound = innerCandidates265_5 := by
  unfold candidates
  have row : edge 3359 = SearchData.row3359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan265_5 : scan edge (search edge 3) 4 [1721] 1721 innerTail265_5 best = best := by
  rw [innerTail265_5, scan, if_neg (by decide), innerCandidates265_5_checked]
  change scan edge (search edge 3) 4 [1721] 1721 innerTail265_6
    (if 3 ≤ innerCandidates265_5.length then search edge 3 [1721, 3359] innerCandidates265_5 5080 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan265_6

def innerTail265_4 : List Nat := 3347 :: innerTail265_5
def innerCandidates265_4 : List Nat := [8231, 8273]
set_option maxRecDepth 4096 in
theorem innerCandidates265_4_checked : candidates edge 4 1721 3347 innerTail265_5 best.bound = innerCandidates265_4 := by
  unfold candidates
  have row : edge 3347 = SearchData.row3347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan265_4 : scan edge (search edge 3) 4 [1721] 1721 innerTail265_4 best = best := by
  rw [innerTail265_4, scan, if_neg (by decide), innerCandidates265_4_checked]
  change scan edge (search edge 3) 4 [1721] 1721 innerTail265_5
    (if 3 ≤ innerCandidates265_4.length then search edge 3 [1721, 3347] innerCandidates265_4 5068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan265_5

def innerTail265_3 : List Nat := 2621 :: innerTail265_4
def innerCandidates265_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates265_3_checked : candidates edge 4 1721 2621 innerTail265_4 best.bound = innerCandidates265_3 := by
  unfold candidates
  have row : edge 2621 = PrimePairCache.smallRow2621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan265_3 : scan edge (search edge 3) 4 [1721] 1721 innerTail265_3 best = best := by
  rw [innerTail265_3, scan, if_neg (by decide), innerCandidates265_3_checked]
  change scan edge (search edge 3) 4 [1721] 1721 innerTail265_4
    (if 3 ≤ innerCandidates265_3.length then search edge 3 [1721, 2621] innerCandidates265_3 4342 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan265_4

def innerTail265_2 : List Nat := 2411 :: innerTail265_3
def innerCandidates265_2 : List Nat := [8273]
set_option maxRecDepth 4096 in
theorem innerCandidates265_2_checked : candidates edge 4 1721 2411 innerTail265_3 best.bound = innerCandidates265_2 := by
  unfold candidates
  have row : edge 2411 = PrimePairCache.smallRow2411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan265_2 : scan edge (search edge 3) 4 [1721] 1721 innerTail265_2 best = best := by
  rw [innerTail265_2, scan, if_neg (by decide), innerCandidates265_2_checked]
  change scan edge (search edge 3) 4 [1721] 1721 innerTail265_3
    (if 3 ≤ innerCandidates265_2.length then search edge 3 [1721, 2411] innerCandidates265_2 4132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan265_3

def innerTail265_1 : List Nat := 2297 :: innerTail265_2
def innerCandidates265_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates265_1_checked : candidates edge 4 1721 2297 innerTail265_2 best.bound = innerCandidates265_1 := by
  unfold candidates
  have row : edge 2297 = PrimePairCache.smallRow2297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan265_1 : scan edge (search edge 3) 4 [1721] 1721 innerTail265_1 best = best := by
  rw [innerTail265_1, scan, if_neg (by decide), innerCandidates265_1_checked]
  change scan edge (search edge 3) 4 [1721] 1721 innerTail265_2
    (if 3 ≤ innerCandidates265_1.length then search edge 3 [1721, 2297] innerCandidates265_1 4018 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan265_2

def innerTail265_0 : List Nat := 2213 :: innerTail265_1
def innerCandidates265_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates265_0_checked : candidates edge 4 1721 2213 innerTail265_1 best.bound = innerCandidates265_0 := by
  unfold candidates
  have row : edge 2213 = PrimePairCache.smallRow2213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan265_0 : scan edge (search edge 3) 4 [1721] 1721 innerTail265_0 best = best := by
  rw [innerTail265_0, scan, if_neg (by decide), innerCandidates265_0_checked]
  change scan edge (search edge 3) 4 [1721] 1721 innerTail265_1
    (if 3 ≤ innerCandidates265_0.length then search edge 3 [1721, 2213] innerCandidates265_0 3934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan265_1

theorem rootBranch265 : search edge 4 [1721] rootCandidates265 1721 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1721] 1721 innerTail265_0 best = best
  exact innerScan265_0

end ProjectEulerSolutions.PrimePairUpper.Search
