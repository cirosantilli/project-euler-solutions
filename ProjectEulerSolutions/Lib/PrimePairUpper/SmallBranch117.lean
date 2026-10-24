import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates331 : List Nat := [2273, 2411, 2789, 3539, 4133, 4457, 5477, 5927, 6599, 6899, 6977, 7193, 7487, 8081]
set_option maxRecDepth 4096 in
theorem rootCandidates331_checked : candidates edge 5 0 2243 tail332 best.bound = rootCandidates331 := by
  unfold candidates
  have row : edge 2243 = PrimePairCache.smallRow2243.contains := by funext q; rfl
  rw [row]
  decide

def innerTail331_14 : List Nat := []
theorem innerScan331_14 : scan edge (search edge 3) 4 [2243] 2243 innerTail331_14 best = best := rfl

def innerTail331_13 : List Nat := 8081 :: innerTail331_14
def innerCandidates331_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates331_13_checked : candidates edge 4 2243 8081 innerTail331_14 best.bound = innerCandidates331_13 := by
  unfold candidates
  have row : edge 8081 = SearchData.row8081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan331_13 : scan edge (search edge 3) 4 [2243] 2243 innerTail331_13 best = best := by
  rw [innerTail331_13, scan, if_neg (by decide), innerCandidates331_13_checked]
  change scan edge (search edge 3) 4 [2243] 2243 innerTail331_14
    (if 3 ≤ innerCandidates331_13.length then search edge 3 [2243, 8081] innerCandidates331_13 10324 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan331_14

def innerTail331_12 : List Nat := 7487 :: innerTail331_13
def innerCandidates331_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates331_12_checked : candidates edge 4 2243 7487 innerTail331_13 best.bound = innerCandidates331_12 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan331_12 : scan edge (search edge 3) 4 [2243] 2243 innerTail331_12 best = best := by
  rw [innerTail331_12, scan, if_neg (by decide), innerCandidates331_12_checked]
  change scan edge (search edge 3) 4 [2243] 2243 innerTail331_13
    (if 3 ≤ innerCandidates331_12.length then search edge 3 [2243, 7487] innerCandidates331_12 9730 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan331_13

def innerTail331_11 : List Nat := 7193 :: innerTail331_12
def innerCandidates331_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates331_11_checked : candidates edge 4 2243 7193 innerTail331_12 best.bound = innerCandidates331_11 := by
  unfold candidates
  have row : edge 7193 = SearchData.row7193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan331_11 : scan edge (search edge 3) 4 [2243] 2243 innerTail331_11 best = best := by
  rw [innerTail331_11, scan, if_neg (by decide), innerCandidates331_11_checked]
  change scan edge (search edge 3) 4 [2243] 2243 innerTail331_12
    (if 3 ≤ innerCandidates331_11.length then search edge 3 [2243, 7193] innerCandidates331_11 9436 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan331_12

def innerTail331_10 : List Nat := 6977 :: innerTail331_11
def innerCandidates331_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates331_10_checked : candidates edge 4 2243 6977 innerTail331_11 best.bound = innerCandidates331_10 := by
  unfold candidates
  have row : edge 6977 = SearchData.row6977.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan331_10 : scan edge (search edge 3) 4 [2243] 2243 innerTail331_10 best = best := by
  rw [innerTail331_10, scan, if_neg (by decide), innerCandidates331_10_checked]
  change scan edge (search edge 3) 4 [2243] 2243 innerTail331_11
    (if 3 ≤ innerCandidates331_10.length then search edge 3 [2243, 6977] innerCandidates331_10 9220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan331_11

def innerTail331_9 : List Nat := 6899 :: innerTail331_10
def innerCandidates331_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates331_9_checked : candidates edge 4 2243 6899 innerTail331_10 best.bound = innerCandidates331_9 := by
  unfold candidates
  have row : edge 6899 = SearchData.row6899.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan331_9 : scan edge (search edge 3) 4 [2243] 2243 innerTail331_9 best = best := by
  rw [innerTail331_9, scan, if_neg (by decide), innerCandidates331_9_checked]
  change scan edge (search edge 3) 4 [2243] 2243 innerTail331_10
    (if 3 ≤ innerCandidates331_9.length then search edge 3 [2243, 6899] innerCandidates331_9 9142 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan331_10

def innerTail331_8 : List Nat := 6599 :: innerTail331_9
def innerCandidates331_8 : List Nat := [8081]
set_option maxRecDepth 4096 in
theorem innerCandidates331_8_checked : candidates edge 4 2243 6599 innerTail331_9 best.bound = innerCandidates331_8 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan331_8 : scan edge (search edge 3) 4 [2243] 2243 innerTail331_8 best = best := by
  rw [innerTail331_8, scan, if_neg (by decide), innerCandidates331_8_checked]
  change scan edge (search edge 3) 4 [2243] 2243 innerTail331_9
    (if 3 ≤ innerCandidates331_8.length then search edge 3 [2243, 6599] innerCandidates331_8 8842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan331_9

def innerTail331_7 : List Nat := 5927 :: innerTail331_8
def innerCandidates331_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates331_7_checked : candidates edge 4 2243 5927 innerTail331_8 best.bound = innerCandidates331_7 := by
  unfold candidates
  have row : edge 5927 = SearchData.row5927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan331_7 : scan edge (search edge 3) 4 [2243] 2243 innerTail331_7 best = best := by
  rw [innerTail331_7, scan, if_neg (by decide), innerCandidates331_7_checked]
  change scan edge (search edge 3) 4 [2243] 2243 innerTail331_8
    (if 3 ≤ innerCandidates331_7.length then search edge 3 [2243, 5927] innerCandidates331_7 8170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan331_8

def innerTail331_6 : List Nat := 5477 :: innerTail331_7
def innerCandidates331_6 : List Nat := [5927, 6599, 8081]
set_option maxRecDepth 4096 in
theorem innerCandidates331_6_checked : candidates edge 4 2243 5477 innerTail331_7 best.bound = innerCandidates331_6 := by
  unfold candidates
  have row : edge 5477 = SearchData.row5477.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild331_6 : search edge 3 [2243, 5477] innerCandidates331_6 7720 best = best := by decide

theorem innerScan331_6 : scan edge (search edge 3) 4 [2243] 2243 innerTail331_6 best = best := by
  rw [innerTail331_6, scan, if_neg (by decide), innerCandidates331_6_checked]
  change scan edge (search edge 3) 4 [2243] 2243 innerTail331_7
    (if 3 ≤ innerCandidates331_6.length then search edge 3 [2243, 5477] innerCandidates331_6 7720 best else best) = best
  rw [if_pos (by decide), innerChild331_6]
  exact innerScan331_7

def innerTail331_5 : List Nat := 4457 :: innerTail331_6
def innerCandidates331_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates331_5_checked : candidates edge 4 2243 4457 innerTail331_6 best.bound = innerCandidates331_5 := by
  unfold candidates
  have row : edge 4457 = SearchData.row4457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan331_5 : scan edge (search edge 3) 4 [2243] 2243 innerTail331_5 best = best := by
  rw [innerTail331_5, scan, if_neg (by decide), innerCandidates331_5_checked]
  change scan edge (search edge 3) 4 [2243] 2243 innerTail331_6
    (if 3 ≤ innerCandidates331_5.length then search edge 3 [2243, 4457] innerCandidates331_5 6700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan331_6

def innerTail331_4 : List Nat := 4133 :: innerTail331_5
def innerCandidates331_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates331_4_checked : candidates edge 4 2243 4133 innerTail331_5 best.bound = innerCandidates331_4 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan331_4 : scan edge (search edge 3) 4 [2243] 2243 innerTail331_4 best = best := by
  rw [innerTail331_4, scan, if_neg (by decide), innerCandidates331_4_checked]
  change scan edge (search edge 3) 4 [2243] 2243 innerTail331_5
    (if 3 ≤ innerCandidates331_4.length then search edge 3 [2243, 4133] innerCandidates331_4 6376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan331_5

def innerTail331_3 : List Nat := 3539 :: innerTail331_4
def innerCandidates331_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates331_3_checked : candidates edge 4 2243 3539 innerTail331_4 best.bound = innerCandidates331_3 := by
  unfold candidates
  have row : edge 3539 = SearchData.row3539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan331_3 : scan edge (search edge 3) 4 [2243] 2243 innerTail331_3 best = best := by
  rw [innerTail331_3, scan, if_neg (by decide), innerCandidates331_3_checked]
  change scan edge (search edge 3) 4 [2243] 2243 innerTail331_4
    (if 3 ≤ innerCandidates331_3.length then search edge 3 [2243, 3539] innerCandidates331_3 5782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan331_4

def innerTail331_2 : List Nat := 2789 :: innerTail331_3
def innerCandidates331_2 : List Nat := [4133]
set_option maxRecDepth 4096 in
theorem innerCandidates331_2_checked : candidates edge 4 2243 2789 innerTail331_3 best.bound = innerCandidates331_2 := by
  unfold candidates
  have row : edge 2789 = SearchData.row2789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan331_2 : scan edge (search edge 3) 4 [2243] 2243 innerTail331_2 best = best := by
  rw [innerTail331_2, scan, if_neg (by decide), innerCandidates331_2_checked]
  change scan edge (search edge 3) 4 [2243] 2243 innerTail331_3
    (if 3 ≤ innerCandidates331_2.length then search edge 3 [2243, 2789] innerCandidates331_2 5032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan331_3

def innerTail331_1 : List Nat := 2411 :: innerTail331_2
def innerCandidates331_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates331_1_checked : candidates edge 4 2243 2411 innerTail331_2 best.bound = innerCandidates331_1 := by
  unfold candidates
  have row : edge 2411 = PrimePairCache.smallRow2411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan331_1 : scan edge (search edge 3) 4 [2243] 2243 innerTail331_1 best = best := by
  rw [innerTail331_1, scan, if_neg (by decide), innerCandidates331_1_checked]
  change scan edge (search edge 3) 4 [2243] 2243 innerTail331_2
    (if 3 ≤ innerCandidates331_1.length then search edge 3 [2243, 2411] innerCandidates331_1 4654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan331_2

def innerTail331_0 : List Nat := 2273 :: innerTail331_1
def innerCandidates331_0 : List Nat := [2789]
set_option maxRecDepth 4096 in
theorem innerCandidates331_0_checked : candidates edge 4 2243 2273 innerTail331_1 best.bound = innerCandidates331_0 := by
  unfold candidates
  have row : edge 2273 = PrimePairCache.smallRow2273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan331_0 : scan edge (search edge 3) 4 [2243] 2243 innerTail331_0 best = best := by
  rw [innerTail331_0, scan, if_neg (by decide), innerCandidates331_0_checked]
  change scan edge (search edge 3) 4 [2243] 2243 innerTail331_1
    (if 3 ≤ innerCandidates331_0.length then search edge 3 [2243, 2273] innerCandidates331_0 4516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan331_1

theorem rootBranch331 : search edge 4 [2243] rootCandidates331 2243 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2243] 2243 innerTail331_0 best = best
  exact innerScan331_0

def rootCandidates330 : List Nat := [2857, 3853, 3919, 4129, 4261, 4441, 4987, 5179, 5209, 5557, 5683, 6007, 6481, 6823, 7027, 7237, 7351, 7393, 7411, 7951, 8221, 8329]
set_option maxRecDepth 4096 in
theorem rootCandidates330_checked : candidates edge 5 0 2239 tail331 best.bound = rootCandidates330 := by
  unfold candidates
  have row : edge 2239 = PrimePairCache.smallRow2239.contains := by funext q; rfl
  rw [row]
  decide

def innerTail330_22 : List Nat := []
theorem innerScan330_22 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_22 best = best := rfl

def innerTail330_21 : List Nat := 8329 :: innerTail330_22
def innerCandidates330_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates330_21_checked : candidates edge 4 2239 8329 innerTail330_22 best.bound = innerCandidates330_21 := by
  unfold candidates
  have row : edge 8329 = SearchData.row8329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan330_21 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_21 best = best := by
  rw [innerTail330_21, scan, if_neg (by decide), innerCandidates330_21_checked]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_22
    (if 3 ≤ innerCandidates330_21.length then search edge 3 [2239, 8329] innerCandidates330_21 10568 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan330_22

def innerTail330_20 : List Nat := 8221 :: innerTail330_21
def innerCandidates330_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates330_20_checked : candidates edge 4 2239 8221 innerTail330_21 best.bound = innerCandidates330_20 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan330_20 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_20 best = best := by
  rw [innerTail330_20, scan, if_neg (by decide), innerCandidates330_20_checked]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_21
    (if 3 ≤ innerCandidates330_20.length then search edge 3 [2239, 8221] innerCandidates330_20 10460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan330_21

def innerTail330_19 : List Nat := 7951 :: innerTail330_20
def innerCandidates330_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates330_19_checked : candidates edge 4 2239 7951 innerTail330_20 best.bound = innerCandidates330_19 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan330_19 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_19 best = best := by
  rw [innerTail330_19, scan, if_neg (by decide), innerCandidates330_19_checked]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_20
    (if 3 ≤ innerCandidates330_19.length then search edge 3 [2239, 7951] innerCandidates330_19 10190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan330_20

def innerTail330_18 : List Nat := 7411 :: innerTail330_19
def innerCandidates330_18 : List Nat := [8329]
set_option maxRecDepth 4096 in
theorem innerCandidates330_18_checked : candidates edge 4 2239 7411 innerTail330_19 best.bound = innerCandidates330_18 := by
  unfold candidates
  have row : edge 7411 = SearchData.row7411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan330_18 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_18 best = best := by
  rw [innerTail330_18, scan, if_neg (by decide), innerCandidates330_18_checked]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_19
    (if 3 ≤ innerCandidates330_18.length then search edge 3 [2239, 7411] innerCandidates330_18 9650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan330_19

def innerTail330_17 : List Nat := 7393 :: innerTail330_18
def innerCandidates330_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates330_17_checked : candidates edge 4 2239 7393 innerTail330_18 best.bound = innerCandidates330_17 := by
  unfold candidates
  have row : edge 7393 = SearchData.row7393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan330_17 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_17 best = best := by
  rw [innerTail330_17, scan, if_neg (by decide), innerCandidates330_17_checked]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_18
    (if 3 ≤ innerCandidates330_17.length then search edge 3 [2239, 7393] innerCandidates330_17 9632 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan330_18

def innerTail330_16 : List Nat := 7351 :: innerTail330_17
def innerCandidates330_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates330_16_checked : candidates edge 4 2239 7351 innerTail330_17 best.bound = innerCandidates330_16 := by
  unfold candidates
  have row : edge 7351 = SearchData.row7351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan330_16 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_16 best = best := by
  rw [innerTail330_16, scan, if_neg (by decide), innerCandidates330_16_checked]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_17
    (if 3 ≤ innerCandidates330_16.length then search edge 3 [2239, 7351] innerCandidates330_16 9590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan330_17

def innerTail330_15 : List Nat := 7237 :: innerTail330_16
def innerCandidates330_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates330_15_checked : candidates edge 4 2239 7237 innerTail330_16 best.bound = innerCandidates330_15 := by
  unfold candidates
  have row : edge 7237 = SearchData.row7237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan330_15 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_15 best = best := by
  rw [innerTail330_15, scan, if_neg (by decide), innerCandidates330_15_checked]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_16
    (if 3 ≤ innerCandidates330_15.length then search edge 3 [2239, 7237] innerCandidates330_15 9476 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan330_16

def innerTail330_14 : List Nat := 7027 :: innerTail330_15
def innerCandidates330_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates330_14_checked : candidates edge 4 2239 7027 innerTail330_15 best.bound = innerCandidates330_14 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan330_14 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_14 best = best := by
  rw [innerTail330_14, scan, if_neg (by decide), innerCandidates330_14_checked]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_15
    (if 3 ≤ innerCandidates330_14.length then search edge 3 [2239, 7027] innerCandidates330_14 9266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan330_15

def innerTail330_13 : List Nat := 6823 :: innerTail330_14
def innerCandidates330_13 : List Nat := [7951]
set_option maxRecDepth 4096 in
theorem innerCandidates330_13_checked : candidates edge 4 2239 6823 innerTail330_14 best.bound = innerCandidates330_13 := by
  unfold candidates
  have row : edge 6823 = SearchData.row6823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan330_13 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_13 best = best := by
  rw [innerTail330_13, scan, if_neg (by decide), innerCandidates330_13_checked]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_14
    (if 3 ≤ innerCandidates330_13.length then search edge 3 [2239, 6823] innerCandidates330_13 9062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan330_14

def innerTail330_12 : List Nat := 6481 :: innerTail330_13
def innerCandidates330_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates330_12_checked : candidates edge 4 2239 6481 innerTail330_13 best.bound = innerCandidates330_12 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan330_12 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_12 best = best := by
  rw [innerTail330_12, scan, if_neg (by decide), innerCandidates330_12_checked]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_13
    (if 3 ≤ innerCandidates330_12.length then search edge 3 [2239, 6481] innerCandidates330_12 8720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan330_13

def innerTail330_11 : List Nat := 6007 :: innerTail330_12
def innerCandidates330_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates330_11_checked : candidates edge 4 2239 6007 innerTail330_12 best.bound = innerCandidates330_11 := by
  unfold candidates
  have row : edge 6007 = SearchData.row6007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan330_11 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_11 best = best := by
  rw [innerTail330_11, scan, if_neg (by decide), innerCandidates330_11_checked]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_12
    (if 3 ≤ innerCandidates330_11.length then search edge 3 [2239, 6007] innerCandidates330_11 8246 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan330_12

def innerTail330_10 : List Nat := 5683 :: innerTail330_11
def innerCandidates330_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates330_10_checked : candidates edge 4 2239 5683 innerTail330_11 best.bound = innerCandidates330_10 := by
  unfold candidates
  have row : edge 5683 = SearchData.row5683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan330_10 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_10 best = best := by
  rw [innerTail330_10, scan, if_neg (by decide), innerCandidates330_10_checked]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_11
    (if 3 ≤ innerCandidates330_10.length then search edge 3 [2239, 5683] innerCandidates330_10 7922 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan330_11

def innerTail330_9 : List Nat := 5557 :: innerTail330_10
def innerCandidates330_9 : List Nat := [7393]
set_option maxRecDepth 4096 in
theorem innerCandidates330_9_checked : candidates edge 4 2239 5557 innerTail330_10 best.bound = innerCandidates330_9 := by
  unfold candidates
  have row : edge 5557 = SearchData.row5557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan330_9 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_9 best = best := by
  rw [innerTail330_9, scan, if_neg (by decide), innerCandidates330_9_checked]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_10
    (if 3 ≤ innerCandidates330_9.length then search edge 3 [2239, 5557] innerCandidates330_9 7796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan330_10

def innerTail330_8 : List Nat := 5209 :: innerTail330_9
def innerCandidates330_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates330_8_checked : candidates edge 4 2239 5209 innerTail330_9 best.bound = innerCandidates330_8 := by
  unfold candidates
  have row : edge 5209 = SearchData.row5209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan330_8 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_8 best = best := by
  rw [innerTail330_8, scan, if_neg (by decide), innerCandidates330_8_checked]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_9
    (if 3 ≤ innerCandidates330_8.length then search edge 3 [2239, 5209] innerCandidates330_8 7448 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan330_9

def innerTail330_7 : List Nat := 5179 :: innerTail330_8
def innerCandidates330_7 : List Nat := [5683]
set_option maxRecDepth 4096 in
theorem innerCandidates330_7_checked : candidates edge 4 2239 5179 innerTail330_8 best.bound = innerCandidates330_7 := by
  unfold candidates
  have row : edge 5179 = SearchData.row5179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan330_7 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_7 best = best := by
  rw [innerTail330_7, scan, if_neg (by decide), innerCandidates330_7_checked]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_8
    (if 3 ≤ innerCandidates330_7.length then search edge 3 [2239, 5179] innerCandidates330_7 7418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan330_8

def innerTail330_6 : List Nat := 4987 :: innerTail330_7
def innerCandidates330_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates330_6_checked : candidates edge 4 2239 4987 innerTail330_7 best.bound = innerCandidates330_6 := by
  unfold candidates
  have row : edge 4987 = SearchData.row4987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan330_6 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_6 best = best := by
  rw [innerTail330_6, scan, if_neg (by decide), innerCandidates330_6_checked]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_7
    (if 3 ≤ innerCandidates330_6.length then search edge 3 [2239, 4987] innerCandidates330_6 7226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan330_7

def innerTail330_5 : List Nat := 4441 :: innerTail330_6
def innerCandidates330_5 : List Nat := [6481]
set_option maxRecDepth 4096 in
theorem innerCandidates330_5_checked : candidates edge 4 2239 4441 innerTail330_6 best.bound = innerCandidates330_5 := by
  unfold candidates
  have row : edge 4441 = SearchData.row4441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan330_5 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_5 best = best := by
  rw [innerTail330_5, scan, if_neg (by decide), innerCandidates330_5_checked]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_6
    (if 3 ≤ innerCandidates330_5.length then search edge 3 [2239, 4441] innerCandidates330_5 6680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan330_6

def innerTail330_4 : List Nat := 4261 :: innerTail330_5
def innerCandidates330_4 : List Nat := [7951]
set_option maxRecDepth 4096 in
theorem innerCandidates330_4_checked : candidates edge 4 2239 4261 innerTail330_5 best.bound = innerCandidates330_4 := by
  unfold candidates
  have row : edge 4261 = SearchData.row4261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan330_4 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_4 best = best := by
  rw [innerTail330_4, scan, if_neg (by decide), innerCandidates330_4_checked]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_5
    (if 3 ≤ innerCandidates330_4.length then search edge 3 [2239, 4261] innerCandidates330_4 6500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan330_5

def innerTail330_3 : List Nat := 4129 :: innerTail330_4
def innerCandidates330_3 : List Nat := [8221]
set_option maxRecDepth 4096 in
theorem innerCandidates330_3_checked : candidates edge 4 2239 4129 innerTail330_4 best.bound = innerCandidates330_3 := by
  unfold candidates
  have row : edge 4129 = SearchData.row4129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan330_3 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_3 best = best := by
  rw [innerTail330_3, scan, if_neg (by decide), innerCandidates330_3_checked]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_4
    (if 3 ≤ innerCandidates330_3.length then search edge 3 [2239, 4129] innerCandidates330_3 6368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan330_4

def innerTail330_2 : List Nat := 3919 :: innerTail330_3
def innerCandidates330_2 : List Nat := [5179, 6007]
set_option maxRecDepth 4096 in
theorem innerCandidates330_2_checked : candidates edge 4 2239 3919 innerTail330_3 best.bound = innerCandidates330_2 := by
  unfold candidates
  have row : edge 3919 = SearchData.row3919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan330_2 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_2 best = best := by
  rw [innerTail330_2, scan, if_neg (by decide), innerCandidates330_2_checked]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_3
    (if 3 ≤ innerCandidates330_2.length then search edge 3 [2239, 3919] innerCandidates330_2 6158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan330_3

def innerTail330_1 : List Nat := 3853 :: innerTail330_2
def innerCandidates330_1 : List Nat := [5683, 7237]
set_option maxRecDepth 4096 in
theorem innerCandidates330_1_checked : candidates edge 4 2239 3853 innerTail330_2 best.bound = innerCandidates330_1 := by
  unfold candidates
  have row : edge 3853 = SearchData.row3853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan330_1 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_1 best = best := by
  rw [innerTail330_1, scan, if_neg (by decide), innerCandidates330_1_checked]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_2
    (if 3 ≤ innerCandidates330_1.length then search edge 3 [2239, 3853] innerCandidates330_1 6092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan330_2

def innerTail330_0 : List Nat := 2857 :: innerTail330_1
def innerCandidates330_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates330_0_checked : candidates edge 4 2239 2857 innerTail330_1 best.bound = innerCandidates330_0 := by
  unfold candidates
  have row : edge 2857 = SearchData.row2857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan330_0 : scan edge (search edge 3) 4 [2239] 2239 innerTail330_0 best = best := by
  rw [innerTail330_0, scan, if_neg (by decide), innerCandidates330_0_checked]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_1
    (if 3 ≤ innerCandidates330_0.length then search edge 3 [2239, 2857] innerCandidates330_0 5096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan330_1

theorem rootBranch330 : search edge 4 [2239] rootCandidates330 2239 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2239] 2239 innerTail330_0 best = best
  exact innerScan330_0

def rootCandidates329 : List Nat := [2381, 2411, 2549, 3251, 3257, 3929, 4019, 4049, 4139, 4349, 4421, 4481, 4637, 5693, 6581, 6659, 6689, 6701, 6779, 6791, 6917]
set_option maxRecDepth 4096 in
theorem rootCandidates329_checked : candidates edge 5 0 2237 tail330 best.bound = rootCandidates329 := by
  unfold candidates
  have row : edge 2237 = PrimePairCache.smallRow2237.contains := by funext q; rfl
  rw [row]
  decide

def innerTail329_21 : List Nat := []
theorem innerScan329_21 : scan edge (search edge 3) 4 [2237] 2237 innerTail329_21 best = best := rfl

def innerTail329_20 : List Nat := 6917 :: innerTail329_21
def innerCandidates329_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates329_20_checked : candidates edge 4 2237 6917 innerTail329_21 best.bound = innerCandidates329_20 := by
  unfold candidates
  have row : edge 6917 = SearchData.row6917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan329_20 : scan edge (search edge 3) 4 [2237] 2237 innerTail329_20 best = best := by
  rw [innerTail329_20, scan, if_neg (by decide), innerCandidates329_20_checked]
  change scan edge (search edge 3) 4 [2237] 2237 innerTail329_21
    (if 3 ≤ innerCandidates329_20.length then search edge 3 [2237, 6917] innerCandidates329_20 9154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan329_21

def innerTail329_19 : List Nat := 6791 :: innerTail329_20
def innerCandidates329_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates329_19_checked : candidates edge 4 2237 6791 innerTail329_20 best.bound = innerCandidates329_19 := by
  unfold candidates
  have row : edge 6791 = SearchData.row6791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan329_19 : scan edge (search edge 3) 4 [2237] 2237 innerTail329_19 best = best := by
  rw [innerTail329_19, scan, if_neg (by decide), innerCandidates329_19_checked]
  change scan edge (search edge 3) 4 [2237] 2237 innerTail329_20
    (if 3 ≤ innerCandidates329_19.length then search edge 3 [2237, 6791] innerCandidates329_19 9028 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan329_20

def innerTail329_18 : List Nat := 6779 :: innerTail329_19
def innerCandidates329_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates329_18_checked : candidates edge 4 2237 6779 innerTail329_19 best.bound = innerCandidates329_18 := by
  unfold candidates
  have row : edge 6779 = SearchData.row6779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan329_18 : scan edge (search edge 3) 4 [2237] 2237 innerTail329_18 best = best := by
  rw [innerTail329_18, scan, if_neg (by decide), innerCandidates329_18_checked]
  change scan edge (search edge 3) 4 [2237] 2237 innerTail329_19
    (if 3 ≤ innerCandidates329_18.length then search edge 3 [2237, 6779] innerCandidates329_18 9016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan329_19

def innerTail329_17 : List Nat := 6701 :: innerTail329_18
def innerCandidates329_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates329_17_checked : candidates edge 4 2237 6701 innerTail329_18 best.bound = innerCandidates329_17 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan329_17 : scan edge (search edge 3) 4 [2237] 2237 innerTail329_17 best = best := by
  rw [innerTail329_17, scan, if_neg (by decide), innerCandidates329_17_checked]
  change scan edge (search edge 3) 4 [2237] 2237 innerTail329_18
    (if 3 ≤ innerCandidates329_17.length then search edge 3 [2237, 6701] innerCandidates329_17 8938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan329_18

def innerTail329_16 : List Nat := 6689 :: innerTail329_17
def innerCandidates329_16 : List Nat := [6779]
set_option maxRecDepth 4096 in
theorem innerCandidates329_16_checked : candidates edge 4 2237 6689 innerTail329_17 best.bound = innerCandidates329_16 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan329_16 : scan edge (search edge 3) 4 [2237] 2237 innerTail329_16 best = best := by
  rw [innerTail329_16, scan, if_neg (by decide), innerCandidates329_16_checked]
  change scan edge (search edge 3) 4 [2237] 2237 innerTail329_17
    (if 3 ≤ innerCandidates329_16.length then search edge 3 [2237, 6689] innerCandidates329_16 8926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan329_17

def innerTail329_15 : List Nat := 6659 :: innerTail329_16
def innerCandidates329_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates329_15_checked : candidates edge 4 2237 6659 innerTail329_16 best.bound = innerCandidates329_15 := by
  unfold candidates
  have row : edge 6659 = SearchData.row6659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan329_15 : scan edge (search edge 3) 4 [2237] 2237 innerTail329_15 best = best := by
  rw [innerTail329_15, scan, if_neg (by decide), innerCandidates329_15_checked]
  change scan edge (search edge 3) 4 [2237] 2237 innerTail329_16
    (if 3 ≤ innerCandidates329_15.length then search edge 3 [2237, 6659] innerCandidates329_15 8896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan329_16

def innerTail329_14 : List Nat := 6581 :: innerTail329_15
def innerCandidates329_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates329_14_checked : candidates edge 4 2237 6581 innerTail329_15 best.bound = innerCandidates329_14 := by
  unfold candidates
  have row : edge 6581 = SearchData.row6581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan329_14 : scan edge (search edge 3) 4 [2237] 2237 innerTail329_14 best = best := by
  rw [innerTail329_14, scan, if_neg (by decide), innerCandidates329_14_checked]
  change scan edge (search edge 3) 4 [2237] 2237 innerTail329_15
    (if 3 ≤ innerCandidates329_14.length then search edge 3 [2237, 6581] innerCandidates329_14 8818 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan329_15

def innerTail329_13 : List Nat := 5693 :: innerTail329_14
def innerCandidates329_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates329_13_checked : candidates edge 4 2237 5693 innerTail329_14 best.bound = innerCandidates329_13 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan329_13 : scan edge (search edge 3) 4 [2237] 2237 innerTail329_13 best = best := by
  rw [innerTail329_13, scan, if_neg (by decide), innerCandidates329_13_checked]
  change scan edge (search edge 3) 4 [2237] 2237 innerTail329_14
    (if 3 ≤ innerCandidates329_13.length then search edge 3 [2237, 5693] innerCandidates329_13 7930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan329_14

def innerTail329_12 : List Nat := 4637 :: innerTail329_13
def innerCandidates329_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates329_12_checked : candidates edge 4 2237 4637 innerTail329_13 best.bound = innerCandidates329_12 := by
  unfold candidates
  have row : edge 4637 = SearchData.row4637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan329_12 : scan edge (search edge 3) 4 [2237] 2237 innerTail329_12 best = best := by
  rw [innerTail329_12, scan, if_neg (by decide), innerCandidates329_12_checked]
  change scan edge (search edge 3) 4 [2237] 2237 innerTail329_13
    (if 3 ≤ innerCandidates329_12.length then search edge 3 [2237, 4637] innerCandidates329_12 6874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan329_13

def innerTail329_11 : List Nat := 4481 :: innerTail329_12
def innerCandidates329_11 : List Nat := [6791]
set_option maxRecDepth 4096 in
theorem innerCandidates329_11_checked : candidates edge 4 2237 4481 innerTail329_12 best.bound = innerCandidates329_11 := by
  unfold candidates
  have row : edge 4481 = SearchData.row4481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan329_11 : scan edge (search edge 3) 4 [2237] 2237 innerTail329_11 best = best := by
  rw [innerTail329_11, scan, if_neg (by decide), innerCandidates329_11_checked]
  change scan edge (search edge 3) 4 [2237] 2237 innerTail329_12
    (if 3 ≤ innerCandidates329_11.length then search edge 3 [2237, 4481] innerCandidates329_11 6718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan329_12

def innerTail329_10 : List Nat := 4421 :: innerTail329_11
def innerCandidates329_10 : List Nat := [6791]
set_option maxRecDepth 4096 in
theorem innerCandidates329_10_checked : candidates edge 4 2237 4421 innerTail329_11 best.bound = innerCandidates329_10 := by
  unfold candidates
  have row : edge 4421 = SearchData.row4421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan329_10 : scan edge (search edge 3) 4 [2237] 2237 innerTail329_10 best = best := by
  rw [innerTail329_10, scan, if_neg (by decide), innerCandidates329_10_checked]
  change scan edge (search edge 3) 4 [2237] 2237 innerTail329_11
    (if 3 ≤ innerCandidates329_10.length then search edge 3 [2237, 4421] innerCandidates329_10 6658 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan329_11

def innerTail329_9 : List Nat := 4349 :: innerTail329_10
def innerCandidates329_9 : List Nat := [6581]
set_option maxRecDepth 4096 in
theorem innerCandidates329_9_checked : candidates edge 4 2237 4349 innerTail329_10 best.bound = innerCandidates329_9 := by
  unfold candidates
  have row : edge 4349 = SearchData.row4349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan329_9 : scan edge (search edge 3) 4 [2237] 2237 innerTail329_9 best = best := by
  rw [innerTail329_9, scan, if_neg (by decide), innerCandidates329_9_checked]
  change scan edge (search edge 3) 4 [2237] 2237 innerTail329_10
    (if 3 ≤ innerCandidates329_9.length then search edge 3 [2237, 4349] innerCandidates329_9 6586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan329_10

def innerTail329_8 : List Nat := 4139 :: innerTail329_9
def innerCandidates329_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates329_8_checked : candidates edge 4 2237 4139 innerTail329_9 best.bound = innerCandidates329_8 := by
  unfold candidates
  have row : edge 4139 = SearchData.row4139.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan329_8 : scan edge (search edge 3) 4 [2237] 2237 innerTail329_8 best = best := by
  rw [innerTail329_8, scan, if_neg (by decide), innerCandidates329_8_checked]
  change scan edge (search edge 3) 4 [2237] 2237 innerTail329_9
    (if 3 ≤ innerCandidates329_8.length then search edge 3 [2237, 4139] innerCandidates329_8 6376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan329_9

def innerTail329_7 : List Nat := 4049 :: innerTail329_8
def innerCandidates329_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates329_7_checked : candidates edge 4 2237 4049 innerTail329_8 best.bound = innerCandidates329_7 := by
  unfold candidates
  have row : edge 4049 = SearchData.row4049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan329_7 : scan edge (search edge 3) 4 [2237] 2237 innerTail329_7 best = best := by
  rw [innerTail329_7, scan, if_neg (by decide), innerCandidates329_7_checked]
  change scan edge (search edge 3) 4 [2237] 2237 innerTail329_8
    (if 3 ≤ innerCandidates329_7.length then search edge 3 [2237, 4049] innerCandidates329_7 6286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan329_8

def innerTail329_6 : List Nat := 4019 :: innerTail329_7
def innerCandidates329_6 : List Nat := [6791]
set_option maxRecDepth 4096 in
theorem innerCandidates329_6_checked : candidates edge 4 2237 4019 innerTail329_7 best.bound = innerCandidates329_6 := by
  unfold candidates
  have row : edge 4019 = SearchData.row4019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan329_6 : scan edge (search edge 3) 4 [2237] 2237 innerTail329_6 best = best := by
  rw [innerTail329_6, scan, if_neg (by decide), innerCandidates329_6_checked]
  change scan edge (search edge 3) 4 [2237] 2237 innerTail329_7
    (if 3 ≤ innerCandidates329_6.length then search edge 3 [2237, 4019] innerCandidates329_6 6256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan329_7

def innerTail329_5 : List Nat := 3929 :: innerTail329_6
def innerCandidates329_5 : List Nat := [4349, 4421]
set_option maxRecDepth 4096 in
theorem innerCandidates329_5_checked : candidates edge 4 2237 3929 innerTail329_6 best.bound = innerCandidates329_5 := by
  unfold candidates
  have row : edge 3929 = SearchData.row3929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan329_5 : scan edge (search edge 3) 4 [2237] 2237 innerTail329_5 best = best := by
  rw [innerTail329_5, scan, if_neg (by decide), innerCandidates329_5_checked]
  change scan edge (search edge 3) 4 [2237] 2237 innerTail329_6
    (if 3 ≤ innerCandidates329_5.length then search edge 3 [2237, 3929] innerCandidates329_5 6166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan329_6

def innerTail329_4 : List Nat := 3257 :: innerTail329_5
def innerCandidates329_4 : List Nat := [6917]
set_option maxRecDepth 4096 in
theorem innerCandidates329_4_checked : candidates edge 4 2237 3257 innerTail329_5 best.bound = innerCandidates329_4 := by
  unfold candidates
  have row : edge 3257 = SearchData.row3257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan329_4 : scan edge (search edge 3) 4 [2237] 2237 innerTail329_4 best = best := by
  rw [innerTail329_4, scan, if_neg (by decide), innerCandidates329_4_checked]
  change scan edge (search edge 3) 4 [2237] 2237 innerTail329_5
    (if 3 ≤ innerCandidates329_4.length then search edge 3 [2237, 3257] innerCandidates329_4 5494 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan329_5

def innerTail329_3 : List Nat := 3251 :: innerTail329_4
def innerCandidates329_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates329_3_checked : candidates edge 4 2237 3251 innerTail329_4 best.bound = innerCandidates329_3 := by
  unfold candidates
  have row : edge 3251 = SearchData.row3251.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan329_3 : scan edge (search edge 3) 4 [2237] 2237 innerTail329_3 best = best := by
  rw [innerTail329_3, scan, if_neg (by decide), innerCandidates329_3_checked]
  change scan edge (search edge 3) 4 [2237] 2237 innerTail329_4
    (if 3 ≤ innerCandidates329_3.length then search edge 3 [2237, 3251] innerCandidates329_3 5488 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan329_4

def innerTail329_2 : List Nat := 2549 :: innerTail329_3
def innerCandidates329_2 : List Nat := [4019, 4349]
set_option maxRecDepth 4096 in
theorem innerCandidates329_2_checked : candidates edge 4 2237 2549 innerTail329_3 best.bound = innerCandidates329_2 := by
  unfold candidates
  have row : edge 2549 = PrimePairCache.smallRow2549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan329_2 : scan edge (search edge 3) 4 [2237] 2237 innerTail329_2 best = best := by
  rw [innerTail329_2, scan, if_neg (by decide), innerCandidates329_2_checked]
  change scan edge (search edge 3) 4 [2237] 2237 innerTail329_3
    (if 3 ≤ innerCandidates329_2.length then search edge 3 [2237, 2549] innerCandidates329_2 4786 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan329_3

def innerTail329_1 : List Nat := 2411 :: innerTail329_2
def innerCandidates329_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates329_1_checked : candidates edge 4 2237 2411 innerTail329_2 best.bound = innerCandidates329_1 := by
  unfold candidates
  have row : edge 2411 = PrimePairCache.smallRow2411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan329_1 : scan edge (search edge 3) 4 [2237] 2237 innerTail329_1 best = best := by
  rw [innerTail329_1, scan, if_neg (by decide), innerCandidates329_1_checked]
  change scan edge (search edge 3) 4 [2237] 2237 innerTail329_2
    (if 3 ≤ innerCandidates329_1.length then search edge 3 [2237, 2411] innerCandidates329_1 4648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan329_2

def innerTail329_0 : List Nat := 2381 :: innerTail329_1
def innerCandidates329_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates329_0_checked : candidates edge 4 2237 2381 innerTail329_1 best.bound = innerCandidates329_0 := by
  unfold candidates
  have row : edge 2381 = PrimePairCache.smallRow2381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan329_0 : scan edge (search edge 3) 4 [2237] 2237 innerTail329_0 best = best := by
  rw [innerTail329_0, scan, if_neg (by decide), innerCandidates329_0_checked]
  change scan edge (search edge 3) 4 [2237] 2237 innerTail329_1
    (if 3 ≤ innerCandidates329_0.length then search edge 3 [2237, 2381] innerCandidates329_0 4618 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan329_1

theorem rootBranch329 : search edge 4 [2237] rootCandidates329 2237 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2237] 2237 innerTail329_0 best = best
  exact innerScan329_0

end ProjectEulerSolutions.PrimePairUpper.Search
