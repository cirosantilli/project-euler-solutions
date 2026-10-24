import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates76 : List Nat := [593, 743, 809, 887, 1289, 1487, 2213, 2441, 2999, 3407, 3623, 3671, 4133, 4211, 4217, 4271, 5051, 5801, 6317, 6833, 7589, 7691]
set_option maxRecDepth 4096 in
theorem rootCandidates76_checked : candidates edge 5 0 401 tail77 best.bound = rootCandidates76 := by
  unfold candidates
  have row : edge 401 = PrimePairCache.smallRow401.contains := by funext q; rfl
  rw [row]
  decide

def innerTail76_22 : List Nat := []
theorem innerScan76_22 : scan edge (search edge 3) 4 [401] 401 innerTail76_22 best = best := rfl

def innerTail76_21 : List Nat := 7691 :: innerTail76_22
def innerCandidates76_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates76_21_checked : candidates edge 4 401 7691 innerTail76_22 best.bound = innerCandidates76_21 := by
  unfold candidates
  have row : edge 7691 = SearchData.row7691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan76_21 : scan edge (search edge 3) 4 [401] 401 innerTail76_21 best = best := by
  rw [innerTail76_21, scan, if_neg (by decide), innerCandidates76_21_checked]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_22
    (if 3 ≤ innerCandidates76_21.length then search edge 3 [401, 7691] innerCandidates76_21 8092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan76_22

def innerTail76_20 : List Nat := 7589 :: innerTail76_21
def innerCandidates76_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates76_20_checked : candidates edge 4 401 7589 innerTail76_21 best.bound = innerCandidates76_20 := by
  unfold candidates
  have row : edge 7589 = SearchData.row7589.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan76_20 : scan edge (search edge 3) 4 [401] 401 innerTail76_20 best = best := by
  rw [innerTail76_20, scan, if_neg (by decide), innerCandidates76_20_checked]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_21
    (if 3 ≤ innerCandidates76_20.length then search edge 3 [401, 7589] innerCandidates76_20 7990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan76_21

def innerTail76_19 : List Nat := 6833 :: innerTail76_20
def innerCandidates76_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates76_19_checked : candidates edge 4 401 6833 innerTail76_20 best.bound = innerCandidates76_19 := by
  unfold candidates
  have row : edge 6833 = SearchData.row6833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan76_19 : scan edge (search edge 3) 4 [401] 401 innerTail76_19 best = best := by
  rw [innerTail76_19, scan, if_neg (by decide), innerCandidates76_19_checked]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_20
    (if 3 ≤ innerCandidates76_19.length then search edge 3 [401, 6833] innerCandidates76_19 7234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan76_20

def innerTail76_18 : List Nat := 6317 :: innerTail76_19
def innerCandidates76_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates76_18_checked : candidates edge 4 401 6317 innerTail76_19 best.bound = innerCandidates76_18 := by
  unfold candidates
  have row : edge 6317 = SearchData.row6317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan76_18 : scan edge (search edge 3) 4 [401] 401 innerTail76_18 best = best := by
  rw [innerTail76_18, scan, if_neg (by decide), innerCandidates76_18_checked]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_19
    (if 3 ≤ innerCandidates76_18.length then search edge 3 [401, 6317] innerCandidates76_18 6718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan76_19

def innerTail76_17 : List Nat := 5801 :: innerTail76_18
def innerCandidates76_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates76_17_checked : candidates edge 4 401 5801 innerTail76_18 best.bound = innerCandidates76_17 := by
  unfold candidates
  have row : edge 5801 = SearchData.row5801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan76_17 : scan edge (search edge 3) 4 [401] 401 innerTail76_17 best = best := by
  rw [innerTail76_17, scan, if_neg (by decide), innerCandidates76_17_checked]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_18
    (if 3 ≤ innerCandidates76_17.length then search edge 3 [401, 5801] innerCandidates76_17 6202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan76_18

def innerTail76_16 : List Nat := 5051 :: innerTail76_17
def innerCandidates76_16 : List Nat := [6833]
set_option maxRecDepth 4096 in
theorem innerCandidates76_16_checked : candidates edge 4 401 5051 innerTail76_17 best.bound = innerCandidates76_16 := by
  unfold candidates
  have row : edge 5051 = SearchData.row5051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan76_16 : scan edge (search edge 3) 4 [401] 401 innerTail76_16 best = best := by
  rw [innerTail76_16, scan, if_neg (by decide), innerCandidates76_16_checked]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_17
    (if 3 ≤ innerCandidates76_16.length then search edge 3 [401, 5051] innerCandidates76_16 5452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan76_17

def innerTail76_15 : List Nat := 4271 :: innerTail76_16
def innerCandidates76_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates76_15_checked : candidates edge 4 401 4271 innerTail76_16 best.bound = innerCandidates76_15 := by
  unfold candidates
  have row : edge 4271 = SearchData.row4271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan76_15 : scan edge (search edge 3) 4 [401] 401 innerTail76_15 best = best := by
  rw [innerTail76_15, scan, if_neg (by decide), innerCandidates76_15_checked]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_16
    (if 3 ≤ innerCandidates76_15.length then search edge 3 [401, 4271] innerCandidates76_15 4672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan76_16

def innerTail76_14 : List Nat := 4217 :: innerTail76_15
def innerCandidates76_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates76_14_checked : candidates edge 4 401 4217 innerTail76_15 best.bound = innerCandidates76_14 := by
  unfold candidates
  have row : edge 4217 = SearchData.row4217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan76_14 : scan edge (search edge 3) 4 [401] 401 innerTail76_14 best = best := by
  rw [innerTail76_14, scan, if_neg (by decide), innerCandidates76_14_checked]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_15
    (if 3 ≤ innerCandidates76_14.length then search edge 3 [401, 4217] innerCandidates76_14 4618 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan76_15

def innerTail76_13 : List Nat := 4211 :: innerTail76_14
def innerCandidates76_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates76_13_checked : candidates edge 4 401 4211 innerTail76_14 best.bound = innerCandidates76_13 := by
  unfold candidates
  have row : edge 4211 = SearchData.row4211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan76_13 : scan edge (search edge 3) 4 [401] 401 innerTail76_13 best = best := by
  rw [innerTail76_13, scan, if_neg (by decide), innerCandidates76_13_checked]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_14
    (if 3 ≤ innerCandidates76_13.length then search edge 3 [401, 4211] innerCandidates76_13 4612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan76_14

def innerTail76_12 : List Nat := 4133 :: innerTail76_13
def innerCandidates76_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates76_12_checked : candidates edge 4 401 4133 innerTail76_13 best.bound = innerCandidates76_12 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan76_12 : scan edge (search edge 3) 4 [401] 401 innerTail76_12 best = best := by
  rw [innerTail76_12, scan, if_neg (by decide), innerCandidates76_12_checked]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_13
    (if 3 ≤ innerCandidates76_12.length then search edge 3 [401, 4133] innerCandidates76_12 4534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan76_13

def innerTail76_11 : List Nat := 3671 :: innerTail76_12
def innerCandidates76_11 : List Nat := [7691]
set_option maxRecDepth 4096 in
theorem innerCandidates76_11_checked : candidates edge 4 401 3671 innerTail76_12 best.bound = innerCandidates76_11 := by
  unfold candidates
  have row : edge 3671 = SearchData.row3671.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan76_11 : scan edge (search edge 3) 4 [401] 401 innerTail76_11 best = best := by
  rw [innerTail76_11, scan, if_neg (by decide), innerCandidates76_11_checked]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_12
    (if 3 ≤ innerCandidates76_11.length then search edge 3 [401, 3671] innerCandidates76_11 4072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan76_12

def innerTail76_10 : List Nat := 3623 :: innerTail76_11
def innerCandidates76_10 : List Nat := [4271]
set_option maxRecDepth 4096 in
theorem innerCandidates76_10_checked : candidates edge 4 401 3623 innerTail76_11 best.bound = innerCandidates76_10 := by
  unfold candidates
  have row : edge 3623 = SearchData.row3623.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan76_10 : scan edge (search edge 3) 4 [401] 401 innerTail76_10 best = best := by
  rw [innerTail76_10, scan, if_neg (by decide), innerCandidates76_10_checked]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_11
    (if 3 ≤ innerCandidates76_10.length then search edge 3 [401, 3623] innerCandidates76_10 4024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan76_11

def innerTail76_9 : List Nat := 3407 :: innerTail76_10
def innerCandidates76_9 : List Nat := [3671, 7691]
set_option maxRecDepth 4096 in
theorem innerCandidates76_9_checked : candidates edge 4 401 3407 innerTail76_10 best.bound = innerCandidates76_9 := by
  unfold candidates
  have row : edge 3407 = SearchData.row3407.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan76_9 : scan edge (search edge 3) 4 [401] 401 innerTail76_9 best = best := by
  rw [innerTail76_9, scan, if_neg (by decide), innerCandidates76_9_checked]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_10
    (if 3 ≤ innerCandidates76_9.length then search edge 3 [401, 3407] innerCandidates76_9 3808 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan76_10

def innerTail76_8 : List Nat := 2999 :: innerTail76_9
def innerCandidates76_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates76_8_checked : candidates edge 4 401 2999 innerTail76_9 best.bound = innerCandidates76_8 := by
  unfold candidates
  have row : edge 2999 = SearchData.row2999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan76_8 : scan edge (search edge 3) 4 [401] 401 innerTail76_8 best = best := by
  rw [innerTail76_8, scan, if_neg (by decide), innerCandidates76_8_checked]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_9
    (if 3 ≤ innerCandidates76_8.length then search edge 3 [401, 2999] innerCandidates76_8 3400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan76_9

def innerTail76_7 : List Nat := 2441 :: innerTail76_8
def innerCandidates76_7 : List Nat := [3671, 7691]
set_option maxRecDepth 4096 in
theorem innerCandidates76_7_checked : candidates edge 4 401 2441 innerTail76_8 best.bound = innerCandidates76_7 := by
  unfold candidates
  have row : edge 2441 = PrimePairCache.smallRow2441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan76_7 : scan edge (search edge 3) 4 [401] 401 innerTail76_7 best = best := by
  rw [innerTail76_7, scan, if_neg (by decide), innerCandidates76_7_checked]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_8
    (if 3 ≤ innerCandidates76_7.length then search edge 3 [401, 2441] innerCandidates76_7 2842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan76_8

def innerTail76_6 : List Nat := 2213 :: innerTail76_7
def innerCandidates76_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates76_6_checked : candidates edge 4 401 2213 innerTail76_7 best.bound = innerCandidates76_6 := by
  unfold candidates
  have row : edge 2213 = PrimePairCache.smallRow2213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan76_6 : scan edge (search edge 3) 4 [401] 401 innerTail76_6 best = best := by
  rw [innerTail76_6, scan, if_neg (by decide), innerCandidates76_6_checked]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_7
    (if 3 ≤ innerCandidates76_6.length then search edge 3 [401, 2213] innerCandidates76_6 2614 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan76_7

def innerTail76_5 : List Nat := 1487 :: innerTail76_6
def innerCandidates76_5 : List Nat := [3671, 5801]
set_option maxRecDepth 4096 in
theorem innerCandidates76_5_checked : candidates edge 4 401 1487 innerTail76_6 best.bound = innerCandidates76_5 := by
  unfold candidates
  have row : edge 1487 = PrimePairCache.smallRow1487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan76_5 : scan edge (search edge 3) 4 [401] 401 innerTail76_5 best = best := by
  rw [innerTail76_5, scan, if_neg (by decide), innerCandidates76_5_checked]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_6
    (if 3 ≤ innerCandidates76_5.length then search edge 3 [401, 1487] innerCandidates76_5 1888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan76_6

def innerTail76_4 : List Nat := 1289 :: innerTail76_5
def innerCandidates76_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates76_4_checked : candidates edge 4 401 1289 innerTail76_5 best.bound = innerCandidates76_4 := by
  unfold candidates
  have row : edge 1289 = PrimePairCache.smallRow1289.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan76_4 : scan edge (search edge 3) 4 [401] 401 innerTail76_4 best = best := by
  rw [innerTail76_4, scan, if_neg (by decide), innerCandidates76_4_checked]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_5
    (if 3 ≤ innerCandidates76_4.length then search edge 3 [401, 1289] innerCandidates76_4 1690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan76_5

def innerTail76_3 : List Nat := 887 :: innerTail76_4
def innerCandidates76_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates76_3_checked : candidates edge 4 401 887 innerTail76_4 best.bound = innerCandidates76_3 := by
  unfold candidates
  have row : edge 887 = PrimePairCache.smallRow887.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan76_3 : scan edge (search edge 3) 4 [401] 401 innerTail76_3 best = best := by
  rw [innerTail76_3, scan, if_neg (by decide), innerCandidates76_3_checked]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_4
    (if 3 ≤ innerCandidates76_3.length then search edge 3 [401, 887] innerCandidates76_3 1288 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan76_4

def innerTail76_2 : List Nat := 809 :: innerTail76_3
def innerCandidates76_2 : List Nat := [4271]
set_option maxRecDepth 4096 in
theorem innerCandidates76_2_checked : candidates edge 4 401 809 innerTail76_3 best.bound = innerCandidates76_2 := by
  unfold candidates
  have row : edge 809 = PrimePairCache.smallRow809.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan76_2 : scan edge (search edge 3) 4 [401] 401 innerTail76_2 best = best := by
  rw [innerTail76_2, scan, if_neg (by decide), innerCandidates76_2_checked]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_3
    (if 3 ≤ innerCandidates76_2.length then search edge 3 [401, 809] innerCandidates76_2 1210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan76_3

def innerTail76_1 : List Nat := 743 :: innerTail76_2
def innerCandidates76_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates76_1_checked : candidates edge 4 401 743 innerTail76_2 best.bound = innerCandidates76_1 := by
  unfold candidates
  have row : edge 743 = PrimePairCache.smallRow743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan76_1 : scan edge (search edge 3) 4 [401] 401 innerTail76_1 best = best := by
  rw [innerTail76_1, scan, if_neg (by decide), innerCandidates76_1_checked]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_2
    (if 3 ≤ innerCandidates76_1.length then search edge 3 [401, 743] innerCandidates76_1 1144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan76_2

def innerTail76_0 : List Nat := 593 :: innerTail76_1
def innerCandidates76_0 : List Nat := [4133, 4211]
set_option maxRecDepth 4096 in
theorem innerCandidates76_0_checked : candidates edge 4 401 593 innerTail76_1 best.bound = innerCandidates76_0 := by
  unfold candidates
  have row : edge 593 = PrimePairCache.smallRow593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan76_0 : scan edge (search edge 3) 4 [401] 401 innerTail76_0 best = best := by
  rw [innerTail76_0, scan, if_neg (by decide), innerCandidates76_0_checked]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_1
    (if 3 ≤ innerCandidates76_0.length then search edge 3 [401, 593] innerCandidates76_0 994 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan76_1

theorem rootBranch76 : search edge 4 [401] rootCandidates76 401 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [401] 401 innerTail76_0 best = best
  exact innerScan76_0

def rootCandidates75 : List Nat := [547, 673, 907, 1063, 1483, 1621, 1861, 2251, 2593, 2851, 3511, 3727, 3847, 3877, 3919, 3943, 4057, 4441, 4483, 4567, 4657, 4759, 4987, 5197, 5749, 5791, 6121, 6397, 6703, 6829, 7177, 7639, 7741, 7927, 7933, 8101, 8329]
set_option maxRecDepth 4096 in
theorem rootCandidates75_checked : candidates edge 5 0 397 tail76 best.bound = rootCandidates75 := by
  unfold candidates
  have row : edge 397 = PrimePairCache.smallRow397.contains := by funext q; rfl
  rw [row]
  decide

def innerTail75_37 : List Nat := []
theorem innerScan75_37 : scan edge (search edge 3) 4 [397] 397 innerTail75_37 best = best := rfl

def innerTail75_36 : List Nat := 8329 :: innerTail75_37
def innerCandidates75_36 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates75_36_checked : candidates edge 4 397 8329 innerTail75_37 best.bound = innerCandidates75_36 := by
  unfold candidates
  have row : edge 8329 = SearchData.row8329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_36 : scan edge (search edge 3) 4 [397] 397 innerTail75_36 best = best := by
  rw [innerTail75_36, scan, if_neg (by decide), innerCandidates75_36_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_37
    (if 3 ≤ innerCandidates75_36.length then search edge 3 [397, 8329] innerCandidates75_36 8726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_37

def innerTail75_35 : List Nat := 8101 :: innerTail75_36
def innerCandidates75_35 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates75_35_checked : candidates edge 4 397 8101 innerTail75_36 best.bound = innerCandidates75_35 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_35 : scan edge (search edge 3) 4 [397] 397 innerTail75_35 best = best := by
  rw [innerTail75_35, scan, if_neg (by decide), innerCandidates75_35_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_36
    (if 3 ≤ innerCandidates75_35.length then search edge 3 [397, 8101] innerCandidates75_35 8498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_36

def innerTail75_34 : List Nat := 7933 :: innerTail75_35
def innerCandidates75_34 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates75_34_checked : candidates edge 4 397 7933 innerTail75_35 best.bound = innerCandidates75_34 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_34 : scan edge (search edge 3) 4 [397] 397 innerTail75_34 best = best := by
  rw [innerTail75_34, scan, if_neg (by decide), innerCandidates75_34_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_35
    (if 3 ≤ innerCandidates75_34.length then search edge 3 [397, 7933] innerCandidates75_34 8330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_35

def innerTail75_33 : List Nat := 7927 :: innerTail75_34
def innerCandidates75_33 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates75_33_checked : candidates edge 4 397 7927 innerTail75_34 best.bound = innerCandidates75_33 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_33 : scan edge (search edge 3) 4 [397] 397 innerTail75_33 best = best := by
  rw [innerTail75_33, scan, if_neg (by decide), innerCandidates75_33_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_34
    (if 3 ≤ innerCandidates75_33.length then search edge 3 [397, 7927] innerCandidates75_33 8324 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_34

def innerTail75_32 : List Nat := 7741 :: innerTail75_33
def innerCandidates75_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates75_32_checked : candidates edge 4 397 7741 innerTail75_33 best.bound = innerCandidates75_32 := by
  unfold candidates
  have row : edge 7741 = SearchData.row7741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_32 : scan edge (search edge 3) 4 [397] 397 innerTail75_32 best = best := by
  rw [innerTail75_32, scan, if_neg (by decide), innerCandidates75_32_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_33
    (if 3 ≤ innerCandidates75_32.length then search edge 3 [397, 7741] innerCandidates75_32 8138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_33

def innerTail75_31 : List Nat := 7639 :: innerTail75_32
def innerCandidates75_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates75_31_checked : candidates edge 4 397 7639 innerTail75_32 best.bound = innerCandidates75_31 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_31 : scan edge (search edge 3) 4 [397] 397 innerTail75_31 best = best := by
  rw [innerTail75_31, scan, if_neg (by decide), innerCandidates75_31_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_32
    (if 3 ≤ innerCandidates75_31.length then search edge 3 [397, 7639] innerCandidates75_31 8036 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_32

def innerTail75_30 : List Nat := 7177 :: innerTail75_31
def innerCandidates75_30 : List Nat := [7933]
set_option maxRecDepth 4096 in
theorem innerCandidates75_30_checked : candidates edge 4 397 7177 innerTail75_31 best.bound = innerCandidates75_30 := by
  unfold candidates
  have row : edge 7177 = SearchData.row7177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_30 : scan edge (search edge 3) 4 [397] 397 innerTail75_30 best = best := by
  rw [innerTail75_30, scan, if_neg (by decide), innerCandidates75_30_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_31
    (if 3 ≤ innerCandidates75_30.length then search edge 3 [397, 7177] innerCandidates75_30 7574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_31

def innerTail75_29 : List Nat := 6829 :: innerTail75_30
def innerCandidates75_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates75_29_checked : candidates edge 4 397 6829 innerTail75_30 best.bound = innerCandidates75_29 := by
  unfold candidates
  have row : edge 6829 = SearchData.row6829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_29 : scan edge (search edge 3) 4 [397] 397 innerTail75_29 best = best := by
  rw [innerTail75_29, scan, if_neg (by decide), innerCandidates75_29_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_30
    (if 3 ≤ innerCandidates75_29.length then search edge 3 [397, 6829] innerCandidates75_29 7226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_30

def innerTail75_28 : List Nat := 6703 :: innerTail75_29
def innerCandidates75_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates75_28_checked : candidates edge 4 397 6703 innerTail75_29 best.bound = innerCandidates75_28 := by
  unfold candidates
  have row : edge 6703 = SearchData.row6703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_28 : scan edge (search edge 3) 4 [397] 397 innerTail75_28 best = best := by
  rw [innerTail75_28, scan, if_neg (by decide), innerCandidates75_28_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_29
    (if 3 ≤ innerCandidates75_28.length then search edge 3 [397, 6703] innerCandidates75_28 7100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_29

def innerTail75_27 : List Nat := 6397 :: innerTail75_28
def innerCandidates75_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates75_27_checked : candidates edge 4 397 6397 innerTail75_28 best.bound = innerCandidates75_27 := by
  unfold candidates
  have row : edge 6397 = SearchData.row6397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_27 : scan edge (search edge 3) 4 [397] 397 innerTail75_27 best = best := by
  rw [innerTail75_27, scan, if_neg (by decide), innerCandidates75_27_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_28
    (if 3 ≤ innerCandidates75_27.length then search edge 3 [397, 6397] innerCandidates75_27 6794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_28

def innerTail75_26 : List Nat := 6121 :: innerTail75_27
def innerCandidates75_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates75_26_checked : candidates edge 4 397 6121 innerTail75_27 best.bound = innerCandidates75_26 := by
  unfold candidates
  have row : edge 6121 = SearchData.row6121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_26 : scan edge (search edge 3) 4 [397] 397 innerTail75_26 best = best := by
  rw [innerTail75_26, scan, if_neg (by decide), innerCandidates75_26_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_27
    (if 3 ≤ innerCandidates75_26.length then search edge 3 [397, 6121] innerCandidates75_26 6518 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_27

def innerTail75_25 : List Nat := 5791 :: innerTail75_26
def innerCandidates75_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates75_25_checked : candidates edge 4 397 5791 innerTail75_26 best.bound = innerCandidates75_25 := by
  unfold candidates
  have row : edge 5791 = SearchData.row5791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_25 : scan edge (search edge 3) 4 [397] 397 innerTail75_25 best = best := by
  rw [innerTail75_25, scan, if_neg (by decide), innerCandidates75_25_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_26
    (if 3 ≤ innerCandidates75_25.length then search edge 3 [397, 5791] innerCandidates75_25 6188 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_26

def innerTail75_24 : List Nat := 5749 :: innerTail75_25
def innerCandidates75_24 : List Nat := [8101]
set_option maxRecDepth 4096 in
theorem innerCandidates75_24_checked : candidates edge 4 397 5749 innerTail75_25 best.bound = innerCandidates75_24 := by
  unfold candidates
  have row : edge 5749 = SearchData.row5749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_24 : scan edge (search edge 3) 4 [397] 397 innerTail75_24 best = best := by
  rw [innerTail75_24, scan, if_neg (by decide), innerCandidates75_24_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_25
    (if 3 ≤ innerCandidates75_24.length then search edge 3 [397, 5749] innerCandidates75_24 6146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_25

def innerTail75_23 : List Nat := 5197 :: innerTail75_24
def innerCandidates75_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates75_23_checked : candidates edge 4 397 5197 innerTail75_24 best.bound = innerCandidates75_23 := by
  unfold candidates
  have row : edge 5197 = SearchData.row5197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_23 : scan edge (search edge 3) 4 [397] 397 innerTail75_23 best = best := by
  rw [innerTail75_23, scan, if_neg (by decide), innerCandidates75_23_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_24
    (if 3 ≤ innerCandidates75_23.length then search edge 3 [397, 5197] innerCandidates75_23 5594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_24

def innerTail75_22 : List Nat := 4987 :: innerTail75_23
def innerCandidates75_22 : List Nat := [5197]
set_option maxRecDepth 4096 in
theorem innerCandidates75_22_checked : candidates edge 4 397 4987 innerTail75_23 best.bound = innerCandidates75_22 := by
  unfold candidates
  have row : edge 4987 = SearchData.row4987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_22 : scan edge (search edge 3) 4 [397] 397 innerTail75_22 best = best := by
  rw [innerTail75_22, scan, if_neg (by decide), innerCandidates75_22_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_23
    (if 3 ≤ innerCandidates75_22.length then search edge 3 [397, 4987] innerCandidates75_22 5384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_23

def innerTail75_21 : List Nat := 4759 :: innerTail75_22
def innerCandidates75_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates75_21_checked : candidates edge 4 397 4759 innerTail75_22 best.bound = innerCandidates75_21 := by
  unfold candidates
  have row : edge 4759 = SearchData.row4759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_21 : scan edge (search edge 3) 4 [397] 397 innerTail75_21 best = best := by
  rw [innerTail75_21, scan, if_neg (by decide), innerCandidates75_21_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_22
    (if 3 ≤ innerCandidates75_21.length then search edge 3 [397, 4759] innerCandidates75_21 5156 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_22

def innerTail75_20 : List Nat := 4657 :: innerTail75_21
def innerCandidates75_20 : List Nat := [7177, 7933]
set_option maxRecDepth 4096 in
theorem innerCandidates75_20_checked : candidates edge 4 397 4657 innerTail75_21 best.bound = innerCandidates75_20 := by
  unfold candidates
  have row : edge 4657 = SearchData.row4657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_20 : scan edge (search edge 3) 4 [397] 397 innerTail75_20 best = best := by
  rw [innerTail75_20, scan, if_neg (by decide), innerCandidates75_20_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_21
    (if 3 ≤ innerCandidates75_20.length then search edge 3 [397, 4657] innerCandidates75_20 5054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_21

def innerTail75_19 : List Nat := 4567 :: innerTail75_20
def innerCandidates75_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates75_19_checked : candidates edge 4 397 4567 innerTail75_20 best.bound = innerCandidates75_19 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_19 : scan edge (search edge 3) 4 [397] 397 innerTail75_19 best = best := by
  rw [innerTail75_19, scan, if_neg (by decide), innerCandidates75_19_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_20
    (if 3 ≤ innerCandidates75_19.length then search edge 3 [397, 4567] innerCandidates75_19 4964 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_20

def innerTail75_18 : List Nat := 4483 :: innerTail75_19
def innerCandidates75_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates75_18_checked : candidates edge 4 397 4483 innerTail75_19 best.bound = innerCandidates75_18 := by
  unfold candidates
  have row : edge 4483 = SearchData.row4483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_18 : scan edge (search edge 3) 4 [397] 397 innerTail75_18 best = best := by
  rw [innerTail75_18, scan, if_neg (by decide), innerCandidates75_18_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_19
    (if 3 ≤ innerCandidates75_18.length then search edge 3 [397, 4483] innerCandidates75_18 4880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_19

def innerTail75_17 : List Nat := 4441 :: innerTail75_18
def innerCandidates75_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates75_17_checked : candidates edge 4 397 4441 innerTail75_18 best.bound = innerCandidates75_17 := by
  unfold candidates
  have row : edge 4441 = SearchData.row4441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_17 : scan edge (search edge 3) 4 [397] 397 innerTail75_17 best = best := by
  rw [innerTail75_17, scan, if_neg (by decide), innerCandidates75_17_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_18
    (if 3 ≤ innerCandidates75_17.length then search edge 3 [397, 4441] innerCandidates75_17 4838 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_18

def innerTail75_16 : List Nat := 4057 :: innerTail75_17
def innerCandidates75_16 : List Nat := [4483]
set_option maxRecDepth 4096 in
theorem innerCandidates75_16_checked : candidates edge 4 397 4057 innerTail75_17 best.bound = innerCandidates75_16 := by
  unfold candidates
  have row : edge 4057 = SearchData.row4057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_16 : scan edge (search edge 3) 4 [397] 397 innerTail75_16 best = best := by
  rw [innerTail75_16, scan, if_neg (by decide), innerCandidates75_16_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_17
    (if 3 ≤ innerCandidates75_16.length then search edge 3 [397, 4057] innerCandidates75_16 4454 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_17

def innerTail75_15 : List Nat := 3943 :: innerTail75_16
def innerCandidates75_15 : List Nat := [6829]
set_option maxRecDepth 4096 in
theorem innerCandidates75_15_checked : candidates edge 4 397 3943 innerTail75_16 best.bound = innerCandidates75_15 := by
  unfold candidates
  have row : edge 3943 = SearchData.row3943.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_15 : scan edge (search edge 3) 4 [397] 397 innerTail75_15 best = best := by
  rw [innerTail75_15, scan, if_neg (by decide), innerCandidates75_15_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_16
    (if 3 ≤ innerCandidates75_15.length then search edge 3 [397, 3943] innerCandidates75_15 4340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_16

def innerTail75_14 : List Nat := 3919 :: innerTail75_15
def innerCandidates75_14 : List Nat := [4567, 6397]
set_option maxRecDepth 4096 in
theorem innerCandidates75_14_checked : candidates edge 4 397 3919 innerTail75_15 best.bound = innerCandidates75_14 := by
  unfold candidates
  have row : edge 3919 = SearchData.row3919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_14 : scan edge (search edge 3) 4 [397] 397 innerTail75_14 best = best := by
  rw [innerTail75_14, scan, if_neg (by decide), innerCandidates75_14_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_15
    (if 3 ≤ innerCandidates75_14.length then search edge 3 [397, 3919] innerCandidates75_14 4316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_15

def innerTail75_13 : List Nat := 3877 :: innerTail75_14
def innerCandidates75_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates75_13_checked : candidates edge 4 397 3877 innerTail75_14 best.bound = innerCandidates75_13 := by
  unfold candidates
  have row : edge 3877 = SearchData.row3877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_13 : scan edge (search edge 3) 4 [397] 397 innerTail75_13 best = best := by
  rw [innerTail75_13, scan, if_neg (by decide), innerCandidates75_13_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_14
    (if 3 ≤ innerCandidates75_13.length then search edge 3 [397, 3877] innerCandidates75_13 4274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_14

def innerTail75_12 : List Nat := 3847 :: innerTail75_13
def innerCandidates75_12 : List Nat := [5197]
set_option maxRecDepth 4096 in
theorem innerCandidates75_12_checked : candidates edge 4 397 3847 innerTail75_13 best.bound = innerCandidates75_12 := by
  unfold candidates
  have row : edge 3847 = SearchData.row3847.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_12 : scan edge (search edge 3) 4 [397] 397 innerTail75_12 best = best := by
  rw [innerTail75_12, scan, if_neg (by decide), innerCandidates75_12_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_13
    (if 3 ≤ innerCandidates75_12.length then search edge 3 [397, 3847] innerCandidates75_12 4244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_13

def innerTail75_11 : List Nat := 3727 :: innerTail75_12
def innerCandidates75_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates75_11_checked : candidates edge 4 397 3727 innerTail75_12 best.bound = innerCandidates75_11 := by
  unfold candidates
  have row : edge 3727 = SearchData.row3727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_11 : scan edge (search edge 3) 4 [397] 397 innerTail75_11 best = best := by
  rw [innerTail75_11, scan, if_neg (by decide), innerCandidates75_11_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_12
    (if 3 ≤ innerCandidates75_11.length then search edge 3 [397, 3727] innerCandidates75_11 4124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_12

def innerTail75_10 : List Nat := 3511 :: innerTail75_11
def innerCandidates75_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates75_10_checked : candidates edge 4 397 3511 innerTail75_11 best.bound = innerCandidates75_10 := by
  unfold candidates
  have row : edge 3511 = SearchData.row3511.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_10 : scan edge (search edge 3) 4 [397] 397 innerTail75_10 best = best := by
  rw [innerTail75_10, scan, if_neg (by decide), innerCandidates75_10_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_11
    (if 3 ≤ innerCandidates75_10.length then search edge 3 [397, 3511] innerCandidates75_10 3908 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_11

def innerTail75_9 : List Nat := 2851 :: innerTail75_10
def innerCandidates75_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates75_9_checked : candidates edge 4 397 2851 innerTail75_10 best.bound = innerCandidates75_9 := by
  unfold candidates
  have row : edge 2851 = SearchData.row2851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_9 : scan edge (search edge 3) 4 [397] 397 innerTail75_9 best = best := by
  rw [innerTail75_9, scan, if_neg (by decide), innerCandidates75_9_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_10
    (if 3 ≤ innerCandidates75_9.length then search edge 3 [397, 2851] innerCandidates75_9 3248 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_10

def innerTail75_8 : List Nat := 2593 :: innerTail75_9
def innerCandidates75_8 : List Nat := [3847, 7927]
set_option maxRecDepth 4096 in
theorem innerCandidates75_8_checked : candidates edge 4 397 2593 innerTail75_9 best.bound = innerCandidates75_8 := by
  unfold candidates
  have row : edge 2593 = PrimePairCache.smallRow2593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_8 : scan edge (search edge 3) 4 [397] 397 innerTail75_8 best = best := by
  rw [innerTail75_8, scan, if_neg (by decide), innerCandidates75_8_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_9
    (if 3 ≤ innerCandidates75_8.length then search edge 3 [397, 2593] innerCandidates75_8 2990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_9

def innerTail75_7 : List Nat := 2251 :: innerTail75_8
def innerCandidates75_7 : List Nat := [2593, 7927]
set_option maxRecDepth 4096 in
theorem innerCandidates75_7_checked : candidates edge 4 397 2251 innerTail75_8 best.bound = innerCandidates75_7 := by
  unfold candidates
  have row : edge 2251 = PrimePairCache.smallRow2251.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_7 : scan edge (search edge 3) 4 [397] 397 innerTail75_7 best = best := by
  rw [innerTail75_7, scan, if_neg (by decide), innerCandidates75_7_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_8
    (if 3 ≤ innerCandidates75_7.length then search edge 3 [397, 2251] innerCandidates75_7 2648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_8

def innerTail75_6 : List Nat := 1861 :: innerTail75_7
def innerCandidates75_6 : List Nat := [6121]
set_option maxRecDepth 4096 in
theorem innerCandidates75_6_checked : candidates edge 4 397 1861 innerTail75_7 best.bound = innerCandidates75_6 := by
  unfold candidates
  have row : edge 1861 = PrimePairCache.smallRow1861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_6 : scan edge (search edge 3) 4 [397] 397 innerTail75_6 best = best := by
  rw [innerTail75_6, scan, if_neg (by decide), innerCandidates75_6_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_7
    (if 3 ≤ innerCandidates75_6.length then search edge 3 [397, 1861] innerCandidates75_6 2258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_7

def innerTail75_5 : List Nat := 1621 :: innerTail75_6
def innerCandidates75_5 : List Nat := [2851, 3943, 4441, 7639, 7933]
set_option maxRecDepth 4096 in
theorem innerCandidates75_5_checked : candidates edge 4 397 1621 innerTail75_6 best.bound = innerCandidates75_5 := by
  unfold candidates
  have row : edge 1621 = PrimePairCache.smallRow1621.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild75_5 : search edge 3 [397, 1621] innerCandidates75_5 2018 best = best := by decide

theorem innerScan75_5 : scan edge (search edge 3) 4 [397] 397 innerTail75_5 best = best := by
  rw [innerTail75_5, scan, if_neg (by decide), innerCandidates75_5_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_6
    (if 3 ≤ innerCandidates75_5.length then search edge 3 [397, 1621] innerCandidates75_5 2018 best else best) = best
  rw [if_pos (by decide), innerChild75_5]
  exact innerScan75_6

def innerTail75_4 : List Nat := 1483 :: innerTail75_5
def innerCandidates75_4 : List Nat := [2593]
set_option maxRecDepth 4096 in
theorem innerCandidates75_4_checked : candidates edge 4 397 1483 innerTail75_5 best.bound = innerCandidates75_4 := by
  unfold candidates
  have row : edge 1483 = PrimePairCache.smallRow1483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_4 : scan edge (search edge 3) 4 [397] 397 innerTail75_4 best = best := by
  rw [innerTail75_4, scan, if_neg (by decide), innerCandidates75_4_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_5
    (if 3 ≤ innerCandidates75_4.length then search edge 3 [397, 1483] innerCandidates75_4 1880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_5

def innerTail75_3 : List Nat := 1063 :: innerTail75_4
def innerCandidates75_3 : List Nat := [3919, 5197]
set_option maxRecDepth 4096 in
theorem innerCandidates75_3_checked : candidates edge 4 397 1063 innerTail75_4 best.bound = innerCandidates75_3 := by
  unfold candidates
  have row : edge 1063 = PrimePairCache.smallRow1063.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_3 : scan edge (search edge 3) 4 [397] 397 innerTail75_3 best = best := by
  rw [innerTail75_3, scan, if_neg (by decide), innerCandidates75_3_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_4
    (if 3 ≤ innerCandidates75_3.length then search edge 3 [397, 1063] innerCandidates75_3 1460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_4

def innerTail75_2 : List Nat := 907 :: innerTail75_3
def innerCandidates75_2 : List Nat := [3919, 4567, 4987]
set_option maxRecDepth 4096 in
theorem innerCandidates75_2_checked : candidates edge 4 397 907 innerTail75_3 best.bound = innerCandidates75_2 := by
  unfold candidates
  have row : edge 907 = PrimePairCache.smallRow907.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild75_2 : search edge 3 [397, 907] innerCandidates75_2 1304 best = best := by decide

theorem innerScan75_2 : scan edge (search edge 3) 4 [397] 397 innerTail75_2 best = best := by
  rw [innerTail75_2, scan, if_neg (by decide), innerCandidates75_2_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_3
    (if 3 ≤ innerCandidates75_2.length then search edge 3 [397, 907] innerCandidates75_2 1304 best else best) = best
  rw [if_pos (by decide), innerChild75_2]
  exact innerScan75_3

def innerTail75_1 : List Nat := 673 :: innerTail75_2
def innerCandidates75_1 : List Nat := [7177]
set_option maxRecDepth 4096 in
theorem innerCandidates75_1_checked : candidates edge 4 397 673 innerTail75_2 best.bound = innerCandidates75_1 := by
  unfold candidates
  have row : edge 673 = PrimePairCache.smallRow673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_1 : scan edge (search edge 3) 4 [397] 397 innerTail75_1 best = best := by
  rw [innerTail75_1, scan, if_neg (by decide), innerCandidates75_1_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_2
    (if 3 ≤ innerCandidates75_1.length then search edge 3 [397, 673] innerCandidates75_1 1070 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_2

def innerTail75_0 : List Nat := 547 :: innerTail75_1
def innerCandidates75_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates75_0_checked : candidates edge 4 397 547 innerTail75_1 best.bound = innerCandidates75_0 := by
  unfold candidates
  have row : edge 547 = PrimePairCache.smallRow547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan75_0 : scan edge (search edge 3) 4 [397] 397 innerTail75_0 best = best := by
  rw [innerTail75_0, scan, if_neg (by decide), innerCandidates75_0_checked]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_1
    (if 3 ≤ innerCandidates75_0.length then search edge 3 [397, 547] innerCandidates75_0 944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan75_1

theorem rootBranch75 : search edge 4 [397] rootCandidates75 397 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [397] 397 innerTail75_0 best = best
  exact innerScan75_0

end ProjectEulerSolutions.PrimePairUpper.Search
