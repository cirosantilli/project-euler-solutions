import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates146 : List Nat := [919, 1129, 1381, 1543, 2659, 3307, 3391, 3847, 3943, 4507, 4549, 5023, 5623, 5683, 6199, 7213, 7219, 7417, 7639, 7669, 7759, 8101, 8221, 8353]
set_option maxRecDepth 4096 in
theorem rootCandidates146_checked : candidates edge 5 0 859 tail147 best.bound = rootCandidates146 := by
  unfold candidates
  have row : edge 859 = PrimePairCache.smallRow859.contains := by funext q; rfl
  rw [row]
  decide

def innerTail146_24 : List Nat := []
theorem innerScan146_24 : scan edge (search edge 3) 4 [859] 859 innerTail146_24 best = best := rfl

def innerTail146_23 : List Nat := 8353 :: innerTail146_24
def innerCandidates146_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates146_23_checked : candidates edge 4 859 8353 innerTail146_24 best.bound = innerCandidates146_23 := by
  unfold candidates
  have row : edge 8353 = SearchData.row8353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_23 : scan edge (search edge 3) 4 [859] 859 innerTail146_23 best = best := by
  rw [innerTail146_23, scan, if_neg (by decide), innerCandidates146_23_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_24
    (if 3 ≤ innerCandidates146_23.length then search edge 3 [859, 8353] innerCandidates146_23 9212 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_24

def innerTail146_22 : List Nat := 8221 :: innerTail146_23
def innerCandidates146_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates146_22_checked : candidates edge 4 859 8221 innerTail146_23 best.bound = innerCandidates146_22 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_22 : scan edge (search edge 3) 4 [859] 859 innerTail146_22 best = best := by
  rw [innerTail146_22, scan, if_neg (by decide), innerCandidates146_22_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_23
    (if 3 ≤ innerCandidates146_22.length then search edge 3 [859, 8221] innerCandidates146_22 9080 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_23

def innerTail146_21 : List Nat := 8101 :: innerTail146_22
def innerCandidates146_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates146_21_checked : candidates edge 4 859 8101 innerTail146_22 best.bound = innerCandidates146_21 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_21 : scan edge (search edge 3) 4 [859] 859 innerTail146_21 best = best := by
  rw [innerTail146_21, scan, if_neg (by decide), innerCandidates146_21_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_22
    (if 3 ≤ innerCandidates146_21.length then search edge 3 [859, 8101] innerCandidates146_21 8960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_22

def innerTail146_20 : List Nat := 7759 :: innerTail146_21
def innerCandidates146_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates146_20_checked : candidates edge 4 859 7759 innerTail146_21 best.bound = innerCandidates146_20 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_20 : scan edge (search edge 3) 4 [859] 859 innerTail146_20 best = best := by
  rw [innerTail146_20, scan, if_neg (by decide), innerCandidates146_20_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_21
    (if 3 ≤ innerCandidates146_20.length then search edge 3 [859, 7759] innerCandidates146_20 8618 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_21

def innerTail146_19 : List Nat := 7669 :: innerTail146_20
def innerCandidates146_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates146_19_checked : candidates edge 4 859 7669 innerTail146_20 best.bound = innerCandidates146_19 := by
  unfold candidates
  have row : edge 7669 = SearchData.row7669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_19 : scan edge (search edge 3) 4 [859] 859 innerTail146_19 best = best := by
  rw [innerTail146_19, scan, if_neg (by decide), innerCandidates146_19_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_20
    (if 3 ≤ innerCandidates146_19.length then search edge 3 [859, 7669] innerCandidates146_19 8528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_20

def innerTail146_18 : List Nat := 7639 :: innerTail146_19
def innerCandidates146_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates146_18_checked : candidates edge 4 859 7639 innerTail146_19 best.bound = innerCandidates146_18 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_18 : scan edge (search edge 3) 4 [859] 859 innerTail146_18 best = best := by
  rw [innerTail146_18, scan, if_neg (by decide), innerCandidates146_18_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_19
    (if 3 ≤ innerCandidates146_18.length then search edge 3 [859, 7639] innerCandidates146_18 8498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_19

def innerTail146_17 : List Nat := 7417 :: innerTail146_18
def innerCandidates146_17 : List Nat := [7639]
set_option maxRecDepth 4096 in
theorem innerCandidates146_17_checked : candidates edge 4 859 7417 innerTail146_18 best.bound = innerCandidates146_17 := by
  unfold candidates
  have row : edge 7417 = SearchData.row7417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_17 : scan edge (search edge 3) 4 [859] 859 innerTail146_17 best = best := by
  rw [innerTail146_17, scan, if_neg (by decide), innerCandidates146_17_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_18
    (if 3 ≤ innerCandidates146_17.length then search edge 3 [859, 7417] innerCandidates146_17 8276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_18

def innerTail146_16 : List Nat := 7219 :: innerTail146_17
def innerCandidates146_16 : List Nat := [7669]
set_option maxRecDepth 4096 in
theorem innerCandidates146_16_checked : candidates edge 4 859 7219 innerTail146_17 best.bound = innerCandidates146_16 := by
  unfold candidates
  have row : edge 7219 = SearchData.row7219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_16 : scan edge (search edge 3) 4 [859] 859 innerTail146_16 best = best := by
  rw [innerTail146_16, scan, if_neg (by decide), innerCandidates146_16_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_17
    (if 3 ≤ innerCandidates146_16.length then search edge 3 [859, 7219] innerCandidates146_16 8078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_17

def innerTail146_15 : List Nat := 7213 :: innerTail146_16
def innerCandidates146_15 : List Nat := [7669]
set_option maxRecDepth 4096 in
theorem innerCandidates146_15_checked : candidates edge 4 859 7213 innerTail146_16 best.bound = innerCandidates146_15 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_15 : scan edge (search edge 3) 4 [859] 859 innerTail146_15 best = best := by
  rw [innerTail146_15, scan, if_neg (by decide), innerCandidates146_15_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_16
    (if 3 ≤ innerCandidates146_15.length then search edge 3 [859, 7213] innerCandidates146_15 8072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_16

def innerTail146_14 : List Nat := 6199 :: innerTail146_15
def innerCandidates146_14 : List Nat := [7219]
set_option maxRecDepth 4096 in
theorem innerCandidates146_14_checked : candidates edge 4 859 6199 innerTail146_15 best.bound = innerCandidates146_14 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_14 : scan edge (search edge 3) 4 [859] 859 innerTail146_14 best = best := by
  rw [innerTail146_14, scan, if_neg (by decide), innerCandidates146_14_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_15
    (if 3 ≤ innerCandidates146_14.length then search edge 3 [859, 6199] innerCandidates146_14 7058 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_15

def innerTail146_13 : List Nat := 5683 :: innerTail146_14
def innerCandidates146_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates146_13_checked : candidates edge 4 859 5683 innerTail146_14 best.bound = innerCandidates146_13 := by
  unfold candidates
  have row : edge 5683 = SearchData.row5683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_13 : scan edge (search edge 3) 4 [859] 859 innerTail146_13 best = best := by
  rw [innerTail146_13, scan, if_neg (by decide), innerCandidates146_13_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_14
    (if 3 ≤ innerCandidates146_13.length then search edge 3 [859, 5683] innerCandidates146_13 6542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_14

def innerTail146_12 : List Nat := 5623 :: innerTail146_13
def innerCandidates146_12 : List Nat := [7639]
set_option maxRecDepth 4096 in
theorem innerCandidates146_12_checked : candidates edge 4 859 5623 innerTail146_13 best.bound = innerCandidates146_12 := by
  unfold candidates
  have row : edge 5623 = SearchData.row5623.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_12 : scan edge (search edge 3) 4 [859] 859 innerTail146_12 best = best := by
  rw [innerTail146_12, scan, if_neg (by decide), innerCandidates146_12_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_13
    (if 3 ≤ innerCandidates146_12.length then search edge 3 [859, 5623] innerCandidates146_12 6482 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_13

def innerTail146_11 : List Nat := 5023 :: innerTail146_12
def innerCandidates146_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates146_11_checked : candidates edge 4 859 5023 innerTail146_12 best.bound = innerCandidates146_11 := by
  unfold candidates
  have row : edge 5023 = SearchData.row5023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_11 : scan edge (search edge 3) 4 [859] 859 innerTail146_11 best = best := by
  rw [innerTail146_11, scan, if_neg (by decide), innerCandidates146_11_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_12
    (if 3 ≤ innerCandidates146_11.length then search edge 3 [859, 5023] innerCandidates146_11 5882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_12

def innerTail146_10 : List Nat := 4549 :: innerTail146_11
def innerCandidates146_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates146_10_checked : candidates edge 4 859 4549 innerTail146_11 best.bound = innerCandidates146_10 := by
  unfold candidates
  have row : edge 4549 = SearchData.row4549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_10 : scan edge (search edge 3) 4 [859] 859 innerTail146_10 best = best := by
  rw [innerTail146_10, scan, if_neg (by decide), innerCandidates146_10_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_11
    (if 3 ≤ innerCandidates146_10.length then search edge 3 [859, 4549] innerCandidates146_10 5408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_11

def innerTail146_9 : List Nat := 4507 :: innerTail146_10
def innerCandidates146_9 : List Nat := [7759]
set_option maxRecDepth 4096 in
theorem innerCandidates146_9_checked : candidates edge 4 859 4507 innerTail146_10 best.bound = innerCandidates146_9 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_9 : scan edge (search edge 3) 4 [859] 859 innerTail146_9 best = best := by
  rw [innerTail146_9, scan, if_neg (by decide), innerCandidates146_9_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_10
    (if 3 ≤ innerCandidates146_9.length then search edge 3 [859, 4507] innerCandidates146_9 5366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_10

def innerTail146_8 : List Nat := 3943 :: innerTail146_9
def innerCandidates146_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates146_8_checked : candidates edge 4 859 3943 innerTail146_9 best.bound = innerCandidates146_8 := by
  unfold candidates
  have row : edge 3943 = SearchData.row3943.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_8 : scan edge (search edge 3) 4 [859] 859 innerTail146_8 best = best := by
  rw [innerTail146_8, scan, if_neg (by decide), innerCandidates146_8_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_9
    (if 3 ≤ innerCandidates146_8.length then search edge 3 [859, 3943] innerCandidates146_8 4802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_9

def innerTail146_7 : List Nat := 3847 :: innerTail146_8
def innerCandidates146_7 : List Nat := [8221]
set_option maxRecDepth 4096 in
theorem innerCandidates146_7_checked : candidates edge 4 859 3847 innerTail146_8 best.bound = innerCandidates146_7 := by
  unfold candidates
  have row : edge 3847 = SearchData.row3847.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_7 : scan edge (search edge 3) 4 [859] 859 innerTail146_7 best = best := by
  rw [innerTail146_7, scan, if_neg (by decide), innerCandidates146_7_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_8
    (if 3 ≤ innerCandidates146_7.length then search edge 3 [859, 3847] innerCandidates146_7 4706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_8

def innerTail146_6 : List Nat := 3391 :: innerTail146_7
def innerCandidates146_6 : List Nat := [6199]
set_option maxRecDepth 4096 in
theorem innerCandidates146_6_checked : candidates edge 4 859 3391 innerTail146_7 best.bound = innerCandidates146_6 := by
  unfold candidates
  have row : edge 3391 = SearchData.row3391.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_6 : scan edge (search edge 3) 4 [859] 859 innerTail146_6 best = best := by
  rw [innerTail146_6, scan, if_neg (by decide), innerCandidates146_6_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_7
    (if 3 ≤ innerCandidates146_6.length then search edge 3 [859, 3391] innerCandidates146_6 4250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_7

def innerTail146_5 : List Nat := 3307 :: innerTail146_6
def innerCandidates146_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates146_5_checked : candidates edge 4 859 3307 innerTail146_6 best.bound = innerCandidates146_5 := by
  unfold candidates
  have row : edge 3307 = SearchData.row3307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_5 : scan edge (search edge 3) 4 [859] 859 innerTail146_5 best = best := by
  rw [innerTail146_5, scan, if_neg (by decide), innerCandidates146_5_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_6
    (if 3 ≤ innerCandidates146_5.length then search edge 3 [859, 3307] innerCandidates146_5 4166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_6

def innerTail146_4 : List Nat := 2659 :: innerTail146_5
def innerCandidates146_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates146_4_checked : candidates edge 4 859 2659 innerTail146_5 best.bound = innerCandidates146_4 := by
  unfold candidates
  have row : edge 2659 = PrimePairCache.smallRow2659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_4 : scan edge (search edge 3) 4 [859] 859 innerTail146_4 best = best := by
  rw [innerTail146_4, scan, if_neg (by decide), innerCandidates146_4_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_5
    (if 3 ≤ innerCandidates146_4.length then search edge 3 [859, 2659] innerCandidates146_4 3518 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_5

def innerTail146_3 : List Nat := 1543 :: innerTail146_4
def innerCandidates146_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates146_3_checked : candidates edge 4 859 1543 innerTail146_4 best.bound = innerCandidates146_3 := by
  unfold candidates
  have row : edge 1543 = PrimePairCache.smallRow1543.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_3 : scan edge (search edge 3) 4 [859] 859 innerTail146_3 best = best := by
  rw [innerTail146_3, scan, if_neg (by decide), innerCandidates146_3_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_4
    (if 3 ≤ innerCandidates146_3.length then search edge 3 [859, 1543] innerCandidates146_3 2402 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_4

def innerTail146_2 : List Nat := 1381 :: innerTail146_3
def innerCandidates146_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates146_2_checked : candidates edge 4 859 1381 innerTail146_3 best.bound = innerCandidates146_2 := by
  unfold candidates
  have row : edge 1381 = PrimePairCache.smallRow1381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_2 : scan edge (search edge 3) 4 [859] 859 innerTail146_2 best = best := by
  rw [innerTail146_2, scan, if_neg (by decide), innerCandidates146_2_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_3
    (if 3 ≤ innerCandidates146_2.length then search edge 3 [859, 1381] innerCandidates146_2 2240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_3

def innerTail146_1 : List Nat := 1129 :: innerTail146_2
def innerCandidates146_1 : List Nat := [7417]
set_option maxRecDepth 4096 in
theorem innerCandidates146_1_checked : candidates edge 4 859 1129 innerTail146_2 best.bound = innerCandidates146_1 := by
  unfold candidates
  have row : edge 1129 = PrimePairCache.smallRow1129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_1 : scan edge (search edge 3) 4 [859] 859 innerTail146_1 best = best := by
  rw [innerTail146_1, scan, if_neg (by decide), innerCandidates146_1_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_2
    (if 3 ≤ innerCandidates146_1.length then search edge 3 [859, 1129] innerCandidates146_1 1988 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_2

def innerTail146_0 : List Nat := 919 :: innerTail146_1
def innerCandidates146_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates146_0_checked : candidates edge 4 859 919 innerTail146_1 best.bound = innerCandidates146_0 := by
  unfold candidates
  have row : edge 919 = PrimePairCache.smallRow919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan146_0 : scan edge (search edge 3) 4 [859] 859 innerTail146_0 best = best := by
  rw [innerTail146_0, scan, if_neg (by decide), innerCandidates146_0_checked]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_1
    (if 3 ≤ innerCandidates146_0.length then search edge 3 [859, 919] innerCandidates146_0 1778 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan146_1

theorem rootBranch146 : search edge 4 [859] rootCandidates146 859 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [859] 859 innerTail146_0 best = best
  exact innerScan146_0

def rootCandidates145 : List Nat := [1301, 1601, 1973, 2633, 2693, 3413, 4079, 4463, 4877, 6131, 6329, 7187, 7307, 7529, 8147, 8243]
set_option maxRecDepth 4096 in
theorem rootCandidates145_checked : candidates edge 5 0 857 tail146 best.bound = rootCandidates145 := by
  unfold candidates
  have row : edge 857 = PrimePairCache.smallRow857.contains := by funext q; rfl
  rw [row]
  decide

def innerTail145_16 : List Nat := []
theorem innerScan145_16 : scan edge (search edge 3) 4 [857] 857 innerTail145_16 best = best := rfl

def innerTail145_15 : List Nat := 8243 :: innerTail145_16
def innerCandidates145_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates145_15_checked : candidates edge 4 857 8243 innerTail145_16 best.bound = innerCandidates145_15 := by
  unfold candidates
  have row : edge 8243 = SearchData.row8243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan145_15 : scan edge (search edge 3) 4 [857] 857 innerTail145_15 best = best := by
  rw [innerTail145_15, scan, if_neg (by decide), innerCandidates145_15_checked]
  change scan edge (search edge 3) 4 [857] 857 innerTail145_16
    (if 3 ≤ innerCandidates145_15.length then search edge 3 [857, 8243] innerCandidates145_15 9100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan145_16

def innerTail145_14 : List Nat := 8147 :: innerTail145_15
def innerCandidates145_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates145_14_checked : candidates edge 4 857 8147 innerTail145_15 best.bound = innerCandidates145_14 := by
  unfold candidates
  have row : edge 8147 = SearchData.row8147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan145_14 : scan edge (search edge 3) 4 [857] 857 innerTail145_14 best = best := by
  rw [innerTail145_14, scan, if_neg (by decide), innerCandidates145_14_checked]
  change scan edge (search edge 3) 4 [857] 857 innerTail145_15
    (if 3 ≤ innerCandidates145_14.length then search edge 3 [857, 8147] innerCandidates145_14 9004 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan145_15

def innerTail145_13 : List Nat := 7529 :: innerTail145_14
def innerCandidates145_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates145_13_checked : candidates edge 4 857 7529 innerTail145_14 best.bound = innerCandidates145_13 := by
  unfold candidates
  have row : edge 7529 = SearchData.row7529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan145_13 : scan edge (search edge 3) 4 [857] 857 innerTail145_13 best = best := by
  rw [innerTail145_13, scan, if_neg (by decide), innerCandidates145_13_checked]
  change scan edge (search edge 3) 4 [857] 857 innerTail145_14
    (if 3 ≤ innerCandidates145_13.length then search edge 3 [857, 7529] innerCandidates145_13 8386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan145_14

def innerTail145_12 : List Nat := 7307 :: innerTail145_13
def innerCandidates145_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates145_12_checked : candidates edge 4 857 7307 innerTail145_13 best.bound = innerCandidates145_12 := by
  unfold candidates
  have row : edge 7307 = SearchData.row7307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan145_12 : scan edge (search edge 3) 4 [857] 857 innerTail145_12 best = best := by
  rw [innerTail145_12, scan, if_neg (by decide), innerCandidates145_12_checked]
  change scan edge (search edge 3) 4 [857] 857 innerTail145_13
    (if 3 ≤ innerCandidates145_12.length then search edge 3 [857, 7307] innerCandidates145_12 8164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan145_13

def innerTail145_11 : List Nat := 7187 :: innerTail145_12
def innerCandidates145_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates145_11_checked : candidates edge 4 857 7187 innerTail145_12 best.bound = innerCandidates145_11 := by
  unfold candidates
  have row : edge 7187 = SearchData.row7187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan145_11 : scan edge (search edge 3) 4 [857] 857 innerTail145_11 best = best := by
  rw [innerTail145_11, scan, if_neg (by decide), innerCandidates145_11_checked]
  change scan edge (search edge 3) 4 [857] 857 innerTail145_12
    (if 3 ≤ innerCandidates145_11.length then search edge 3 [857, 7187] innerCandidates145_11 8044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan145_12

def innerTail145_10 : List Nat := 6329 :: innerTail145_11
def innerCandidates145_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates145_10_checked : candidates edge 4 857 6329 innerTail145_11 best.bound = innerCandidates145_10 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan145_10 : scan edge (search edge 3) 4 [857] 857 innerTail145_10 best = best := by
  rw [innerTail145_10, scan, if_neg (by decide), innerCandidates145_10_checked]
  change scan edge (search edge 3) 4 [857] 857 innerTail145_11
    (if 3 ≤ innerCandidates145_10.length then search edge 3 [857, 6329] innerCandidates145_10 7186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan145_11

def innerTail145_9 : List Nat := 6131 :: innerTail145_10
def innerCandidates145_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates145_9_checked : candidates edge 4 857 6131 innerTail145_10 best.bound = innerCandidates145_9 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan145_9 : scan edge (search edge 3) 4 [857] 857 innerTail145_9 best = best := by
  rw [innerTail145_9, scan, if_neg (by decide), innerCandidates145_9_checked]
  change scan edge (search edge 3) 4 [857] 857 innerTail145_10
    (if 3 ≤ innerCandidates145_9.length then search edge 3 [857, 6131] innerCandidates145_9 6988 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan145_10

def innerTail145_8 : List Nat := 4877 :: innerTail145_9
def innerCandidates145_8 : List Nat := [7307]
set_option maxRecDepth 4096 in
theorem innerCandidates145_8_checked : candidates edge 4 857 4877 innerTail145_9 best.bound = innerCandidates145_8 := by
  unfold candidates
  have row : edge 4877 = SearchData.row4877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan145_8 : scan edge (search edge 3) 4 [857] 857 innerTail145_8 best = best := by
  rw [innerTail145_8, scan, if_neg (by decide), innerCandidates145_8_checked]
  change scan edge (search edge 3) 4 [857] 857 innerTail145_9
    (if 3 ≤ innerCandidates145_8.length then search edge 3 [857, 4877] innerCandidates145_8 5734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan145_9

def innerTail145_7 : List Nat := 4463 :: innerTail145_8
def innerCandidates145_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates145_7_checked : candidates edge 4 857 4463 innerTail145_8 best.bound = innerCandidates145_7 := by
  unfold candidates
  have row : edge 4463 = SearchData.row4463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan145_7 : scan edge (search edge 3) 4 [857] 857 innerTail145_7 best = best := by
  rw [innerTail145_7, scan, if_neg (by decide), innerCandidates145_7_checked]
  change scan edge (search edge 3) 4 [857] 857 innerTail145_8
    (if 3 ≤ innerCandidates145_7.length then search edge 3 [857, 4463] innerCandidates145_7 5320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan145_8

def innerTail145_6 : List Nat := 4079 :: innerTail145_7
def innerCandidates145_6 : List Nat := [6131]
set_option maxRecDepth 4096 in
theorem innerCandidates145_6_checked : candidates edge 4 857 4079 innerTail145_7 best.bound = innerCandidates145_6 := by
  unfold candidates
  have row : edge 4079 = SearchData.row4079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan145_6 : scan edge (search edge 3) 4 [857] 857 innerTail145_6 best = best := by
  rw [innerTail145_6, scan, if_neg (by decide), innerCandidates145_6_checked]
  change scan edge (search edge 3) 4 [857] 857 innerTail145_7
    (if 3 ≤ innerCandidates145_6.length then search edge 3 [857, 4079] innerCandidates145_6 4936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan145_7

def innerTail145_5 : List Nat := 3413 :: innerTail145_6
def innerCandidates145_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates145_5_checked : candidates edge 4 857 3413 innerTail145_6 best.bound = innerCandidates145_5 := by
  unfold candidates
  have row : edge 3413 = SearchData.row3413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan145_5 : scan edge (search edge 3) 4 [857] 857 innerTail145_5 best = best := by
  rw [innerTail145_5, scan, if_neg (by decide), innerCandidates145_5_checked]
  change scan edge (search edge 3) 4 [857] 857 innerTail145_6
    (if 3 ≤ innerCandidates145_5.length then search edge 3 [857, 3413] innerCandidates145_5 4270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan145_6

def innerTail145_4 : List Nat := 2693 :: innerTail145_5
def innerCandidates145_4 : List Nat := [4079]
set_option maxRecDepth 4096 in
theorem innerCandidates145_4_checked : candidates edge 4 857 2693 innerTail145_5 best.bound = innerCandidates145_4 := by
  unfold candidates
  have row : edge 2693 = SearchData.row2693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan145_4 : scan edge (search edge 3) 4 [857] 857 innerTail145_4 best = best := by
  rw [innerTail145_4, scan, if_neg (by decide), innerCandidates145_4_checked]
  change scan edge (search edge 3) 4 [857] 857 innerTail145_5
    (if 3 ≤ innerCandidates145_4.length then search edge 3 [857, 2693] innerCandidates145_4 3550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan145_5

def innerTail145_3 : List Nat := 2633 :: innerTail145_4
def innerCandidates145_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates145_3_checked : candidates edge 4 857 2633 innerTail145_4 best.bound = innerCandidates145_3 := by
  unfold candidates
  have row : edge 2633 = PrimePairCache.smallRow2633.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan145_3 : scan edge (search edge 3) 4 [857] 857 innerTail145_3 best = best := by
  rw [innerTail145_3, scan, if_neg (by decide), innerCandidates145_3_checked]
  change scan edge (search edge 3) 4 [857] 857 innerTail145_4
    (if 3 ≤ innerCandidates145_3.length then search edge 3 [857, 2633] innerCandidates145_3 3490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan145_4

def innerTail145_2 : List Nat := 1973 :: innerTail145_3
def innerCandidates145_2 : List Nat := [6131]
set_option maxRecDepth 4096 in
theorem innerCandidates145_2_checked : candidates edge 4 857 1973 innerTail145_3 best.bound = innerCandidates145_2 := by
  unfold candidates
  have row : edge 1973 = PrimePairCache.smallRow1973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan145_2 : scan edge (search edge 3) 4 [857] 857 innerTail145_2 best = best := by
  rw [innerTail145_2, scan, if_neg (by decide), innerCandidates145_2_checked]
  change scan edge (search edge 3) 4 [857] 857 innerTail145_3
    (if 3 ≤ innerCandidates145_2.length then search edge 3 [857, 1973] innerCandidates145_2 2830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan145_3

def innerTail145_1 : List Nat := 1601 :: innerTail145_2
def innerCandidates145_1 : List Nat := [7307]
set_option maxRecDepth 4096 in
theorem innerCandidates145_1_checked : candidates edge 4 857 1601 innerTail145_2 best.bound = innerCandidates145_1 := by
  unfold candidates
  have row : edge 1601 = PrimePairCache.smallRow1601.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan145_1 : scan edge (search edge 3) 4 [857] 857 innerTail145_1 best = best := by
  rw [innerTail145_1, scan, if_neg (by decide), innerCandidates145_1_checked]
  change scan edge (search edge 3) 4 [857] 857 innerTail145_2
    (if 3 ≤ innerCandidates145_1.length then search edge 3 [857, 1601] innerCandidates145_1 2458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan145_2

def innerTail145_0 : List Nat := 1301 :: innerTail145_1
def innerCandidates145_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates145_0_checked : candidates edge 4 857 1301 innerTail145_1 best.bound = innerCandidates145_0 := by
  unfold candidates
  have row : edge 1301 = PrimePairCache.smallRow1301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan145_0 : scan edge (search edge 3) 4 [857] 857 innerTail145_0 best = best := by
  rw [innerTail145_0, scan, if_neg (by decide), innerCandidates145_0_checked]
  change scan edge (search edge 3) 4 [857] 857 innerTail145_1
    (if 3 ≤ innerCandidates145_0.length then search edge 3 [857, 1301] innerCandidates145_0 2158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan145_1

theorem rootBranch145 : search edge 4 [857] rootCandidates145 857 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [857] 857 innerTail145_0 best = best
  exact innerScan145_0

def rootCandidates144 : List Nat := [1129, 1153, 1279, 1801, 2269, 2593, 2707, 2917, 3001, 3697, 3853, 4261, 4447, 4621, 4639, 4657, 4789, 5413, 6379, 6637, 6961, 7603, 7759, 8017]
set_option maxRecDepth 4096 in
theorem rootCandidates144_checked : candidates edge 5 0 853 tail145 best.bound = rootCandidates144 := by
  unfold candidates
  have row : edge 853 = PrimePairCache.smallRow853.contains := by funext q; rfl
  rw [row]
  decide

def innerTail144_24 : List Nat := []
theorem innerScan144_24 : scan edge (search edge 3) 4 [853] 853 innerTail144_24 best = best := rfl

def innerTail144_23 : List Nat := 8017 :: innerTail144_24
def innerCandidates144_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates144_23_checked : candidates edge 4 853 8017 innerTail144_24 best.bound = innerCandidates144_23 := by
  unfold candidates
  have row : edge 8017 = SearchData.row8017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan144_23 : scan edge (search edge 3) 4 [853] 853 innerTail144_23 best = best := by
  rw [innerTail144_23, scan, if_neg (by decide), innerCandidates144_23_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_24
    (if 3 ≤ innerCandidates144_23.length then search edge 3 [853, 8017] innerCandidates144_23 8870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan144_24

def innerTail144_22 : List Nat := 7759 :: innerTail144_23
def innerCandidates144_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates144_22_checked : candidates edge 4 853 7759 innerTail144_23 best.bound = innerCandidates144_22 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan144_22 : scan edge (search edge 3) 4 [853] 853 innerTail144_22 best = best := by
  rw [innerTail144_22, scan, if_neg (by decide), innerCandidates144_22_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_23
    (if 3 ≤ innerCandidates144_22.length then search edge 3 [853, 7759] innerCandidates144_22 8612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan144_23

def innerTail144_21 : List Nat := 7603 :: innerTail144_22
def innerCandidates144_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates144_21_checked : candidates edge 4 853 7603 innerTail144_22 best.bound = innerCandidates144_21 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan144_21 : scan edge (search edge 3) 4 [853] 853 innerTail144_21 best = best := by
  rw [innerTail144_21, scan, if_neg (by decide), innerCandidates144_21_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_22
    (if 3 ≤ innerCandidates144_21.length then search edge 3 [853, 7603] innerCandidates144_21 8456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan144_22

def innerTail144_20 : List Nat := 6961 :: innerTail144_21
def innerCandidates144_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates144_20_checked : candidates edge 4 853 6961 innerTail144_21 best.bound = innerCandidates144_20 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan144_20 : scan edge (search edge 3) 4 [853] 853 innerTail144_20 best = best := by
  rw [innerTail144_20, scan, if_neg (by decide), innerCandidates144_20_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_21
    (if 3 ≤ innerCandidates144_20.length then search edge 3 [853, 6961] innerCandidates144_20 7814 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan144_21

def innerTail144_19 : List Nat := 6637 :: innerTail144_20
def innerCandidates144_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates144_19_checked : candidates edge 4 853 6637 innerTail144_20 best.bound = innerCandidates144_19 := by
  unfold candidates
  have row : edge 6637 = SearchData.row6637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan144_19 : scan edge (search edge 3) 4 [853] 853 innerTail144_19 best = best := by
  rw [innerTail144_19, scan, if_neg (by decide), innerCandidates144_19_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_20
    (if 3 ≤ innerCandidates144_19.length then search edge 3 [853, 6637] innerCandidates144_19 7490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan144_20

def innerTail144_18 : List Nat := 6379 :: innerTail144_19
def innerCandidates144_18 : List Nat := [8017]
set_option maxRecDepth 4096 in
theorem innerCandidates144_18_checked : candidates edge 4 853 6379 innerTail144_19 best.bound = innerCandidates144_18 := by
  unfold candidates
  have row : edge 6379 = SearchData.row6379.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan144_18 : scan edge (search edge 3) 4 [853] 853 innerTail144_18 best = best := by
  rw [innerTail144_18, scan, if_neg (by decide), innerCandidates144_18_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_19
    (if 3 ≤ innerCandidates144_18.length then search edge 3 [853, 6379] innerCandidates144_18 7232 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan144_19

def innerTail144_17 : List Nat := 5413 :: innerTail144_18
def innerCandidates144_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates144_17_checked : candidates edge 4 853 5413 innerTail144_18 best.bound = innerCandidates144_17 := by
  unfold candidates
  have row : edge 5413 = SearchData.row5413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan144_17 : scan edge (search edge 3) 4 [853] 853 innerTail144_17 best = best := by
  rw [innerTail144_17, scan, if_neg (by decide), innerCandidates144_17_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_18
    (if 3 ≤ innerCandidates144_17.length then search edge 3 [853, 5413] innerCandidates144_17 6266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan144_18

def innerTail144_16 : List Nat := 4789 :: innerTail144_17
def innerCandidates144_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates144_16_checked : candidates edge 4 853 4789 innerTail144_17 best.bound = innerCandidates144_16 := by
  unfold candidates
  have row : edge 4789 = SearchData.row4789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan144_16 : scan edge (search edge 3) 4 [853] 853 innerTail144_16 best = best := by
  rw [innerTail144_16, scan, if_neg (by decide), innerCandidates144_16_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_17
    (if 3 ≤ innerCandidates144_16.length then search edge 3 [853, 4789] innerCandidates144_16 5642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan144_17

def innerTail144_15 : List Nat := 4657 :: innerTail144_16
def innerCandidates144_15 : List Nat := [4789]
set_option maxRecDepth 4096 in
theorem innerCandidates144_15_checked : candidates edge 4 853 4657 innerTail144_16 best.bound = innerCandidates144_15 := by
  unfold candidates
  have row : edge 4657 = SearchData.row4657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan144_15 : scan edge (search edge 3) 4 [853] 853 innerTail144_15 best = best := by
  rw [innerTail144_15, scan, if_neg (by decide), innerCandidates144_15_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_16
    (if 3 ≤ innerCandidates144_15.length then search edge 3 [853, 4657] innerCandidates144_15 5510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan144_16

def innerTail144_14 : List Nat := 4639 :: innerTail144_15
def innerCandidates144_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates144_14_checked : candidates edge 4 853 4639 innerTail144_15 best.bound = innerCandidates144_14 := by
  unfold candidates
  have row : edge 4639 = SearchData.row4639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan144_14 : scan edge (search edge 3) 4 [853] 853 innerTail144_14 best = best := by
  rw [innerTail144_14, scan, if_neg (by decide), innerCandidates144_14_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_15
    (if 3 ≤ innerCandidates144_14.length then search edge 3 [853, 4639] innerCandidates144_14 5492 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan144_15

def innerTail144_13 : List Nat := 4621 :: innerTail144_14
def innerCandidates144_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates144_13_checked : candidates edge 4 853 4621 innerTail144_14 best.bound = innerCandidates144_13 := by
  unfold candidates
  have row : edge 4621 = SearchData.row4621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan144_13 : scan edge (search edge 3) 4 [853] 853 innerTail144_13 best = best := by
  rw [innerTail144_13, scan, if_neg (by decide), innerCandidates144_13_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_14
    (if 3 ≤ innerCandidates144_13.length then search edge 3 [853, 4621] innerCandidates144_13 5474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan144_14

def innerTail144_12 : List Nat := 4447 :: innerTail144_13
def innerCandidates144_12 : List Nat := [4789]
set_option maxRecDepth 4096 in
theorem innerCandidates144_12_checked : candidates edge 4 853 4447 innerTail144_13 best.bound = innerCandidates144_12 := by
  unfold candidates
  have row : edge 4447 = SearchData.row4447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan144_12 : scan edge (search edge 3) 4 [853] 853 innerTail144_12 best = best := by
  rw [innerTail144_12, scan, if_neg (by decide), innerCandidates144_12_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_13
    (if 3 ≤ innerCandidates144_12.length then search edge 3 [853, 4447] innerCandidates144_12 5300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan144_13

def innerTail144_11 : List Nat := 4261 :: innerTail144_12
def innerCandidates144_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates144_11_checked : candidates edge 4 853 4261 innerTail144_12 best.bound = innerCandidates144_11 := by
  unfold candidates
  have row : edge 4261 = SearchData.row4261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan144_11 : scan edge (search edge 3) 4 [853] 853 innerTail144_11 best = best := by
  rw [innerTail144_11, scan, if_neg (by decide), innerCandidates144_11_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_12
    (if 3 ≤ innerCandidates144_11.length then search edge 3 [853, 4261] innerCandidates144_11 5114 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan144_12

def innerTail144_10 : List Nat := 3853 :: innerTail144_11
def innerCandidates144_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates144_10_checked : candidates edge 4 853 3853 innerTail144_11 best.bound = innerCandidates144_10 := by
  unfold candidates
  have row : edge 3853 = SearchData.row3853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan144_10 : scan edge (search edge 3) 4 [853] 853 innerTail144_10 best = best := by
  rw [innerTail144_10, scan, if_neg (by decide), innerCandidates144_10_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_11
    (if 3 ≤ innerCandidates144_10.length then search edge 3 [853, 3853] innerCandidates144_10 4706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan144_11

def innerTail144_9 : List Nat := 3697 :: innerTail144_10
def innerCandidates144_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates144_9_checked : candidates edge 4 853 3697 innerTail144_10 best.bound = innerCandidates144_9 := by
  unfold candidates
  have row : edge 3697 = SearchData.row3697.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan144_9 : scan edge (search edge 3) 4 [853] 853 innerTail144_9 best = best := by
  rw [innerTail144_9, scan, if_neg (by decide), innerCandidates144_9_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_10
    (if 3 ≤ innerCandidates144_9.length then search edge 3 [853, 3697] innerCandidates144_9 4550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan144_10

def innerTail144_8 : List Nat := 3001 :: innerTail144_9
def innerCandidates144_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates144_8_checked : candidates edge 4 853 3001 innerTail144_9 best.bound = innerCandidates144_8 := by
  unfold candidates
  have row : edge 3001 = SearchData.row3001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan144_8 : scan edge (search edge 3) 4 [853] 853 innerTail144_8 best = best := by
  rw [innerTail144_8, scan, if_neg (by decide), innerCandidates144_8_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_9
    (if 3 ≤ innerCandidates144_8.length then search edge 3 [853, 3001] innerCandidates144_8 3854 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan144_9

def innerTail144_7 : List Nat := 2917 :: innerTail144_8
def innerCandidates144_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates144_7_checked : candidates edge 4 853 2917 innerTail144_8 best.bound = innerCandidates144_7 := by
  unfold candidates
  have row : edge 2917 = SearchData.row2917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan144_7 : scan edge (search edge 3) 4 [853] 853 innerTail144_7 best = best := by
  rw [innerTail144_7, scan, if_neg (by decide), innerCandidates144_7_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_8
    (if 3 ≤ innerCandidates144_7.length then search edge 3 [853, 2917] innerCandidates144_7 3770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan144_8

def innerTail144_6 : List Nat := 2707 :: innerTail144_7
def innerCandidates144_6 : List Nat := [4261, 5413, 6961, 7759]
set_option maxRecDepth 4096 in
theorem innerCandidates144_6_checked : candidates edge 4 853 2707 innerTail144_7 best.bound = innerCandidates144_6 := by
  unfold candidates
  have row : edge 2707 = SearchData.row2707.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild144_6 : search edge 3 [853, 2707] innerCandidates144_6 3560 best = best := by decide

theorem innerScan144_6 : scan edge (search edge 3) 4 [853] 853 innerTail144_6 best = best := by
  rw [innerTail144_6, scan, if_neg (by decide), innerCandidates144_6_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_7
    (if 3 ≤ innerCandidates144_6.length then search edge 3 [853, 2707] innerCandidates144_6 3560 best else best) = best
  rw [if_pos (by decide), innerChild144_6]
  exact innerScan144_7

def innerTail144_5 : List Nat := 2593 :: innerTail144_6
def innerCandidates144_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates144_5_checked : candidates edge 4 853 2593 innerTail144_6 best.bound = innerCandidates144_5 := by
  unfold candidates
  have row : edge 2593 = PrimePairCache.smallRow2593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan144_5 : scan edge (search edge 3) 4 [853] 853 innerTail144_5 best = best := by
  rw [innerTail144_5, scan, if_neg (by decide), innerCandidates144_5_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_6
    (if 3 ≤ innerCandidates144_5.length then search edge 3 [853, 2593] innerCandidates144_5 3446 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan144_6

def innerTail144_4 : List Nat := 2269 :: innerTail144_5
def innerCandidates144_4 : List Nat := [4621]
set_option maxRecDepth 4096 in
theorem innerCandidates144_4_checked : candidates edge 4 853 2269 innerTail144_5 best.bound = innerCandidates144_4 := by
  unfold candidates
  have row : edge 2269 = PrimePairCache.smallRow2269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan144_4 : scan edge (search edge 3) 4 [853] 853 innerTail144_4 best = best := by
  rw [innerTail144_4, scan, if_neg (by decide), innerCandidates144_4_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_5
    (if 3 ≤ innerCandidates144_4.length then search edge 3 [853, 2269] innerCandidates144_4 3122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan144_5

def innerTail144_3 : List Nat := 1801 :: innerTail144_4
def innerCandidates144_3 : List Nat := [6637]
set_option maxRecDepth 4096 in
theorem innerCandidates144_3_checked : candidates edge 4 853 1801 innerTail144_4 best.bound = innerCandidates144_3 := by
  unfold candidates
  have row : edge 1801 = PrimePairCache.smallRow1801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan144_3 : scan edge (search edge 3) 4 [853] 853 innerTail144_3 best = best := by
  rw [innerTail144_3, scan, if_neg (by decide), innerCandidates144_3_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_4
    (if 3 ≤ innerCandidates144_3.length then search edge 3 [853, 1801] innerCandidates144_3 2654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan144_4

def innerTail144_2 : List Nat := 1279 :: innerTail144_3
def innerCandidates144_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates144_2_checked : candidates edge 4 853 1279 innerTail144_3 best.bound = innerCandidates144_2 := by
  unfold candidates
  have row : edge 1279 = PrimePairCache.smallRow1279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan144_2 : scan edge (search edge 3) 4 [853] 853 innerTail144_2 best = best := by
  rw [innerTail144_2, scan, if_neg (by decide), innerCandidates144_2_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_3
    (if 3 ≤ innerCandidates144_2.length then search edge 3 [853, 1279] innerCandidates144_2 2132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan144_3

def innerTail144_1 : List Nat := 1153 :: innerTail144_2
def innerCandidates144_1 : List Nat := [1801, 3853, 8017]
set_option maxRecDepth 4096 in
theorem innerCandidates144_1_checked : candidates edge 4 853 1153 innerTail144_2 best.bound = innerCandidates144_1 := by
  unfold candidates
  have row : edge 1153 = PrimePairCache.smallRow1153.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild144_1 : search edge 3 [853, 1153] innerCandidates144_1 2006 best = best := by decide

theorem innerScan144_1 : scan edge (search edge 3) 4 [853] 853 innerTail144_1 best = best := by
  rw [innerTail144_1, scan, if_neg (by decide), innerCandidates144_1_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_2
    (if 3 ≤ innerCandidates144_1.length then search edge 3 [853, 1153] innerCandidates144_1 2006 best else best) = best
  rw [if_pos (by decide), innerChild144_1]
  exact innerScan144_2

def innerTail144_0 : List Nat := 1129 :: innerTail144_1
def innerCandidates144_0 : List Nat := [2269, 3001, 7603]
set_option maxRecDepth 4096 in
theorem innerCandidates144_0_checked : candidates edge 4 853 1129 innerTail144_1 best.bound = innerCandidates144_0 := by
  unfold candidates
  have row : edge 1129 = PrimePairCache.smallRow1129.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild144_0 : search edge 3 [853, 1129] innerCandidates144_0 1982 best = best := by decide

theorem innerScan144_0 : scan edge (search edge 3) 4 [853] 853 innerTail144_0 best = best := by
  rw [innerTail144_0, scan, if_neg (by decide), innerCandidates144_0_checked]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_1
    (if 3 ≤ innerCandidates144_0.length then search edge 3 [853, 1129] innerCandidates144_0 1982 best else best) = best
  rw [if_pos (by decide), innerChild144_0]
  exact innerScan144_1

theorem rootBranch144 : search edge 4 [853] rootCandidates144 853 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [853] 853 innerTail144_0 best = best
  exact innerScan144_0

end ProjectEulerSolutions.PrimePairUpper.Search
