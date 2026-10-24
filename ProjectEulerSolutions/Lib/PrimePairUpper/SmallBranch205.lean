import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates92 : List Nat := [673, 691, 1009, 1033, 1237, 1249, 1303, 1453, 1933, 2551, 2707, 3343, 3847, 3907, 4051, 4357, 4513, 4621, 5281, 5503, 5827, 6361, 7393, 7687, 7717]
set_option maxRecDepth 4096 in
theorem rootCandidates92_checked : candidates edge 5 0 499 tail93 best.bound = rootCandidates92 := by
  unfold candidates
  have row : edge 499 = PrimePairCache.smallRow499.contains := by funext q; rfl
  rw [row]
  decide

def innerTail92_25 : List Nat := []
theorem innerScan92_25 : scan edge (search edge 3) 4 [499] 499 innerTail92_25 best = best := rfl

def innerTail92_24 : List Nat := 7717 :: innerTail92_25
def innerCandidates92_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates92_24_checked : candidates edge 4 499 7717 innerTail92_25 best.bound = innerCandidates92_24 := by
  unfold candidates
  have row : edge 7717 = SearchData.row7717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_24 : scan edge (search edge 3) 4 [499] 499 innerTail92_24 best = best := by
  rw [innerTail92_24, scan, if_neg (by decide), innerCandidates92_24_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_25
    (if 3 ≤ innerCandidates92_24.length then search edge 3 [499, 7717] innerCandidates92_24 8216 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_25

def innerTail92_23 : List Nat := 7687 :: innerTail92_24
def innerCandidates92_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates92_23_checked : candidates edge 4 499 7687 innerTail92_24 best.bound = innerCandidates92_23 := by
  unfold candidates
  have row : edge 7687 = SearchData.row7687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_23 : scan edge (search edge 3) 4 [499] 499 innerTail92_23 best = best := by
  rw [innerTail92_23, scan, if_neg (by decide), innerCandidates92_23_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_24
    (if 3 ≤ innerCandidates92_23.length then search edge 3 [499, 7687] innerCandidates92_23 8186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_24

def innerTail92_22 : List Nat := 7393 :: innerTail92_23
def innerCandidates92_22 : List Nat := [7687]
set_option maxRecDepth 4096 in
theorem innerCandidates92_22_checked : candidates edge 4 499 7393 innerTail92_23 best.bound = innerCandidates92_22 := by
  unfold candidates
  have row : edge 7393 = SearchData.row7393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_22 : scan edge (search edge 3) 4 [499] 499 innerTail92_22 best = best := by
  rw [innerTail92_22, scan, if_neg (by decide), innerCandidates92_22_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_23
    (if 3 ≤ innerCandidates92_22.length then search edge 3 [499, 7393] innerCandidates92_22 7892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_23

def innerTail92_21 : List Nat := 6361 :: innerTail92_22
def innerCandidates92_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates92_21_checked : candidates edge 4 499 6361 innerTail92_22 best.bound = innerCandidates92_21 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_21 : scan edge (search edge 3) 4 [499] 499 innerTail92_21 best = best := by
  rw [innerTail92_21, scan, if_neg (by decide), innerCandidates92_21_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_22
    (if 3 ≤ innerCandidates92_21.length then search edge 3 [499, 6361] innerCandidates92_21 6860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_22

def innerTail92_20 : List Nat := 5827 :: innerTail92_21
def innerCandidates92_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates92_20_checked : candidates edge 4 499 5827 innerTail92_21 best.bound = innerCandidates92_20 := by
  unfold candidates
  have row : edge 5827 = SearchData.row5827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_20 : scan edge (search edge 3) 4 [499] 499 innerTail92_20 best = best := by
  rw [innerTail92_20, scan, if_neg (by decide), innerCandidates92_20_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_21
    (if 3 ≤ innerCandidates92_20.length then search edge 3 [499, 5827] innerCandidates92_20 6326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_21

def innerTail92_19 : List Nat := 5503 :: innerTail92_20
def innerCandidates92_19 : List Nat := [7393]
set_option maxRecDepth 4096 in
theorem innerCandidates92_19_checked : candidates edge 4 499 5503 innerTail92_20 best.bound = innerCandidates92_19 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_19 : scan edge (search edge 3) 4 [499] 499 innerTail92_19 best = best := by
  rw [innerTail92_19, scan, if_neg (by decide), innerCandidates92_19_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_20
    (if 3 ≤ innerCandidates92_19.length then search edge 3 [499, 5503] innerCandidates92_19 6002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_20

def innerTail92_18 : List Nat := 5281 :: innerTail92_19
def innerCandidates92_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates92_18_checked : candidates edge 4 499 5281 innerTail92_19 best.bound = innerCandidates92_18 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_18 : scan edge (search edge 3) 4 [499] 499 innerTail92_18 best = best := by
  rw [innerTail92_18, scan, if_neg (by decide), innerCandidates92_18_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_19
    (if 3 ≤ innerCandidates92_18.length then search edge 3 [499, 5281] innerCandidates92_18 5780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_19

def innerTail92_17 : List Nat := 4621 :: innerTail92_18
def innerCandidates92_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates92_17_checked : candidates edge 4 499 4621 innerTail92_18 best.bound = innerCandidates92_17 := by
  unfold candidates
  have row : edge 4621 = SearchData.row4621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_17 : scan edge (search edge 3) 4 [499] 499 innerTail92_17 best = best := by
  rw [innerTail92_17, scan, if_neg (by decide), innerCandidates92_17_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_18
    (if 3 ≤ innerCandidates92_17.length then search edge 3 [499, 4621] innerCandidates92_17 5120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_18

def innerTail92_16 : List Nat := 4513 :: innerTail92_17
def innerCandidates92_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates92_16_checked : candidates edge 4 499 4513 innerTail92_17 best.bound = innerCandidates92_16 := by
  unfold candidates
  have row : edge 4513 = SearchData.row4513.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_16 : scan edge (search edge 3) 4 [499] 499 innerTail92_16 best = best := by
  rw [innerTail92_16, scan, if_neg (by decide), innerCandidates92_16_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_17
    (if 3 ≤ innerCandidates92_16.length then search edge 3 [499, 4513] innerCandidates92_16 5012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_17

def innerTail92_15 : List Nat := 4357 :: innerTail92_16
def innerCandidates92_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates92_15_checked : candidates edge 4 499 4357 innerTail92_16 best.bound = innerCandidates92_15 := by
  unfold candidates
  have row : edge 4357 = SearchData.row4357.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_15 : scan edge (search edge 3) 4 [499] 499 innerTail92_15 best = best := by
  rw [innerTail92_15, scan, if_neg (by decide), innerCandidates92_15_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_16
    (if 3 ≤ innerCandidates92_15.length then search edge 3 [499, 4357] innerCandidates92_15 4856 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_16

def innerTail92_14 : List Nat := 4051 :: innerTail92_15
def innerCandidates92_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates92_14_checked : candidates edge 4 499 4051 innerTail92_15 best.bound = innerCandidates92_14 := by
  unfold candidates
  have row : edge 4051 = SearchData.row4051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_14 : scan edge (search edge 3) 4 [499] 499 innerTail92_14 best = best := by
  rw [innerTail92_14, scan, if_neg (by decide), innerCandidates92_14_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_15
    (if 3 ≤ innerCandidates92_14.length then search edge 3 [499, 4051] innerCandidates92_14 4550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_15

def innerTail92_13 : List Nat := 3907 :: innerTail92_14
def innerCandidates92_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates92_13_checked : candidates edge 4 499 3907 innerTail92_14 best.bound = innerCandidates92_13 := by
  unfold candidates
  have row : edge 3907 = SearchData.row3907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_13 : scan edge (search edge 3) 4 [499] 499 innerTail92_13 best = best := by
  rw [innerTail92_13, scan, if_neg (by decide), innerCandidates92_13_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_14
    (if 3 ≤ innerCandidates92_13.length then search edge 3 [499, 3907] innerCandidates92_13 4406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_14

def innerTail92_12 : List Nat := 3847 :: innerTail92_13
def innerCandidates92_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates92_12_checked : candidates edge 4 499 3847 innerTail92_13 best.bound = innerCandidates92_12 := by
  unfold candidates
  have row : edge 3847 = SearchData.row3847.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_12 : scan edge (search edge 3) 4 [499] 499 innerTail92_12 best = best := by
  rw [innerTail92_12, scan, if_neg (by decide), innerCandidates92_12_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_13
    (if 3 ≤ innerCandidates92_12.length then search edge 3 [499, 3847] innerCandidates92_12 4346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_13

def innerTail92_11 : List Nat := 3343 :: innerTail92_12
def innerCandidates92_11 : List Nat := [7687]
set_option maxRecDepth 4096 in
theorem innerCandidates92_11_checked : candidates edge 4 499 3343 innerTail92_12 best.bound = innerCandidates92_11 := by
  unfold candidates
  have row : edge 3343 = SearchData.row3343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_11 : scan edge (search edge 3) 4 [499] 499 innerTail92_11 best = best := by
  rw [innerTail92_11, scan, if_neg (by decide), innerCandidates92_11_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_12
    (if 3 ≤ innerCandidates92_11.length then search edge 3 [499, 3343] innerCandidates92_11 3842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_12

def innerTail92_10 : List Nat := 2707 :: innerTail92_11
def innerCandidates92_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates92_10_checked : candidates edge 4 499 2707 innerTail92_11 best.bound = innerCandidates92_10 := by
  unfold candidates
  have row : edge 2707 = SearchData.row2707.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_10 : scan edge (search edge 3) 4 [499] 499 innerTail92_10 best = best := by
  rw [innerTail92_10, scan, if_neg (by decide), innerCandidates92_10_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_11
    (if 3 ≤ innerCandidates92_10.length then search edge 3 [499, 2707] innerCandidates92_10 3206 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_11

def innerTail92_9 : List Nat := 2551 :: innerTail92_10
def innerCandidates92_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates92_9_checked : candidates edge 4 499 2551 innerTail92_10 best.bound = innerCandidates92_9 := by
  unfold candidates
  have row : edge 2551 = PrimePairCache.smallRow2551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_9 : scan edge (search edge 3) 4 [499] 499 innerTail92_9 best = best := by
  rw [innerTail92_9, scan, if_neg (by decide), innerCandidates92_9_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_10
    (if 3 ≤ innerCandidates92_9.length then search edge 3 [499, 2551] innerCandidates92_9 3050 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_10

def innerTail92_8 : List Nat := 1933 :: innerTail92_9
def innerCandidates92_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates92_8_checked : candidates edge 4 499 1933 innerTail92_9 best.bound = innerCandidates92_8 := by
  unfold candidates
  have row : edge 1933 = PrimePairCache.smallRow1933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_8 : scan edge (search edge 3) 4 [499] 499 innerTail92_8 best = best := by
  rw [innerTail92_8, scan, if_neg (by decide), innerCandidates92_8_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_9
    (if 3 ≤ innerCandidates92_8.length then search edge 3 [499, 1933] innerCandidates92_8 2432 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_9

def innerTail92_7 : List Nat := 1453 :: innerTail92_8
def innerCandidates92_7 : List Nat := [5281]
set_option maxRecDepth 4096 in
theorem innerCandidates92_7_checked : candidates edge 4 499 1453 innerTail92_8 best.bound = innerCandidates92_7 := by
  unfold candidates
  have row : edge 1453 = PrimePairCache.smallRow1453.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_7 : scan edge (search edge 3) 4 [499] 499 innerTail92_7 best = best := by
  rw [innerTail92_7, scan, if_neg (by decide), innerCandidates92_7_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_8
    (if 3 ≤ innerCandidates92_7.length then search edge 3 [499, 1453] innerCandidates92_7 1952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_8

def innerTail92_6 : List Nat := 1303 :: innerTail92_7
def innerCandidates92_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates92_6_checked : candidates edge 4 499 1303 innerTail92_7 best.bound = innerCandidates92_6 := by
  unfold candidates
  have row : edge 1303 = PrimePairCache.smallRow1303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_6 : scan edge (search edge 3) 4 [499] 499 innerTail92_6 best = best := by
  rw [innerTail92_6, scan, if_neg (by decide), innerCandidates92_6_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_7
    (if 3 ≤ innerCandidates92_6.length then search edge 3 [499, 1303] innerCandidates92_6 1802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_7

def innerTail92_5 : List Nat := 1249 :: innerTail92_6
def innerCandidates92_5 : List Nat := [2551]
set_option maxRecDepth 4096 in
theorem innerCandidates92_5_checked : candidates edge 4 499 1249 innerTail92_6 best.bound = innerCandidates92_5 := by
  unfold candidates
  have row : edge 1249 = PrimePairCache.smallRow1249.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_5 : scan edge (search edge 3) 4 [499] 499 innerTail92_5 best = best := by
  rw [innerTail92_5, scan, if_neg (by decide), innerCandidates92_5_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_6
    (if 3 ≤ innerCandidates92_5.length then search edge 3 [499, 1249] innerCandidates92_5 1748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_6

def innerTail92_4 : List Nat := 1237 :: innerTail92_5
def innerCandidates92_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates92_4_checked : candidates edge 4 499 1237 innerTail92_5 best.bound = innerCandidates92_4 := by
  unfold candidates
  have row : edge 1237 = PrimePairCache.smallRow1237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_4 : scan edge (search edge 3) 4 [499] 499 innerTail92_4 best = best := by
  rw [innerTail92_4, scan, if_neg (by decide), innerCandidates92_4_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_5
    (if 3 ≤ innerCandidates92_4.length then search edge 3 [499, 1237] innerCandidates92_4 1736 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_5

def innerTail92_3 : List Nat := 1033 :: innerTail92_4
def innerCandidates92_3 : List Nat := [1249]
set_option maxRecDepth 4096 in
theorem innerCandidates92_3_checked : candidates edge 4 499 1033 innerTail92_4 best.bound = innerCandidates92_3 := by
  unfold candidates
  have row : edge 1033 = PrimePairCache.smallRow1033.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_3 : scan edge (search edge 3) 4 [499] 499 innerTail92_3 best = best := by
  rw [innerTail92_3, scan, if_neg (by decide), innerCandidates92_3_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_4
    (if 3 ≤ innerCandidates92_3.length then search edge 3 [499, 1033] innerCandidates92_3 1532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_4

def innerTail92_2 : List Nat := 1009 :: innerTail92_3
def innerCandidates92_2 : List Nat := [1933, 7393]
set_option maxRecDepth 4096 in
theorem innerCandidates92_2_checked : candidates edge 4 499 1009 innerTail92_3 best.bound = innerCandidates92_2 := by
  unfold candidates
  have row : edge 1009 = PrimePairCache.smallRow1009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_2 : scan edge (search edge 3) 4 [499] 499 innerTail92_2 best = best := by
  rw [innerTail92_2, scan, if_neg (by decide), innerCandidates92_2_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_3
    (if 3 ≤ innerCandidates92_2.length then search edge 3 [499, 1009] innerCandidates92_2 1508 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_3

def innerTail92_1 : List Nat := 691 :: innerTail92_2
def innerCandidates92_1 : List Nat := [1249, 5827]
set_option maxRecDepth 4096 in
theorem innerCandidates92_1_checked : candidates edge 4 499 691 innerTail92_2 best.bound = innerCandidates92_1 := by
  unfold candidates
  have row : edge 691 = PrimePairCache.smallRow691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_1 : scan edge (search edge 3) 4 [499] 499 innerTail92_1 best = best := by
  rw [innerTail92_1, scan, if_neg (by decide), innerCandidates92_1_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_2
    (if 3 ≤ innerCandidates92_1.length then search edge 3 [499, 691] innerCandidates92_1 1190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_2

def innerTail92_0 : List Nat := 673 :: innerTail92_1
def innerCandidates92_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates92_0_checked : candidates edge 4 499 673 innerTail92_1 best.bound = innerCandidates92_0 := by
  unfold candidates
  have row : edge 673 = PrimePairCache.smallRow673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan92_0 : scan edge (search edge 3) 4 [499] 499 innerTail92_0 best = best := by
  rw [innerTail92_0, scan, if_neg (by decide), innerCandidates92_0_checked]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_1
    (if 3 ≤ innerCandidates92_0.length then search edge 3 [499, 673] innerCandidates92_0 1172 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan92_1

theorem rootBranch92 : search edge 4 [499] rootCandidates92 499 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [499] 499 innerTail92_0 best = best
  exact innerScan92_0

def rootCandidates91 : List Nat := [593, 653, 773, 983, 1409, 1523, 1607, 1871, 2309, 2447, 2459, 2837, 3449, 3617, 3917, 3947, 4211, 4457, 5843, 6053, 6257, 6323, 6719, 6803, 7079, 7481, 7853]
set_option maxRecDepth 4096 in
theorem rootCandidates91_checked : candidates edge 5 0 491 tail92 best.bound = rootCandidates91 := by
  unfold candidates
  have row : edge 491 = PrimePairCache.smallRow491.contains := by funext q; rfl
  rw [row]
  decide

def innerTail91_27 : List Nat := []
theorem innerScan91_27 : scan edge (search edge 3) 4 [491] 491 innerTail91_27 best = best := rfl

def innerTail91_26 : List Nat := 7853 :: innerTail91_27
def innerCandidates91_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates91_26_checked : candidates edge 4 491 7853 innerTail91_27 best.bound = innerCandidates91_26 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_26 : scan edge (search edge 3) 4 [491] 491 innerTail91_26 best = best := by
  rw [innerTail91_26, scan, if_neg (by decide), innerCandidates91_26_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_27
    (if 3 ≤ innerCandidates91_26.length then search edge 3 [491, 7853] innerCandidates91_26 8344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_27

def innerTail91_25 : List Nat := 7481 :: innerTail91_26
def innerCandidates91_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates91_25_checked : candidates edge 4 491 7481 innerTail91_26 best.bound = innerCandidates91_25 := by
  unfold candidates
  have row : edge 7481 = SearchData.row7481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_25 : scan edge (search edge 3) 4 [491] 491 innerTail91_25 best = best := by
  rw [innerTail91_25, scan, if_neg (by decide), innerCandidates91_25_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_26
    (if 3 ≤ innerCandidates91_25.length then search edge 3 [491, 7481] innerCandidates91_25 7972 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_26

def innerTail91_24 : List Nat := 7079 :: innerTail91_25
def innerCandidates91_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates91_24_checked : candidates edge 4 491 7079 innerTail91_25 best.bound = innerCandidates91_24 := by
  unfold candidates
  have row : edge 7079 = SearchData.row7079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_24 : scan edge (search edge 3) 4 [491] 491 innerTail91_24 best = best := by
  rw [innerTail91_24, scan, if_neg (by decide), innerCandidates91_24_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_25
    (if 3 ≤ innerCandidates91_24.length then search edge 3 [491, 7079] innerCandidates91_24 7570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_25

def innerTail91_23 : List Nat := 6803 :: innerTail91_24
def innerCandidates91_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates91_23_checked : candidates edge 4 491 6803 innerTail91_24 best.bound = innerCandidates91_23 := by
  unfold candidates
  have row : edge 6803 = SearchData.row6803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_23 : scan edge (search edge 3) 4 [491] 491 innerTail91_23 best = best := by
  rw [innerTail91_23, scan, if_neg (by decide), innerCandidates91_23_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_24
    (if 3 ≤ innerCandidates91_23.length then search edge 3 [491, 6803] innerCandidates91_23 7294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_24

def innerTail91_22 : List Nat := 6719 :: innerTail91_23
def innerCandidates91_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates91_22_checked : candidates edge 4 491 6719 innerTail91_23 best.bound = innerCandidates91_22 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_22 : scan edge (search edge 3) 4 [491] 491 innerTail91_22 best = best := by
  rw [innerTail91_22, scan, if_neg (by decide), innerCandidates91_22_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_23
    (if 3 ≤ innerCandidates91_22.length then search edge 3 [491, 6719] innerCandidates91_22 7210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_23

def innerTail91_21 : List Nat := 6323 :: innerTail91_22
def innerCandidates91_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates91_21_checked : candidates edge 4 491 6323 innerTail91_22 best.bound = innerCandidates91_21 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_21 : scan edge (search edge 3) 4 [491] 491 innerTail91_21 best = best := by
  rw [innerTail91_21, scan, if_neg (by decide), innerCandidates91_21_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_22
    (if 3 ≤ innerCandidates91_21.length then search edge 3 [491, 6323] innerCandidates91_21 6814 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_22

def innerTail91_20 : List Nat := 6257 :: innerTail91_21
def innerCandidates91_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates91_20_checked : candidates edge 4 491 6257 innerTail91_21 best.bound = innerCandidates91_20 := by
  unfold candidates
  have row : edge 6257 = SearchData.row6257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_20 : scan edge (search edge 3) 4 [491] 491 innerTail91_20 best = best := by
  rw [innerTail91_20, scan, if_neg (by decide), innerCandidates91_20_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_21
    (if 3 ≤ innerCandidates91_20.length then search edge 3 [491, 6257] innerCandidates91_20 6748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_21

def innerTail91_19 : List Nat := 6053 :: innerTail91_20
def innerCandidates91_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates91_19_checked : candidates edge 4 491 6053 innerTail91_20 best.bound = innerCandidates91_19 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_19 : scan edge (search edge 3) 4 [491] 491 innerTail91_19 best = best := by
  rw [innerTail91_19, scan, if_neg (by decide), innerCandidates91_19_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_20
    (if 3 ≤ innerCandidates91_19.length then search edge 3 [491, 6053] innerCandidates91_19 6544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_20

def innerTail91_18 : List Nat := 5843 :: innerTail91_19
def innerCandidates91_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates91_18_checked : candidates edge 4 491 5843 innerTail91_19 best.bound = innerCandidates91_18 := by
  unfold candidates
  have row : edge 5843 = SearchData.row5843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_18 : scan edge (search edge 3) 4 [491] 491 innerTail91_18 best = best := by
  rw [innerTail91_18, scan, if_neg (by decide), innerCandidates91_18_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_19
    (if 3 ≤ innerCandidates91_18.length then search edge 3 [491, 5843] innerCandidates91_18 6334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_19

def innerTail91_17 : List Nat := 4457 :: innerTail91_18
def innerCandidates91_17 : List Nat := [7853]
set_option maxRecDepth 4096 in
theorem innerCandidates91_17_checked : candidates edge 4 491 4457 innerTail91_18 best.bound = innerCandidates91_17 := by
  unfold candidates
  have row : edge 4457 = SearchData.row4457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_17 : scan edge (search edge 3) 4 [491] 491 innerTail91_17 best = best := by
  rw [innerTail91_17, scan, if_neg (by decide), innerCandidates91_17_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_18
    (if 3 ≤ innerCandidates91_17.length then search edge 3 [491, 4457] innerCandidates91_17 4948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_18

def innerTail91_16 : List Nat := 4211 :: innerTail91_17
def innerCandidates91_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates91_16_checked : candidates edge 4 491 4211 innerTail91_17 best.bound = innerCandidates91_16 := by
  unfold candidates
  have row : edge 4211 = SearchData.row4211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_16 : scan edge (search edge 3) 4 [491] 491 innerTail91_16 best = best := by
  rw [innerTail91_16, scan, if_neg (by decide), innerCandidates91_16_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_17
    (if 3 ≤ innerCandidates91_16.length then search edge 3 [491, 4211] innerCandidates91_16 4702 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_17

def innerTail91_15 : List Nat := 3947 :: innerTail91_16
def innerCandidates91_15 : List Nat := [6803]
set_option maxRecDepth 4096 in
theorem innerCandidates91_15_checked : candidates edge 4 491 3947 innerTail91_16 best.bound = innerCandidates91_15 := by
  unfold candidates
  have row : edge 3947 = SearchData.row3947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_15 : scan edge (search edge 3) 4 [491] 491 innerTail91_15 best = best := by
  rw [innerTail91_15, scan, if_neg (by decide), innerCandidates91_15_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_16
    (if 3 ≤ innerCandidates91_15.length then search edge 3 [491, 3947] innerCandidates91_15 4438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_16

def innerTail91_14 : List Nat := 3917 :: innerTail91_15
def innerCandidates91_14 : List Nat := [4211, 6323]
set_option maxRecDepth 4096 in
theorem innerCandidates91_14_checked : candidates edge 4 491 3917 innerTail91_15 best.bound = innerCandidates91_14 := by
  unfold candidates
  have row : edge 3917 = SearchData.row3917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_14 : scan edge (search edge 3) 4 [491] 491 innerTail91_14 best = best := by
  rw [innerTail91_14, scan, if_neg (by decide), innerCandidates91_14_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_15
    (if 3 ≤ innerCandidates91_14.length then search edge 3 [491, 3917] innerCandidates91_14 4408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_15

def innerTail91_13 : List Nat := 3617 :: innerTail91_14
def innerCandidates91_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates91_13_checked : candidates edge 4 491 3617 innerTail91_14 best.bound = innerCandidates91_13 := by
  unfold candidates
  have row : edge 3617 = SearchData.row3617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_13 : scan edge (search edge 3) 4 [491] 491 innerTail91_13 best = best := by
  rw [innerTail91_13, scan, if_neg (by decide), innerCandidates91_13_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_14
    (if 3 ≤ innerCandidates91_13.length then search edge 3 [491, 3617] innerCandidates91_13 4108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_14

def innerTail91_12 : List Nat := 3449 :: innerTail91_13
def innerCandidates91_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates91_12_checked : candidates edge 4 491 3449 innerTail91_13 best.bound = innerCandidates91_12 := by
  unfold candidates
  have row : edge 3449 = SearchData.row3449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_12 : scan edge (search edge 3) 4 [491] 491 innerTail91_12 best = best := by
  rw [innerTail91_12, scan, if_neg (by decide), innerCandidates91_12_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_13
    (if 3 ≤ innerCandidates91_12.length then search edge 3 [491, 3449] innerCandidates91_12 3940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_13

def innerTail91_11 : List Nat := 2837 :: innerTail91_12
def innerCandidates91_11 : List Nat := [3617]
set_option maxRecDepth 4096 in
theorem innerCandidates91_11_checked : candidates edge 4 491 2837 innerTail91_12 best.bound = innerCandidates91_11 := by
  unfold candidates
  have row : edge 2837 = SearchData.row2837.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_11 : scan edge (search edge 3) 4 [491] 491 innerTail91_11 best = best := by
  rw [innerTail91_11, scan, if_neg (by decide), innerCandidates91_11_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_12
    (if 3 ≤ innerCandidates91_11.length then search edge 3 [491, 2837] innerCandidates91_11 3328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_12

def innerTail91_10 : List Nat := 2459 :: innerTail91_11
def innerCandidates91_10 : List Nat := [5843]
set_option maxRecDepth 4096 in
theorem innerCandidates91_10_checked : candidates edge 4 491 2459 innerTail91_11 best.bound = innerCandidates91_10 := by
  unfold candidates
  have row : edge 2459 = PrimePairCache.smallRow2459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_10 : scan edge (search edge 3) 4 [491] 491 innerTail91_10 best = best := by
  rw [innerTail91_10, scan, if_neg (by decide), innerCandidates91_10_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_11
    (if 3 ≤ innerCandidates91_10.length then search edge 3 [491, 2459] innerCandidates91_10 2950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_11

def innerTail91_9 : List Nat := 2447 :: innerTail91_10
def innerCandidates91_9 : List Nat := [6323]
set_option maxRecDepth 4096 in
theorem innerCandidates91_9_checked : candidates edge 4 491 2447 innerTail91_10 best.bound = innerCandidates91_9 := by
  unfold candidates
  have row : edge 2447 = PrimePairCache.smallRow2447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_9 : scan edge (search edge 3) 4 [491] 491 innerTail91_9 best = best := by
  rw [innerTail91_9, scan, if_neg (by decide), innerCandidates91_9_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_10
    (if 3 ≤ innerCandidates91_9.length then search edge 3 [491, 2447] innerCandidates91_9 2938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_10

def innerTail91_8 : List Nat := 2309 :: innerTail91_9
def innerCandidates91_8 : List Nat := [6257]
set_option maxRecDepth 4096 in
theorem innerCandidates91_8_checked : candidates edge 4 491 2309 innerTail91_9 best.bound = innerCandidates91_8 := by
  unfold candidates
  have row : edge 2309 = PrimePairCache.smallRow2309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_8 : scan edge (search edge 3) 4 [491] 491 innerTail91_8 best = best := by
  rw [innerTail91_8, scan, if_neg (by decide), innerCandidates91_8_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_9
    (if 3 ≤ innerCandidates91_8.length then search edge 3 [491, 2309] innerCandidates91_8 2800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_9

def innerTail91_7 : List Nat := 1871 :: innerTail91_8
def innerCandidates91_7 : List Nat := [6323]
set_option maxRecDepth 4096 in
theorem innerCandidates91_7_checked : candidates edge 4 491 1871 innerTail91_8 best.bound = innerCandidates91_7 := by
  unfold candidates
  have row : edge 1871 = PrimePairCache.smallRow1871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_7 : scan edge (search edge 3) 4 [491] 491 innerTail91_7 best = best := by
  rw [innerTail91_7, scan, if_neg (by decide), innerCandidates91_7_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_8
    (if 3 ≤ innerCandidates91_7.length then search edge 3 [491, 1871] innerCandidates91_7 2362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_8

def innerTail91_6 : List Nat := 1607 :: innerTail91_7
def innerCandidates91_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates91_6_checked : candidates edge 4 491 1607 innerTail91_7 best.bound = innerCandidates91_6 := by
  unfold candidates
  have row : edge 1607 = PrimePairCache.smallRow1607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_6 : scan edge (search edge 3) 4 [491] 491 innerTail91_6 best = best := by
  rw [innerTail91_6, scan, if_neg (by decide), innerCandidates91_6_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_7
    (if 3 ≤ innerCandidates91_6.length then search edge 3 [491, 1607] innerCandidates91_6 2098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_7

def innerTail91_5 : List Nat := 1523 :: innerTail91_6
def innerCandidates91_5 : List Nat := [2447]
set_option maxRecDepth 4096 in
theorem innerCandidates91_5_checked : candidates edge 4 491 1523 innerTail91_6 best.bound = innerCandidates91_5 := by
  unfold candidates
  have row : edge 1523 = PrimePairCache.smallRow1523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_5 : scan edge (search edge 3) 4 [491] 491 innerTail91_5 best = best := by
  rw [innerTail91_5, scan, if_neg (by decide), innerCandidates91_5_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_6
    (if 3 ≤ innerCandidates91_5.length then search edge 3 [491, 1523] innerCandidates91_5 2014 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_6

def innerTail91_4 : List Nat := 1409 :: innerTail91_5
def innerCandidates91_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates91_4_checked : candidates edge 4 491 1409 innerTail91_5 best.bound = innerCandidates91_4 := by
  unfold candidates
  have row : edge 1409 = PrimePairCache.smallRow1409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_4 : scan edge (search edge 3) 4 [491] 491 innerTail91_4 best = best := by
  rw [innerTail91_4, scan, if_neg (by decide), innerCandidates91_4_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_5
    (if 3 ≤ innerCandidates91_4.length then search edge 3 [491, 1409] innerCandidates91_4 1900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_5

def innerTail91_3 : List Nat := 983 :: innerTail91_4
def innerCandidates91_3 : List Nat := [3617, 6719, 7481]
set_option maxRecDepth 4096 in
theorem innerCandidates91_3_checked : candidates edge 4 491 983 innerTail91_4 best.bound = innerCandidates91_3 := by
  unfold candidates
  have row : edge 983 = PrimePairCache.smallRow983.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild91_3 : search edge 3 [491, 983] innerCandidates91_3 1474 best = best := by decide

theorem innerScan91_3 : scan edge (search edge 3) 4 [491] 491 innerTail91_3 best = best := by
  rw [innerTail91_3, scan, if_neg (by decide), innerCandidates91_3_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_4
    (if 3 ≤ innerCandidates91_3.length then search edge 3 [491, 983] innerCandidates91_3 1474 best else best) = best
  rw [if_pos (by decide), innerChild91_3]
  exact innerScan91_4

def innerTail91_2 : List Nat := 773 :: innerTail91_3
def innerCandidates91_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates91_2_checked : candidates edge 4 491 773 innerTail91_3 best.bound = innerCandidates91_2 := by
  unfold candidates
  have row : edge 773 = PrimePairCache.smallRow773.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_2 : scan edge (search edge 3) 4 [491] 491 innerTail91_2 best = best := by
  rw [innerTail91_2, scan, if_neg (by decide), innerCandidates91_2_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_3
    (if 3 ≤ innerCandidates91_2.length then search edge 3 [491, 773] innerCandidates91_2 1264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_3

def innerTail91_1 : List Nat := 653 :: innerTail91_2
def innerCandidates91_1 : List Nat := [1523]
set_option maxRecDepth 4096 in
theorem innerCandidates91_1_checked : candidates edge 4 491 653 innerTail91_2 best.bound = innerCandidates91_1 := by
  unfold candidates
  have row : edge 653 = PrimePairCache.smallRow653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_1 : scan edge (search edge 3) 4 [491] 491 innerTail91_1 best = best := by
  rw [innerTail91_1, scan, if_neg (by decide), innerCandidates91_1_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_2
    (if 3 ≤ innerCandidates91_1.length then search edge 3 [491, 653] innerCandidates91_1 1144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_2

def innerTail91_0 : List Nat := 593 :: innerTail91_1
def innerCandidates91_0 : List Nat := [4211]
set_option maxRecDepth 4096 in
theorem innerCandidates91_0_checked : candidates edge 4 491 593 innerTail91_1 best.bound = innerCandidates91_0 := by
  unfold candidates
  have row : edge 593 = PrimePairCache.smallRow593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan91_0 : scan edge (search edge 3) 4 [491] 491 innerTail91_0 best = best := by
  rw [innerTail91_0, scan, if_neg (by decide), innerCandidates91_0_checked]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_1
    (if 3 ≤ innerCandidates91_0.length then search edge 3 [491, 593] innerCandidates91_0 1084 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan91_1

theorem rootBranch91 : search edge 4 [491] rootCandidates91 491 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [491] 491 innerTail91_0 best = best
  exact innerScan91_0

end ProjectEulerSolutions.PrimePairUpper.Search
