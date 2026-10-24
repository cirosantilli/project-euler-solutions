import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates80 : List Nat := [479, 857, 929, 947, 983, 1031, 1163, 1193, 1361, 1823, 2393, 2417, 2549, 2837, 3671, 3851, 4217, 4943, 5903, 5981, 6197, 6329, 6803, 7127, 7283, 7517]
set_option maxRecDepth 4096 in
theorem rootCandidates80_checked : candidates edge 5 0 431 tail81 best.bound = rootCandidates80 := by
  unfold candidates
  have row : edge 431 = PrimePairCache.smallRow431.contains := by funext q; rfl
  rw [row]
  decide

def innerTail80_26 : List Nat := []
theorem innerScan80_26 : scan edge (search edge 3) 4 [431] 431 innerTail80_26 best = best := rfl

def innerTail80_25 : List Nat := 7517 :: innerTail80_26
def innerCandidates80_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates80_25_checked : candidates edge 4 431 7517 innerTail80_26 best.bound = innerCandidates80_25 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_25 : scan edge (search edge 3) 4 [431] 431 innerTail80_25 best = best := by
  rw [innerTail80_25, scan, if_neg (by decide), innerCandidates80_25_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_26
    (if 3 ≤ innerCandidates80_25.length then search edge 3 [431, 7517] innerCandidates80_25 7948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_26

def innerTail80_24 : List Nat := 7283 :: innerTail80_25
def innerCandidates80_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates80_24_checked : candidates edge 4 431 7283 innerTail80_25 best.bound = innerCandidates80_24 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_24 : scan edge (search edge 3) 4 [431] 431 innerTail80_24 best = best := by
  rw [innerTail80_24, scan, if_neg (by decide), innerCandidates80_24_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_25
    (if 3 ≤ innerCandidates80_24.length then search edge 3 [431, 7283] innerCandidates80_24 7714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_25

def innerTail80_23 : List Nat := 7127 :: innerTail80_24
def innerCandidates80_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates80_23_checked : candidates edge 4 431 7127 innerTail80_24 best.bound = innerCandidates80_23 := by
  unfold candidates
  have row : edge 7127 = SearchData.row7127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_23 : scan edge (search edge 3) 4 [431] 431 innerTail80_23 best = best := by
  rw [innerTail80_23, scan, if_neg (by decide), innerCandidates80_23_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_24
    (if 3 ≤ innerCandidates80_23.length then search edge 3 [431, 7127] innerCandidates80_23 7558 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_24

def innerTail80_22 : List Nat := 6803 :: innerTail80_23
def innerCandidates80_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates80_22_checked : candidates edge 4 431 6803 innerTail80_23 best.bound = innerCandidates80_22 := by
  unfold candidates
  have row : edge 6803 = SearchData.row6803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_22 : scan edge (search edge 3) 4 [431] 431 innerTail80_22 best = best := by
  rw [innerTail80_22, scan, if_neg (by decide), innerCandidates80_22_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_23
    (if 3 ≤ innerCandidates80_22.length then search edge 3 [431, 6803] innerCandidates80_22 7234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_23

def innerTail80_21 : List Nat := 6329 :: innerTail80_22
def innerCandidates80_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates80_21_checked : candidates edge 4 431 6329 innerTail80_22 best.bound = innerCandidates80_21 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_21 : scan edge (search edge 3) 4 [431] 431 innerTail80_21 best = best := by
  rw [innerTail80_21, scan, if_neg (by decide), innerCandidates80_21_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_22
    (if 3 ≤ innerCandidates80_21.length then search edge 3 [431, 6329] innerCandidates80_21 6760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_22

def innerTail80_20 : List Nat := 6197 :: innerTail80_21
def innerCandidates80_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates80_20_checked : candidates edge 4 431 6197 innerTail80_21 best.bound = innerCandidates80_20 := by
  unfold candidates
  have row : edge 6197 = SearchData.row6197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_20 : scan edge (search edge 3) 4 [431] 431 innerTail80_20 best = best := by
  rw [innerTail80_20, scan, if_neg (by decide), innerCandidates80_20_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_21
    (if 3 ≤ innerCandidates80_20.length then search edge 3 [431, 6197] innerCandidates80_20 6628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_21

def innerTail80_19 : List Nat := 5981 :: innerTail80_20
def innerCandidates80_19 : List Nat := [7283]
set_option maxRecDepth 4096 in
theorem innerCandidates80_19_checked : candidates edge 4 431 5981 innerTail80_20 best.bound = innerCandidates80_19 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_19 : scan edge (search edge 3) 4 [431] 431 innerTail80_19 best = best := by
  rw [innerTail80_19, scan, if_neg (by decide), innerCandidates80_19_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_20
    (if 3 ≤ innerCandidates80_19.length then search edge 3 [431, 5981] innerCandidates80_19 6412 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_20

def innerTail80_18 : List Nat := 5903 :: innerTail80_19
def innerCandidates80_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates80_18_checked : candidates edge 4 431 5903 innerTail80_19 best.bound = innerCandidates80_18 := by
  unfold candidates
  have row : edge 5903 = SearchData.row5903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_18 : scan edge (search edge 3) 4 [431] 431 innerTail80_18 best = best := by
  rw [innerTail80_18, scan, if_neg (by decide), innerCandidates80_18_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_19
    (if 3 ≤ innerCandidates80_18.length then search edge 3 [431, 5903] innerCandidates80_18 6334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_19

def innerTail80_17 : List Nat := 4943 :: innerTail80_18
def innerCandidates80_17 : List Nat := [7517]
set_option maxRecDepth 4096 in
theorem innerCandidates80_17_checked : candidates edge 4 431 4943 innerTail80_18 best.bound = innerCandidates80_17 := by
  unfold candidates
  have row : edge 4943 = SearchData.row4943.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_17 : scan edge (search edge 3) 4 [431] 431 innerTail80_17 best = best := by
  rw [innerTail80_17, scan, if_neg (by decide), innerCandidates80_17_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_18
    (if 3 ≤ innerCandidates80_17.length then search edge 3 [431, 4943] innerCandidates80_17 5374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_18

def innerTail80_16 : List Nat := 4217 :: innerTail80_17
def innerCandidates80_16 : List Nat := [5981]
set_option maxRecDepth 4096 in
theorem innerCandidates80_16_checked : candidates edge 4 431 4217 innerTail80_17 best.bound = innerCandidates80_16 := by
  unfold candidates
  have row : edge 4217 = SearchData.row4217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_16 : scan edge (search edge 3) 4 [431] 431 innerTail80_16 best = best := by
  rw [innerTail80_16, scan, if_neg (by decide), innerCandidates80_16_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_17
    (if 3 ≤ innerCandidates80_16.length then search edge 3 [431, 4217] innerCandidates80_16 4648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_17

def innerTail80_15 : List Nat := 3851 :: innerTail80_16
def innerCandidates80_15 : List Nat := [5903]
set_option maxRecDepth 4096 in
theorem innerCandidates80_15_checked : candidates edge 4 431 3851 innerTail80_16 best.bound = innerCandidates80_15 := by
  unfold candidates
  have row : edge 3851 = SearchData.row3851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_15 : scan edge (search edge 3) 4 [431] 431 innerTail80_15 best = best := by
  rw [innerTail80_15, scan, if_neg (by decide), innerCandidates80_15_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_16
    (if 3 ≤ innerCandidates80_15.length then search edge 3 [431, 3851] innerCandidates80_15 4282 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_16

def innerTail80_14 : List Nat := 3671 :: innerTail80_15
def innerCandidates80_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates80_14_checked : candidates edge 4 431 3671 innerTail80_15 best.bound = innerCandidates80_14 := by
  unfold candidates
  have row : edge 3671 = SearchData.row3671.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_14 : scan edge (search edge 3) 4 [431] 431 innerTail80_14 best = best := by
  rw [innerTail80_14, scan, if_neg (by decide), innerCandidates80_14_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_15
    (if 3 ≤ innerCandidates80_14.length then search edge 3 [431, 3671] innerCandidates80_14 4102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_15

def innerTail80_13 : List Nat := 2837 :: innerTail80_14
def innerCandidates80_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates80_13_checked : candidates edge 4 431 2837 innerTail80_14 best.bound = innerCandidates80_13 := by
  unfold candidates
  have row : edge 2837 = SearchData.row2837.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_13 : scan edge (search edge 3) 4 [431] 431 innerTail80_13 best = best := by
  rw [innerTail80_13, scan, if_neg (by decide), innerCandidates80_13_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_14
    (if 3 ≤ innerCandidates80_13.length then search edge 3 [431, 2837] innerCandidates80_13 3268 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_14

def innerTail80_12 : List Nat := 2549 :: innerTail80_13
def innerCandidates80_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates80_12_checked : candidates edge 4 431 2549 innerTail80_13 best.bound = innerCandidates80_12 := by
  unfold candidates
  have row : edge 2549 = PrimePairCache.smallRow2549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_12 : scan edge (search edge 3) 4 [431] 431 innerTail80_12 best = best := by
  rw [innerTail80_12, scan, if_neg (by decide), innerCandidates80_12_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_13
    (if 3 ≤ innerCandidates80_12.length then search edge 3 [431, 2549] innerCandidates80_12 2980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_13

def innerTail80_11 : List Nat := 2417 :: innerTail80_12
def innerCandidates80_11 : List Nat := [5903]
set_option maxRecDepth 4096 in
theorem innerCandidates80_11_checked : candidates edge 4 431 2417 innerTail80_12 best.bound = innerCandidates80_11 := by
  unfold candidates
  have row : edge 2417 = PrimePairCache.smallRow2417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_11 : scan edge (search edge 3) 4 [431] 431 innerTail80_11 best = best := by
  rw [innerTail80_11, scan, if_neg (by decide), innerCandidates80_11_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_12
    (if 3 ≤ innerCandidates80_11.length then search edge 3 [431, 2417] innerCandidates80_11 2848 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_12

def innerTail80_10 : List Nat := 2393 :: innerTail80_11
def innerCandidates80_10 : List Nat := [3851, 5981, 6329, 7127]
set_option maxRecDepth 4096 in
theorem innerCandidates80_10_checked : candidates edge 4 431 2393 innerTail80_11 best.bound = innerCandidates80_10 := by
  unfold candidates
  have row : edge 2393 = PrimePairCache.smallRow2393.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild80_10 : search edge 3 [431, 2393] innerCandidates80_10 2824 best = best := by decide

theorem innerScan80_10 : scan edge (search edge 3) 4 [431] 431 innerTail80_10 best = best := by
  rw [innerTail80_10, scan, if_neg (by decide), innerCandidates80_10_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_11
    (if 3 ≤ innerCandidates80_10.length then search edge 3 [431, 2393] innerCandidates80_10 2824 best else best) = best
  rw [if_pos (by decide), innerChild80_10]
  exact innerScan80_11

def innerTail80_9 : List Nat := 1823 :: innerTail80_10
def innerCandidates80_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates80_9_checked : candidates edge 4 431 1823 innerTail80_10 best.bound = innerCandidates80_9 := by
  unfold candidates
  have row : edge 1823 = PrimePairCache.smallRow1823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_9 : scan edge (search edge 3) 4 [431] 431 innerTail80_9 best = best := by
  rw [innerTail80_9, scan, if_neg (by decide), innerCandidates80_9_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_10
    (if 3 ≤ innerCandidates80_9.length then search edge 3 [431, 1823] innerCandidates80_9 2254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_10

def innerTail80_8 : List Nat := 1361 :: innerTail80_9
def innerCandidates80_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates80_8_checked : candidates edge 4 431 1361 innerTail80_9 best.bound = innerCandidates80_8 := by
  unfold candidates
  have row : edge 1361 = PrimePairCache.smallRow1361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_8 : scan edge (search edge 3) 4 [431] 431 innerTail80_8 best = best := by
  rw [innerTail80_8, scan, if_neg (by decide), innerCandidates80_8_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_9
    (if 3 ≤ innerCandidates80_8.length then search edge 3 [431, 1361] innerCandidates80_8 1792 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_9

def innerTail80_7 : List Nat := 1193 :: innerTail80_8
def innerCandidates80_7 : List Nat := [3671]
set_option maxRecDepth 4096 in
theorem innerCandidates80_7_checked : candidates edge 4 431 1193 innerTail80_8 best.bound = innerCandidates80_7 := by
  unfold candidates
  have row : edge 1193 = PrimePairCache.smallRow1193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_7 : scan edge (search edge 3) 4 [431] 431 innerTail80_7 best = best := by
  rw [innerTail80_7, scan, if_neg (by decide), innerCandidates80_7_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_8
    (if 3 ≤ innerCandidates80_7.length then search edge 3 [431, 1193] innerCandidates80_7 1624 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_8

def innerTail80_6 : List Nat := 1163 :: innerTail80_7
def innerCandidates80_6 : List Nat := [2417, 6197, 6803]
set_option maxRecDepth 4096 in
theorem innerCandidates80_6_checked : candidates edge 4 431 1163 innerTail80_7 best.bound = innerCandidates80_6 := by
  unfold candidates
  have row : edge 1163 = PrimePairCache.smallRow1163.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild80_6 : search edge 3 [431, 1163] innerCandidates80_6 1594 best = best := by decide

theorem innerScan80_6 : scan edge (search edge 3) 4 [431] 431 innerTail80_6 best = best := by
  rw [innerTail80_6, scan, if_neg (by decide), innerCandidates80_6_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_7
    (if 3 ≤ innerCandidates80_6.length then search edge 3 [431, 1163] innerCandidates80_6 1594 best else best) = best
  rw [if_pos (by decide), innerChild80_6]
  exact innerScan80_7

def innerTail80_5 : List Nat := 1031 :: innerTail80_6
def innerCandidates80_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates80_5_checked : candidates edge 4 431 1031 innerTail80_6 best.bound = innerCandidates80_5 := by
  unfold candidates
  have row : edge 1031 = PrimePairCache.smallRow1031.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_5 : scan edge (search edge 3) 4 [431] 431 innerTail80_5 best = best := by
  rw [innerTail80_5, scan, if_neg (by decide), innerCandidates80_5_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_6
    (if 3 ≤ innerCandidates80_5.length then search edge 3 [431, 1031] innerCandidates80_5 1462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_6

def innerTail80_4 : List Nat := 983 :: innerTail80_5
def innerCandidates80_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates80_4_checked : candidates edge 4 431 983 innerTail80_5 best.bound = innerCandidates80_4 := by
  unfold candidates
  have row : edge 983 = PrimePairCache.smallRow983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_4 : scan edge (search edge 3) 4 [431] 431 innerTail80_4 best = best := by
  rw [innerTail80_4, scan, if_neg (by decide), innerCandidates80_4_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_5
    (if 3 ≤ innerCandidates80_4.length then search edge 3 [431, 983] innerCandidates80_4 1414 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_5

def innerTail80_3 : List Nat := 947 :: innerTail80_4
def innerCandidates80_3 : List Nat := [5981]
set_option maxRecDepth 4096 in
theorem innerCandidates80_3_checked : candidates edge 4 431 947 innerTail80_4 best.bound = innerCandidates80_3 := by
  unfold candidates
  have row : edge 947 = PrimePairCache.smallRow947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_3 : scan edge (search edge 3) 4 [431] 431 innerTail80_3 best = best := by
  rw [innerTail80_3, scan, if_neg (by decide), innerCandidates80_3_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_4
    (if 3 ≤ innerCandidates80_3.length then search edge 3 [431, 947] innerCandidates80_3 1378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_4

def innerTail80_2 : List Nat := 929 :: innerTail80_3
def innerCandidates80_2 : List Nat := [983]
set_option maxRecDepth 4096 in
theorem innerCandidates80_2_checked : candidates edge 4 431 929 innerTail80_3 best.bound = innerCandidates80_2 := by
  unfold candidates
  have row : edge 929 = PrimePairCache.smallRow929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_2 : scan edge (search edge 3) 4 [431] 431 innerTail80_2 best = best := by
  rw [innerTail80_2, scan, if_neg (by decide), innerCandidates80_2_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_3
    (if 3 ≤ innerCandidates80_2.length then search edge 3 [431, 929] innerCandidates80_2 1360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_3

def innerTail80_1 : List Nat := 857 :: innerTail80_2
def innerCandidates80_1 : List Nat := [6329]
set_option maxRecDepth 4096 in
theorem innerCandidates80_1_checked : candidates edge 4 431 857 innerTail80_2 best.bound = innerCandidates80_1 := by
  unfold candidates
  have row : edge 857 = PrimePairCache.smallRow857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_1 : scan edge (search edge 3) 4 [431] 431 innerTail80_1 best = best := by
  rw [innerTail80_1, scan, if_neg (by decide), innerCandidates80_1_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_2
    (if 3 ≤ innerCandidates80_1.length then search edge 3 [431, 857] innerCandidates80_1 1288 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_2

def innerTail80_0 : List Nat := 479 :: innerTail80_1
def innerCandidates80_0 : List Nat := [7517]
set_option maxRecDepth 4096 in
theorem innerCandidates80_0_checked : candidates edge 4 431 479 innerTail80_1 best.bound = innerCandidates80_0 := by
  unfold candidates
  have row : edge 479 = PrimePairCache.smallRow479.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan80_0 : scan edge (search edge 3) 4 [431] 431 innerTail80_0 best = best := by
  rw [innerTail80_0, scan, if_neg (by decide), innerCandidates80_0_checked]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_1
    (if 3 ≤ innerCandidates80_0.length then search edge 3 [431, 479] innerCandidates80_0 910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan80_1

theorem rootBranch80 : search edge 4 [431] rootCandidates80 431 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [431] 431 innerTail80_0 best = best
  exact innerScan80_0

def rootCandidates79 : List Nat := [433, 607, 643, 661, 709, 1321, 1579, 1657, 2113, 2287, 3217, 4519, 4933, 5347, 5641, 5683, 5749, 6073, 6211, 6871, 7159, 7237, 7459, 7669, 8287]
set_option maxRecDepth 4096 in
theorem rootCandidates79_checked : candidates edge 5 0 421 tail80 best.bound = rootCandidates79 := by
  unfold candidates
  have row : edge 421 = PrimePairCache.smallRow421.contains := by funext q; rfl
  rw [row]
  decide

def innerTail79_25 : List Nat := []
theorem innerScan79_25 : scan edge (search edge 3) 4 [421] 421 innerTail79_25 best = best := rfl

def innerTail79_24 : List Nat := 8287 :: innerTail79_25
def innerCandidates79_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates79_24_checked : candidates edge 4 421 8287 innerTail79_25 best.bound = innerCandidates79_24 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_24 : scan edge (search edge 3) 4 [421] 421 innerTail79_24 best = best := by
  rw [innerTail79_24, scan, if_neg (by decide), innerCandidates79_24_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_25
    (if 3 ≤ innerCandidates79_24.length then search edge 3 [421, 8287] innerCandidates79_24 8708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_25

def innerTail79_23 : List Nat := 7669 :: innerTail79_24
def innerCandidates79_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates79_23_checked : candidates edge 4 421 7669 innerTail79_24 best.bound = innerCandidates79_23 := by
  unfold candidates
  have row : edge 7669 = SearchData.row7669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_23 : scan edge (search edge 3) 4 [421] 421 innerTail79_23 best = best := by
  rw [innerTail79_23, scan, if_neg (by decide), innerCandidates79_23_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_24
    (if 3 ≤ innerCandidates79_23.length then search edge 3 [421, 7669] innerCandidates79_23 8090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_24

def innerTail79_22 : List Nat := 7459 :: innerTail79_23
def innerCandidates79_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates79_22_checked : candidates edge 4 421 7459 innerTail79_23 best.bound = innerCandidates79_22 := by
  unfold candidates
  have row : edge 7459 = SearchData.row7459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_22 : scan edge (search edge 3) 4 [421] 421 innerTail79_22 best = best := by
  rw [innerTail79_22, scan, if_neg (by decide), innerCandidates79_22_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_23
    (if 3 ≤ innerCandidates79_22.length then search edge 3 [421, 7459] innerCandidates79_22 7880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_23

def innerTail79_21 : List Nat := 7237 :: innerTail79_22
def innerCandidates79_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates79_21_checked : candidates edge 4 421 7237 innerTail79_22 best.bound = innerCandidates79_21 := by
  unfold candidates
  have row : edge 7237 = SearchData.row7237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_21 : scan edge (search edge 3) 4 [421] 421 innerTail79_21 best = best := by
  rw [innerTail79_21, scan, if_neg (by decide), innerCandidates79_21_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_22
    (if 3 ≤ innerCandidates79_21.length then search edge 3 [421, 7237] innerCandidates79_21 7658 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_22

def innerTail79_20 : List Nat := 7159 :: innerTail79_21
def innerCandidates79_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates79_20_checked : candidates edge 4 421 7159 innerTail79_21 best.bound = innerCandidates79_20 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_20 : scan edge (search edge 3) 4 [421] 421 innerTail79_20 best = best := by
  rw [innerTail79_20, scan, if_neg (by decide), innerCandidates79_20_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_21
    (if 3 ≤ innerCandidates79_20.length then search edge 3 [421, 7159] innerCandidates79_20 7580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_21

def innerTail79_19 : List Nat := 6871 :: innerTail79_20
def innerCandidates79_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates79_19_checked : candidates edge 4 421 6871 innerTail79_20 best.bound = innerCandidates79_19 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_19 : scan edge (search edge 3) 4 [421] 421 innerTail79_19 best = best := by
  rw [innerTail79_19, scan, if_neg (by decide), innerCandidates79_19_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_20
    (if 3 ≤ innerCandidates79_19.length then search edge 3 [421, 6871] innerCandidates79_19 7292 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_20

def innerTail79_18 : List Nat := 6211 :: innerTail79_19
def innerCandidates79_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates79_18_checked : candidates edge 4 421 6211 innerTail79_19 best.bound = innerCandidates79_18 := by
  unfold candidates
  have row : edge 6211 = SearchData.row6211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_18 : scan edge (search edge 3) 4 [421] 421 innerTail79_18 best = best := by
  rw [innerTail79_18, scan, if_neg (by decide), innerCandidates79_18_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_19
    (if 3 ≤ innerCandidates79_18.length then search edge 3 [421, 6211] innerCandidates79_18 6632 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_19

def innerTail79_17 : List Nat := 6073 :: innerTail79_18
def innerCandidates79_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates79_17_checked : candidates edge 4 421 6073 innerTail79_18 best.bound = innerCandidates79_17 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_17 : scan edge (search edge 3) 4 [421] 421 innerTail79_17 best = best := by
  rw [innerTail79_17, scan, if_neg (by decide), innerCandidates79_17_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_18
    (if 3 ≤ innerCandidates79_17.length then search edge 3 [421, 6073] innerCandidates79_17 6494 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_18

def innerTail79_16 : List Nat := 5749 :: innerTail79_17
def innerCandidates79_16 : List Nat := [6211]
set_option maxRecDepth 4096 in
theorem innerCandidates79_16_checked : candidates edge 4 421 5749 innerTail79_17 best.bound = innerCandidates79_16 := by
  unfold candidates
  have row : edge 5749 = SearchData.row5749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_16 : scan edge (search edge 3) 4 [421] 421 innerTail79_16 best = best := by
  rw [innerTail79_16, scan, if_neg (by decide), innerCandidates79_16_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_17
    (if 3 ≤ innerCandidates79_16.length then search edge 3 [421, 5749] innerCandidates79_16 6170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_17

def innerTail79_15 : List Nat := 5683 :: innerTail79_16
def innerCandidates79_15 : List Nat := [7159]
set_option maxRecDepth 4096 in
theorem innerCandidates79_15_checked : candidates edge 4 421 5683 innerTail79_16 best.bound = innerCandidates79_15 := by
  unfold candidates
  have row : edge 5683 = SearchData.row5683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_15 : scan edge (search edge 3) 4 [421] 421 innerTail79_15 best = best := by
  rw [innerTail79_15, scan, if_neg (by decide), innerCandidates79_15_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_16
    (if 3 ≤ innerCandidates79_15.length then search edge 3 [421, 5683] innerCandidates79_15 6104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_16

def innerTail79_14 : List Nat := 5641 :: innerTail79_15
def innerCandidates79_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates79_14_checked : candidates edge 4 421 5641 innerTail79_15 best.bound = innerCandidates79_14 := by
  unfold candidates
  have row : edge 5641 = SearchData.row5641.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_14 : scan edge (search edge 3) 4 [421] 421 innerTail79_14 best = best := by
  rw [innerTail79_14, scan, if_neg (by decide), innerCandidates79_14_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_15
    (if 3 ≤ innerCandidates79_14.length then search edge 3 [421, 5641] innerCandidates79_14 6062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_15

def innerTail79_13 : List Nat := 5347 :: innerTail79_14
def innerCandidates79_13 : List Nat := [8287]
set_option maxRecDepth 4096 in
theorem innerCandidates79_13_checked : candidates edge 4 421 5347 innerTail79_14 best.bound = innerCandidates79_13 := by
  unfold candidates
  have row : edge 5347 = SearchData.row5347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_13 : scan edge (search edge 3) 4 [421] 421 innerTail79_13 best = best := by
  rw [innerTail79_13, scan, if_neg (by decide), innerCandidates79_13_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_14
    (if 3 ≤ innerCandidates79_13.length then search edge 3 [421, 5347] innerCandidates79_13 5768 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_14

def innerTail79_12 : List Nat := 4933 :: innerTail79_13
def innerCandidates79_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates79_12_checked : candidates edge 4 421 4933 innerTail79_13 best.bound = innerCandidates79_12 := by
  unfold candidates
  have row : edge 4933 = SearchData.row4933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_12 : scan edge (search edge 3) 4 [421] 421 innerTail79_12 best = best := by
  rw [innerTail79_12, scan, if_neg (by decide), innerCandidates79_12_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_13
    (if 3 ≤ innerCandidates79_12.length then search edge 3 [421, 4933] innerCandidates79_12 5354 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_13

def innerTail79_11 : List Nat := 4519 :: innerTail79_12
def innerCandidates79_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates79_11_checked : candidates edge 4 421 4519 innerTail79_12 best.bound = innerCandidates79_11 := by
  unfold candidates
  have row : edge 4519 = SearchData.row4519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_11 : scan edge (search edge 3) 4 [421] 421 innerTail79_11 best = best := by
  rw [innerTail79_11, scan, if_neg (by decide), innerCandidates79_11_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_12
    (if 3 ≤ innerCandidates79_11.length then search edge 3 [421, 4519] innerCandidates79_11 4940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_12

def innerTail79_10 : List Nat := 3217 :: innerTail79_11
def innerCandidates79_10 : List Nat := [6073, 6211]
set_option maxRecDepth 4096 in
theorem innerCandidates79_10_checked : candidates edge 4 421 3217 innerTail79_11 best.bound = innerCandidates79_10 := by
  unfold candidates
  have row : edge 3217 = SearchData.row3217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_10 : scan edge (search edge 3) 4 [421] 421 innerTail79_10 best = best := by
  rw [innerTail79_10, scan, if_neg (by decide), innerCandidates79_10_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_11
    (if 3 ≤ innerCandidates79_10.length then search edge 3 [421, 3217] innerCandidates79_10 3638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_11

def innerTail79_9 : List Nat := 2287 :: innerTail79_10
def innerCandidates79_9 : List Nat := [4933]
set_option maxRecDepth 4096 in
theorem innerCandidates79_9_checked : candidates edge 4 421 2287 innerTail79_10 best.bound = innerCandidates79_9 := by
  unfold candidates
  have row : edge 2287 = PrimePairCache.smallRow2287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_9 : scan edge (search edge 3) 4 [421] 421 innerTail79_9 best = best := by
  rw [innerTail79_9, scan, if_neg (by decide), innerCandidates79_9_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_10
    (if 3 ≤ innerCandidates79_9.length then search edge 3 [421, 2287] innerCandidates79_9 2708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_10

def innerTail79_8 : List Nat := 2113 :: innerTail79_9
def innerCandidates79_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates79_8_checked : candidates edge 4 421 2113 innerTail79_9 best.bound = innerCandidates79_8 := by
  unfold candidates
  have row : edge 2113 = PrimePairCache.smallRow2113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_8 : scan edge (search edge 3) 4 [421] 421 innerTail79_8 best = best := by
  rw [innerTail79_8, scan, if_neg (by decide), innerCandidates79_8_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_9
    (if 3 ≤ innerCandidates79_8.length then search edge 3 [421, 2113] innerCandidates79_8 2534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_9

def innerTail79_7 : List Nat := 1657 :: innerTail79_8
def innerCandidates79_7 : List Nat := [7237]
set_option maxRecDepth 4096 in
theorem innerCandidates79_7_checked : candidates edge 4 421 1657 innerTail79_8 best.bound = innerCandidates79_7 := by
  unfold candidates
  have row : edge 1657 = PrimePairCache.smallRow1657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_7 : scan edge (search edge 3) 4 [421] 421 innerTail79_7 best = best := by
  rw [innerTail79_7, scan, if_neg (by decide), innerCandidates79_7_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_8
    (if 3 ≤ innerCandidates79_7.length then search edge 3 [421, 1657] innerCandidates79_7 2078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_8

def innerTail79_6 : List Nat := 1579 :: innerTail79_7
def innerCandidates79_6 : List Nat := [6073]
set_option maxRecDepth 4096 in
theorem innerCandidates79_6_checked : candidates edge 4 421 1579 innerTail79_7 best.bound = innerCandidates79_6 := by
  unfold candidates
  have row : edge 1579 = PrimePairCache.smallRow1579.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_6 : scan edge (search edge 3) 4 [421] 421 innerTail79_6 best = best := by
  rw [innerTail79_6, scan, if_neg (by decide), innerCandidates79_6_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_7
    (if 3 ≤ innerCandidates79_6.length then search edge 3 [421, 1579] innerCandidates79_6 2000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_7

def innerTail79_5 : List Nat := 1321 :: innerTail79_6
def innerCandidates79_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates79_5_checked : candidates edge 4 421 1321 innerTail79_6 best.bound = innerCandidates79_5 := by
  unfold candidates
  have row : edge 1321 = PrimePairCache.smallRow1321.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_5 : scan edge (search edge 3) 4 [421] 421 innerTail79_5 best = best := by
  rw [innerTail79_5, scan, if_neg (by decide), innerCandidates79_5_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_6
    (if 3 ≤ innerCandidates79_5.length then search edge 3 [421, 1321] innerCandidates79_5 1742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_6

def innerTail79_4 : List Nat := 709 :: innerTail79_5
def innerCandidates79_4 : List Nat := [4519, 7459]
set_option maxRecDepth 4096 in
theorem innerCandidates79_4_checked : candidates edge 4 421 709 innerTail79_5 best.bound = innerCandidates79_4 := by
  unfold candidates
  have row : edge 709 = PrimePairCache.smallRow709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_4 : scan edge (search edge 3) 4 [421] 421 innerTail79_4 best = best := by
  rw [innerTail79_4, scan, if_neg (by decide), innerCandidates79_4_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_5
    (if 3 ≤ innerCandidates79_4.length then search edge 3 [421, 709] innerCandidates79_4 1130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_5

def innerTail79_3 : List Nat := 661 :: innerTail79_4
def innerCandidates79_3 : List Nat := [5641, 6871, 7159]
set_option maxRecDepth 4096 in
theorem innerCandidates79_3_checked : candidates edge 4 421 661 innerTail79_4 best.bound = innerCandidates79_3 := by
  unfold candidates
  have row : edge 661 = PrimePairCache.smallRow661.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild79_3 : search edge 3 [421, 661] innerCandidates79_3 1082 best = best := by decide

theorem innerScan79_3 : scan edge (search edge 3) 4 [421] 421 innerTail79_3 best = best := by
  rw [innerTail79_3, scan, if_neg (by decide), innerCandidates79_3_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_4
    (if 3 ≤ innerCandidates79_3.length then search edge 3 [421, 661] innerCandidates79_3 1082 best else best) = best
  rw [if_pos (by decide), innerChild79_3]
  exact innerScan79_4

def innerTail79_2 : List Nat := 643 :: innerTail79_3
def innerCandidates79_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates79_2_checked : candidates edge 4 421 643 innerTail79_3 best.bound = innerCandidates79_2 := by
  unfold candidates
  have row : edge 643 = PrimePairCache.smallRow643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_2 : scan edge (search edge 3) 4 [421] 421 innerTail79_2 best = best := by
  rw [innerTail79_2, scan, if_neg (by decide), innerCandidates79_2_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_3
    (if 3 ≤ innerCandidates79_2.length then search edge 3 [421, 643] innerCandidates79_2 1064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_3

def innerTail79_1 : List Nat := 607 :: innerTail79_2
def innerCandidates79_1 : List Nat := [8287]
set_option maxRecDepth 4096 in
theorem innerCandidates79_1_checked : candidates edge 4 421 607 innerTail79_2 best.bound = innerCandidates79_1 := by
  unfold candidates
  have row : edge 607 = PrimePairCache.smallRow607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_1 : scan edge (search edge 3) 4 [421] 421 innerTail79_1 best = best := by
  rw [innerTail79_1, scan, if_neg (by decide), innerCandidates79_1_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_2
    (if 3 ≤ innerCandidates79_1.length then search edge 3 [421, 607] innerCandidates79_1 1028 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_2

def innerTail79_0 : List Nat := 433 :: innerTail79_1
def innerCandidates79_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates79_0_checked : candidates edge 4 421 433 innerTail79_1 best.bound = innerCandidates79_0 := by
  unfold candidates
  have row : edge 433 = PrimePairCache.smallRow433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan79_0 : scan edge (search edge 3) 4 [421] 421 innerTail79_0 best = best := by
  rw [innerTail79_0, scan, if_neg (by decide), innerCandidates79_0_checked]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_1
    (if 3 ≤ innerCandidates79_0.length then search edge 3 [421, 433] innerCandidates79_0 854 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan79_1

theorem rootBranch79 : search edge 4 [421] rootCandidates79 421 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [421] 421 innerTail79_0 best = best
  exact innerScan79_0

end ProjectEulerSolutions.PrimePairUpper.Search
