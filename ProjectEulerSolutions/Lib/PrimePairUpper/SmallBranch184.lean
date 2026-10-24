import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates139 : List Nat := [827, 857, 971, 1109, 1163, 1433, 1607, 1949, 2549, 2591, 2969, 3581, 4691, 5783, 5843, 5849, 6317, 6521, 6737, 7523]
set_option maxRecDepth 4096 in
theorem rootCandidates139_checked : candidates edge 5 0 821 tail140 best.bound = rootCandidates139 := by
  unfold candidates
  have row : edge 821 = PrimePairCache.smallRow821.contains := by funext q; rfl
  rw [row]
  decide

def innerTail139_20 : List Nat := []
theorem innerScan139_20 : scan edge (search edge 3) 4 [821] 821 innerTail139_20 best = best := rfl

def innerTail139_19 : List Nat := 7523 :: innerTail139_20
def innerCandidates139_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates139_19_checked : candidates edge 4 821 7523 innerTail139_20 best.bound = innerCandidates139_19 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan139_19 : scan edge (search edge 3) 4 [821] 821 innerTail139_19 best = best := by
  rw [innerTail139_19, scan, if_neg (by decide), innerCandidates139_19_checked]
  change scan edge (search edge 3) 4 [821] 821 innerTail139_20
    (if 3 ≤ innerCandidates139_19.length then search edge 3 [821, 7523] innerCandidates139_19 8344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan139_20

def innerTail139_18 : List Nat := 6737 :: innerTail139_19
def innerCandidates139_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates139_18_checked : candidates edge 4 821 6737 innerTail139_19 best.bound = innerCandidates139_18 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan139_18 : scan edge (search edge 3) 4 [821] 821 innerTail139_18 best = best := by
  rw [innerTail139_18, scan, if_neg (by decide), innerCandidates139_18_checked]
  change scan edge (search edge 3) 4 [821] 821 innerTail139_19
    (if 3 ≤ innerCandidates139_18.length then search edge 3 [821, 6737] innerCandidates139_18 7558 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan139_19

def innerTail139_17 : List Nat := 6521 :: innerTail139_18
def innerCandidates139_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates139_17_checked : candidates edge 4 821 6521 innerTail139_18 best.bound = innerCandidates139_17 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan139_17 : scan edge (search edge 3) 4 [821] 821 innerTail139_17 best = best := by
  rw [innerTail139_17, scan, if_neg (by decide), innerCandidates139_17_checked]
  change scan edge (search edge 3) 4 [821] 821 innerTail139_18
    (if 3 ≤ innerCandidates139_17.length then search edge 3 [821, 6521] innerCandidates139_17 7342 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan139_18

def innerTail139_16 : List Nat := 6317 :: innerTail139_17
def innerCandidates139_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates139_16_checked : candidates edge 4 821 6317 innerTail139_17 best.bound = innerCandidates139_16 := by
  unfold candidates
  have row : edge 6317 = SearchData.row6317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan139_16 : scan edge (search edge 3) 4 [821] 821 innerTail139_16 best = best := by
  rw [innerTail139_16, scan, if_neg (by decide), innerCandidates139_16_checked]
  change scan edge (search edge 3) 4 [821] 821 innerTail139_17
    (if 3 ≤ innerCandidates139_16.length then search edge 3 [821, 6317] innerCandidates139_16 7138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan139_17

def innerTail139_15 : List Nat := 5849 :: innerTail139_16
def innerCandidates139_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates139_15_checked : candidates edge 4 821 5849 innerTail139_16 best.bound = innerCandidates139_15 := by
  unfold candidates
  have row : edge 5849 = SearchData.row5849.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan139_15 : scan edge (search edge 3) 4 [821] 821 innerTail139_15 best = best := by
  rw [innerTail139_15, scan, if_neg (by decide), innerCandidates139_15_checked]
  change scan edge (search edge 3) 4 [821] 821 innerTail139_16
    (if 3 ≤ innerCandidates139_15.length then search edge 3 [821, 5849] innerCandidates139_15 6670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan139_16

def innerTail139_14 : List Nat := 5843 :: innerTail139_15
def innerCandidates139_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates139_14_checked : candidates edge 4 821 5843 innerTail139_15 best.bound = innerCandidates139_14 := by
  unfold candidates
  have row : edge 5843 = SearchData.row5843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan139_14 : scan edge (search edge 3) 4 [821] 821 innerTail139_14 best = best := by
  rw [innerTail139_14, scan, if_neg (by decide), innerCandidates139_14_checked]
  change scan edge (search edge 3) 4 [821] 821 innerTail139_15
    (if 3 ≤ innerCandidates139_14.length then search edge 3 [821, 5843] innerCandidates139_14 6664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan139_15

def innerTail139_13 : List Nat := 5783 :: innerTail139_14
def innerCandidates139_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates139_13_checked : candidates edge 4 821 5783 innerTail139_14 best.bound = innerCandidates139_13 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan139_13 : scan edge (search edge 3) 4 [821] 821 innerTail139_13 best = best := by
  rw [innerTail139_13, scan, if_neg (by decide), innerCandidates139_13_checked]
  change scan edge (search edge 3) 4 [821] 821 innerTail139_14
    (if 3 ≤ innerCandidates139_13.length then search edge 3 [821, 5783] innerCandidates139_13 6604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan139_14

def innerTail139_12 : List Nat := 4691 :: innerTail139_13
def innerCandidates139_12 : List Nat := [6521]
set_option maxRecDepth 4096 in
theorem innerCandidates139_12_checked : candidates edge 4 821 4691 innerTail139_13 best.bound = innerCandidates139_12 := by
  unfold candidates
  have row : edge 4691 = SearchData.row4691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan139_12 : scan edge (search edge 3) 4 [821] 821 innerTail139_12 best = best := by
  rw [innerTail139_12, scan, if_neg (by decide), innerCandidates139_12_checked]
  change scan edge (search edge 3) 4 [821] 821 innerTail139_13
    (if 3 ≤ innerCandidates139_12.length then search edge 3 [821, 4691] innerCandidates139_12 5512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan139_13

def innerTail139_11 : List Nat := 3581 :: innerTail139_12
def innerCandidates139_11 : List Nat := [5849]
set_option maxRecDepth 4096 in
theorem innerCandidates139_11_checked : candidates edge 4 821 3581 innerTail139_12 best.bound = innerCandidates139_11 := by
  unfold candidates
  have row : edge 3581 = SearchData.row3581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan139_11 : scan edge (search edge 3) 4 [821] 821 innerTail139_11 best = best := by
  rw [innerTail139_11, scan, if_neg (by decide), innerCandidates139_11_checked]
  change scan edge (search edge 3) 4 [821] 821 innerTail139_12
    (if 3 ≤ innerCandidates139_11.length then search edge 3 [821, 3581] innerCandidates139_11 4402 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan139_12

def innerTail139_10 : List Nat := 2969 :: innerTail139_11
def innerCandidates139_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates139_10_checked : candidates edge 4 821 2969 innerTail139_11 best.bound = innerCandidates139_10 := by
  unfold candidates
  have row : edge 2969 = SearchData.row2969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan139_10 : scan edge (search edge 3) 4 [821] 821 innerTail139_10 best = best := by
  rw [innerTail139_10, scan, if_neg (by decide), innerCandidates139_10_checked]
  change scan edge (search edge 3) 4 [821] 821 innerTail139_11
    (if 3 ≤ innerCandidates139_10.length then search edge 3 [821, 2969] innerCandidates139_10 3790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan139_11

def innerTail139_9 : List Nat := 2591 :: innerTail139_10
def innerCandidates139_9 : List Nat := [5849]
set_option maxRecDepth 4096 in
theorem innerCandidates139_9_checked : candidates edge 4 821 2591 innerTail139_10 best.bound = innerCandidates139_9 := by
  unfold candidates
  have row : edge 2591 = PrimePairCache.smallRow2591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan139_9 : scan edge (search edge 3) 4 [821] 821 innerTail139_9 best = best := by
  rw [innerTail139_9, scan, if_neg (by decide), innerCandidates139_9_checked]
  change scan edge (search edge 3) 4 [821] 821 innerTail139_10
    (if 3 ≤ innerCandidates139_9.length then search edge 3 [821, 2591] innerCandidates139_9 3412 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan139_10

def innerTail139_8 : List Nat := 2549 :: innerTail139_9
def innerCandidates139_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates139_8_checked : candidates edge 4 821 2549 innerTail139_9 best.bound = innerCandidates139_8 := by
  unfold candidates
  have row : edge 2549 = PrimePairCache.smallRow2549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan139_8 : scan edge (search edge 3) 4 [821] 821 innerTail139_8 best = best := by
  rw [innerTail139_8, scan, if_neg (by decide), innerCandidates139_8_checked]
  change scan edge (search edge 3) 4 [821] 821 innerTail139_9
    (if 3 ≤ innerCandidates139_8.length then search edge 3 [821, 2549] innerCandidates139_8 3370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan139_9

def innerTail139_7 : List Nat := 1949 :: innerTail139_8
def innerCandidates139_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates139_7_checked : candidates edge 4 821 1949 innerTail139_8 best.bound = innerCandidates139_7 := by
  unfold candidates
  have row : edge 1949 = PrimePairCache.smallRow1949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan139_7 : scan edge (search edge 3) 4 [821] 821 innerTail139_7 best = best := by
  rw [innerTail139_7, scan, if_neg (by decide), innerCandidates139_7_checked]
  change scan edge (search edge 3) 4 [821] 821 innerTail139_8
    (if 3 ≤ innerCandidates139_7.length then search edge 3 [821, 1949] innerCandidates139_7 2770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan139_8

def innerTail139_6 : List Nat := 1607 :: innerTail139_7
def innerCandidates139_6 : List Nat := [2549, 5849]
set_option maxRecDepth 4096 in
theorem innerCandidates139_6_checked : candidates edge 4 821 1607 innerTail139_7 best.bound = innerCandidates139_6 := by
  unfold candidates
  have row : edge 1607 = PrimePairCache.smallRow1607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan139_6 : scan edge (search edge 3) 4 [821] 821 innerTail139_6 best = best := by
  rw [innerTail139_6, scan, if_neg (by decide), innerCandidates139_6_checked]
  change scan edge (search edge 3) 4 [821] 821 innerTail139_7
    (if 3 ≤ innerCandidates139_6.length then search edge 3 [821, 1607] innerCandidates139_6 2428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan139_7

def innerTail139_5 : List Nat := 1433 :: innerTail139_6
def innerCandidates139_5 : List Nat := [6737]
set_option maxRecDepth 4096 in
theorem innerCandidates139_5_checked : candidates edge 4 821 1433 innerTail139_6 best.bound = innerCandidates139_5 := by
  unfold candidates
  have row : edge 1433 = PrimePairCache.smallRow1433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan139_5 : scan edge (search edge 3) 4 [821] 821 innerTail139_5 best = best := by
  rw [innerTail139_5, scan, if_neg (by decide), innerCandidates139_5_checked]
  change scan edge (search edge 3) 4 [821] 821 innerTail139_6
    (if 3 ≤ innerCandidates139_5.length then search edge 3 [821, 1433] innerCandidates139_5 2254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan139_6

def innerTail139_4 : List Nat := 1163 :: innerTail139_5
def innerCandidates139_4 : List Nat := [5849]
set_option maxRecDepth 4096 in
theorem innerCandidates139_4_checked : candidates edge 4 821 1163 innerTail139_5 best.bound = innerCandidates139_4 := by
  unfold candidates
  have row : edge 1163 = PrimePairCache.smallRow1163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan139_4 : scan edge (search edge 3) 4 [821] 821 innerTail139_4 best = best := by
  rw [innerTail139_4, scan, if_neg (by decide), innerCandidates139_4_checked]
  change scan edge (search edge 3) 4 [821] 821 innerTail139_5
    (if 3 ≤ innerCandidates139_4.length then search edge 3 [821, 1163] innerCandidates139_4 1984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan139_5

def innerTail139_3 : List Nat := 1109 :: innerTail139_4
def innerCandidates139_3 : List Nat := [6521]
set_option maxRecDepth 4096 in
theorem innerCandidates139_3_checked : candidates edge 4 821 1109 innerTail139_4 best.bound = innerCandidates139_3 := by
  unfold candidates
  have row : edge 1109 = PrimePairCache.smallRow1109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan139_3 : scan edge (search edge 3) 4 [821] 821 innerTail139_3 best = best := by
  rw [innerTail139_3, scan, if_neg (by decide), innerCandidates139_3_checked]
  change scan edge (search edge 3) 4 [821] 821 innerTail139_4
    (if 3 ≤ innerCandidates139_3.length then search edge 3 [821, 1109] innerCandidates139_3 1930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan139_4

def innerTail139_2 : List Nat := 971 :: innerTail139_3
def innerCandidates139_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates139_2_checked : candidates edge 4 821 971 innerTail139_3 best.bound = innerCandidates139_2 := by
  unfold candidates
  have row : edge 971 = PrimePairCache.smallRow971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan139_2 : scan edge (search edge 3) 4 [821] 821 innerTail139_2 best = best := by
  rw [innerTail139_2, scan, if_neg (by decide), innerCandidates139_2_checked]
  change scan edge (search edge 3) 4 [821] 821 innerTail139_3
    (if 3 ≤ innerCandidates139_2.length then search edge 3 [821, 971] innerCandidates139_2 1792 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan139_3

def innerTail139_1 : List Nat := 857 :: innerTail139_2
def innerCandidates139_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates139_1_checked : candidates edge 4 821 857 innerTail139_2 best.bound = innerCandidates139_1 := by
  unfold candidates
  have row : edge 857 = PrimePairCache.smallRow857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan139_1 : scan edge (search edge 3) 4 [821] 821 innerTail139_1 best = best := by
  rw [innerTail139_1, scan, if_neg (by decide), innerCandidates139_1_checked]
  change scan edge (search edge 3) 4 [821] 821 innerTail139_2
    (if 3 ≤ innerCandidates139_1.length then search edge 3 [821, 857] innerCandidates139_1 1678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan139_2

def innerTail139_0 : List Nat := 827 :: innerTail139_1
def innerCandidates139_0 : List Nat := [857]
set_option maxRecDepth 4096 in
theorem innerCandidates139_0_checked : candidates edge 4 821 827 innerTail139_1 best.bound = innerCandidates139_0 := by
  unfold candidates
  have row : edge 827 = PrimePairCache.smallRow827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan139_0 : scan edge (search edge 3) 4 [821] 821 innerTail139_0 best = best := by
  rw [innerTail139_0, scan, if_neg (by decide), innerCandidates139_0_checked]
  change scan edge (search edge 3) 4 [821] 821 innerTail139_1
    (if 3 ≤ innerCandidates139_0.length then search edge 3 [821, 827] innerCandidates139_0 1648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan139_1

theorem rootBranch139 : search edge 4 [821] rootCandidates139 821 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [821] 821 innerTail139_0 best = best
  exact innerScan139_0

def rootCandidates138 : List Nat := [919, 991, 997, 1171, 1447, 1459, 1531, 1657, 1723, 2011, 3109, 3631, 3643, 3691, 3967, 4201, 4861, 5119, 5449, 5791, 6133, 6469, 6547, 6967, 7561, 8101, 8179, 8191, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates138_checked : candidates edge 5 0 811 tail139 best.bound = rootCandidates138 := by
  unfold candidates
  have row : edge 811 = PrimePairCache.smallRow811.contains := by funext q; rfl
  rw [row]
  decide

def innerTail138_29 : List Nat := []
theorem innerScan138_29 : scan edge (search edge 3) 4 [811] 811 innerTail138_29 best = best := rfl

def innerTail138_28 : List Nat := 8233 :: innerTail138_29
def innerCandidates138_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates138_28_checked : candidates edge 4 811 8233 innerTail138_29 best.bound = innerCandidates138_28 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_28 : scan edge (search edge 3) 4 [811] 811 innerTail138_28 best = best := by
  rw [innerTail138_28, scan, if_neg (by decide), innerCandidates138_28_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_29
    (if 3 ≤ innerCandidates138_28.length then search edge 3 [811, 8233] innerCandidates138_28 9044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_29

def innerTail138_27 : List Nat := 8191 :: innerTail138_28
def innerCandidates138_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates138_27_checked : candidates edge 4 811 8191 innerTail138_28 best.bound = innerCandidates138_27 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_27 : scan edge (search edge 3) 4 [811] 811 innerTail138_27 best = best := by
  rw [innerTail138_27, scan, if_neg (by decide), innerCandidates138_27_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_28
    (if 3 ≤ innerCandidates138_27.length then search edge 3 [811, 8191] innerCandidates138_27 9002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_28

def innerTail138_26 : List Nat := 8179 :: innerTail138_27
def innerCandidates138_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates138_26_checked : candidates edge 4 811 8179 innerTail138_27 best.bound = innerCandidates138_26 := by
  unfold candidates
  have row : edge 8179 = SearchData.row8179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_26 : scan edge (search edge 3) 4 [811] 811 innerTail138_26 best = best := by
  rw [innerTail138_26, scan, if_neg (by decide), innerCandidates138_26_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_27
    (if 3 ≤ innerCandidates138_26.length then search edge 3 [811, 8179] innerCandidates138_26 8990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_27

def innerTail138_25 : List Nat := 8101 :: innerTail138_26
def innerCandidates138_25 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates138_25_checked : candidates edge 4 811 8101 innerTail138_26 best.bound = innerCandidates138_25 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_25 : scan edge (search edge 3) 4 [811] 811 innerTail138_25 best = best := by
  rw [innerTail138_25, scan, if_neg (by decide), innerCandidates138_25_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_26
    (if 3 ≤ innerCandidates138_25.length then search edge 3 [811, 8101] innerCandidates138_25 8912 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_26

def innerTail138_24 : List Nat := 7561 :: innerTail138_25
def innerCandidates138_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates138_24_checked : candidates edge 4 811 7561 innerTail138_25 best.bound = innerCandidates138_24 := by
  unfold candidates
  have row : edge 7561 = SearchData.row7561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_24 : scan edge (search edge 3) 4 [811] 811 innerTail138_24 best = best := by
  rw [innerTail138_24, scan, if_neg (by decide), innerCandidates138_24_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_25
    (if 3 ≤ innerCandidates138_24.length then search edge 3 [811, 7561] innerCandidates138_24 8372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_25

def innerTail138_23 : List Nat := 6967 :: innerTail138_24
def innerCandidates138_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates138_23_checked : candidates edge 4 811 6967 innerTail138_24 best.bound = innerCandidates138_23 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_23 : scan edge (search edge 3) 4 [811] 811 innerTail138_23 best = best := by
  rw [innerTail138_23, scan, if_neg (by decide), innerCandidates138_23_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_24
    (if 3 ≤ innerCandidates138_23.length then search edge 3 [811, 6967] innerCandidates138_23 7778 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_24

def innerTail138_22 : List Nat := 6547 :: innerTail138_23
def innerCandidates138_22 : List Nat := [6967, 7561]
set_option maxRecDepth 4096 in
theorem innerCandidates138_22_checked : candidates edge 4 811 6547 innerTail138_23 best.bound = innerCandidates138_22 := by
  unfold candidates
  have row : edge 6547 = SearchData.row6547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_22 : scan edge (search edge 3) 4 [811] 811 innerTail138_22 best = best := by
  rw [innerTail138_22, scan, if_neg (by decide), innerCandidates138_22_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_23
    (if 3 ≤ innerCandidates138_22.length then search edge 3 [811, 6547] innerCandidates138_22 7358 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_23

def innerTail138_21 : List Nat := 6469 :: innerTail138_22
def innerCandidates138_21 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates138_21_checked : candidates edge 4 811 6469 innerTail138_22 best.bound = innerCandidates138_21 := by
  unfold candidates
  have row : edge 6469 = SearchData.row6469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_21 : scan edge (search edge 3) 4 [811] 811 innerTail138_21 best = best := by
  rw [innerTail138_21, scan, if_neg (by decide), innerCandidates138_21_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_22
    (if 3 ≤ innerCandidates138_21.length then search edge 3 [811, 6469] innerCandidates138_21 7280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_22

def innerTail138_20 : List Nat := 6133 :: innerTail138_21
def innerCandidates138_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates138_20_checked : candidates edge 4 811 6133 innerTail138_21 best.bound = innerCandidates138_20 := by
  unfold candidates
  have row : edge 6133 = SearchData.row6133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_20 : scan edge (search edge 3) 4 [811] 811 innerTail138_20 best = best := by
  rw [innerTail138_20, scan, if_neg (by decide), innerCandidates138_20_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_21
    (if 3 ≤ innerCandidates138_20.length then search edge 3 [811, 6133] innerCandidates138_20 6944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_21

def innerTail138_19 : List Nat := 5791 :: innerTail138_20
def innerCandidates138_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates138_19_checked : candidates edge 4 811 5791 innerTail138_20 best.bound = innerCandidates138_19 := by
  unfold candidates
  have row : edge 5791 = SearchData.row5791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_19 : scan edge (search edge 3) 4 [811] 811 innerTail138_19 best = best := by
  rw [innerTail138_19, scan, if_neg (by decide), innerCandidates138_19_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_20
    (if 3 ≤ innerCandidates138_19.length then search edge 3 [811, 5791] innerCandidates138_19 6602 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_20

def innerTail138_18 : List Nat := 5449 :: innerTail138_19
def innerCandidates138_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates138_18_checked : candidates edge 4 811 5449 innerTail138_19 best.bound = innerCandidates138_18 := by
  unfold candidates
  have row : edge 5449 = SearchData.row5449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_18 : scan edge (search edge 3) 4 [811] 811 innerTail138_18 best = best := by
  rw [innerTail138_18, scan, if_neg (by decide), innerCandidates138_18_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_19
    (if 3 ≤ innerCandidates138_18.length then search edge 3 [811, 5449] innerCandidates138_18 6260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_19

def innerTail138_17 : List Nat := 5119 :: innerTail138_18
def innerCandidates138_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates138_17_checked : candidates edge 4 811 5119 innerTail138_18 best.bound = innerCandidates138_17 := by
  unfold candidates
  have row : edge 5119 = SearchData.row5119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_17 : scan edge (search edge 3) 4 [811] 811 innerTail138_17 best = best := by
  rw [innerTail138_17, scan, if_neg (by decide), innerCandidates138_17_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_18
    (if 3 ≤ innerCandidates138_17.length then search edge 3 [811, 5119] innerCandidates138_17 5930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_18

def innerTail138_16 : List Nat := 4861 :: innerTail138_17
def innerCandidates138_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates138_16_checked : candidates edge 4 811 4861 innerTail138_17 best.bound = innerCandidates138_16 := by
  unfold candidates
  have row : edge 4861 = SearchData.row4861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_16 : scan edge (search edge 3) 4 [811] 811 innerTail138_16 best = best := by
  rw [innerTail138_16, scan, if_neg (by decide), innerCandidates138_16_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_17
    (if 3 ≤ innerCandidates138_16.length then search edge 3 [811, 4861] innerCandidates138_16 5672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_17

def innerTail138_15 : List Nat := 4201 :: innerTail138_16
def innerCandidates138_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates138_15_checked : candidates edge 4 811 4201 innerTail138_16 best.bound = innerCandidates138_15 := by
  unfold candidates
  have row : edge 4201 = SearchData.row4201.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_15 : scan edge (search edge 3) 4 [811] 811 innerTail138_15 best = best := by
  rw [innerTail138_15, scan, if_neg (by decide), innerCandidates138_15_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_16
    (if 3 ≤ innerCandidates138_15.length then search edge 3 [811, 4201] innerCandidates138_15 5012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_16

def innerTail138_14 : List Nat := 3967 :: innerTail138_15
def innerCandidates138_14 : List Nat := [5791]
set_option maxRecDepth 4096 in
theorem innerCandidates138_14_checked : candidates edge 4 811 3967 innerTail138_15 best.bound = innerCandidates138_14 := by
  unfold candidates
  have row : edge 3967 = SearchData.row3967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_14 : scan edge (search edge 3) 4 [811] 811 innerTail138_14 best = best := by
  rw [innerTail138_14, scan, if_neg (by decide), innerCandidates138_14_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_15
    (if 3 ≤ innerCandidates138_14.length then search edge 3 [811, 3967] innerCandidates138_14 4778 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_15

def innerTail138_13 : List Nat := 3691 :: innerTail138_14
def innerCandidates138_13 : List Nat := [7561]
set_option maxRecDepth 4096 in
theorem innerCandidates138_13_checked : candidates edge 4 811 3691 innerTail138_14 best.bound = innerCandidates138_13 := by
  unfold candidates
  have row : edge 3691 = SearchData.row3691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_13 : scan edge (search edge 3) 4 [811] 811 innerTail138_13 best = best := by
  rw [innerTail138_13, scan, if_neg (by decide), innerCandidates138_13_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_14
    (if 3 ≤ innerCandidates138_13.length then search edge 3 [811, 3691] innerCandidates138_13 4502 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_14

def innerTail138_12 : List Nat := 3643 :: innerTail138_13
def innerCandidates138_12 : List Nat := [5449]
set_option maxRecDepth 4096 in
theorem innerCandidates138_12_checked : candidates edge 4 811 3643 innerTail138_13 best.bound = innerCandidates138_12 := by
  unfold candidates
  have row : edge 3643 = SearchData.row3643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_12 : scan edge (search edge 3) 4 [811] 811 innerTail138_12 best = best := by
  rw [innerTail138_12, scan, if_neg (by decide), innerCandidates138_12_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_13
    (if 3 ≤ innerCandidates138_12.length then search edge 3 [811, 3643] innerCandidates138_12 4454 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_13

def innerTail138_11 : List Nat := 3631 :: innerTail138_12
def innerCandidates138_11 : List Nat := [5119, 6547]
set_option maxRecDepth 4096 in
theorem innerCandidates138_11_checked : candidates edge 4 811 3631 innerTail138_12 best.bound = innerCandidates138_11 := by
  unfold candidates
  have row : edge 3631 = SearchData.row3631.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_11 : scan edge (search edge 3) 4 [811] 811 innerTail138_11 best = best := by
  rw [innerTail138_11, scan, if_neg (by decide), innerCandidates138_11_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_12
    (if 3 ≤ innerCandidates138_11.length then search edge 3 [811, 3631] innerCandidates138_11 4442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_12

def innerTail138_10 : List Nat := 3109 :: innerTail138_11
def innerCandidates138_10 : List Nat := [5791]
set_option maxRecDepth 4096 in
theorem innerCandidates138_10_checked : candidates edge 4 811 3109 innerTail138_11 best.bound = innerCandidates138_10 := by
  unfold candidates
  have row : edge 3109 = SearchData.row3109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_10 : scan edge (search edge 3) 4 [811] 811 innerTail138_10 best = best := by
  rw [innerTail138_10, scan, if_neg (by decide), innerCandidates138_10_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_11
    (if 3 ≤ innerCandidates138_10.length then search edge 3 [811, 3109] innerCandidates138_10 3920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_11

def innerTail138_9 : List Nat := 2011 :: innerTail138_10
def innerCandidates138_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates138_9_checked : candidates edge 4 811 2011 innerTail138_10 best.bound = innerCandidates138_9 := by
  unfold candidates
  have row : edge 2011 = PrimePairCache.smallRow2011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_9 : scan edge (search edge 3) 4 [811] 811 innerTail138_9 best = best := by
  rw [innerTail138_9, scan, if_neg (by decide), innerCandidates138_9_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_10
    (if 3 ≤ innerCandidates138_9.length then search edge 3 [811, 2011] innerCandidates138_9 2822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_10

def innerTail138_8 : List Nat := 1723 :: innerTail138_9
def innerCandidates138_8 : List Nat := [6547, 8233]
set_option maxRecDepth 4096 in
theorem innerCandidates138_8_checked : candidates edge 4 811 1723 innerTail138_9 best.bound = innerCandidates138_8 := by
  unfold candidates
  have row : edge 1723 = PrimePairCache.smallRow1723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_8 : scan edge (search edge 3) 4 [811] 811 innerTail138_8 best = best := by
  rw [innerTail138_8, scan, if_neg (by decide), innerCandidates138_8_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_9
    (if 3 ≤ innerCandidates138_8.length then search edge 3 [811, 1723] innerCandidates138_8 2534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_9

def innerTail138_7 : List Nat := 1657 :: innerTail138_8
def innerCandidates138_7 : List Nat := [3967]
set_option maxRecDepth 4096 in
theorem innerCandidates138_7_checked : candidates edge 4 811 1657 innerTail138_8 best.bound = innerCandidates138_7 := by
  unfold candidates
  have row : edge 1657 = PrimePairCache.smallRow1657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_7 : scan edge (search edge 3) 4 [811] 811 innerTail138_7 best = best := by
  rw [innerTail138_7, scan, if_neg (by decide), innerCandidates138_7_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_8
    (if 3 ≤ innerCandidates138_7.length then search edge 3 [811, 1657] innerCandidates138_7 2468 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_8

def innerTail138_6 : List Nat := 1531 :: innerTail138_7
def innerCandidates138_6 : List Nat := [5119]
set_option maxRecDepth 4096 in
theorem innerCandidates138_6_checked : candidates edge 4 811 1531 innerTail138_7 best.bound = innerCandidates138_6 := by
  unfold candidates
  have row : edge 1531 = PrimePairCache.smallRow1531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_6 : scan edge (search edge 3) 4 [811] 811 innerTail138_6 best = best := by
  rw [innerTail138_6, scan, if_neg (by decide), innerCandidates138_6_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_7
    (if 3 ≤ innerCandidates138_6.length then search edge 3 [811, 1531] innerCandidates138_6 2342 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_7

def innerTail138_5 : List Nat := 1459 :: innerTail138_6
def innerCandidates138_5 : List Nat := [1657]
set_option maxRecDepth 4096 in
theorem innerCandidates138_5_checked : candidates edge 4 811 1459 innerTail138_6 best.bound = innerCandidates138_5 := by
  unfold candidates
  have row : edge 1459 = PrimePairCache.smallRow1459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_5 : scan edge (search edge 3) 4 [811] 811 innerTail138_5 best = best := by
  rw [innerTail138_5, scan, if_neg (by decide), innerCandidates138_5_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_6
    (if 3 ≤ innerCandidates138_5.length then search edge 3 [811, 1459] innerCandidates138_5 2270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_6

def innerTail138_4 : List Nat := 1447 :: innerTail138_5
def innerCandidates138_4 : List Nat := [2011, 6967]
set_option maxRecDepth 4096 in
theorem innerCandidates138_4_checked : candidates edge 4 811 1447 innerTail138_5 best.bound = innerCandidates138_4 := by
  unfold candidates
  have row : edge 1447 = PrimePairCache.smallRow1447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_4 : scan edge (search edge 3) 4 [811] 811 innerTail138_4 best = best := by
  rw [innerTail138_4, scan, if_neg (by decide), innerCandidates138_4_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_5
    (if 3 ≤ innerCandidates138_4.length then search edge 3 [811, 1447] innerCandidates138_4 2258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_5

def innerTail138_3 : List Nat := 1171 :: innerTail138_4
def innerCandidates138_3 : List Nat := [2011]
set_option maxRecDepth 4096 in
theorem innerCandidates138_3_checked : candidates edge 4 811 1171 innerTail138_4 best.bound = innerCandidates138_3 := by
  unfold candidates
  have row : edge 1171 = PrimePairCache.smallRow1171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_3 : scan edge (search edge 3) 4 [811] 811 innerTail138_3 best = best := by
  rw [innerTail138_3, scan, if_neg (by decide), innerCandidates138_3_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_4
    (if 3 ≤ innerCandidates138_3.length then search edge 3 [811, 1171] innerCandidates138_3 1982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_4

def innerTail138_2 : List Nat := 997 :: innerTail138_3
def innerCandidates138_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates138_2_checked : candidates edge 4 811 997 innerTail138_3 best.bound = innerCandidates138_2 := by
  unfold candidates
  have row : edge 997 = PrimePairCache.smallRow997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_2 : scan edge (search edge 3) 4 [811] 811 innerTail138_2 best = best := by
  rw [innerTail138_2, scan, if_neg (by decide), innerCandidates138_2_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_3
    (if 3 ≤ innerCandidates138_2.length then search edge 3 [811, 997] innerCandidates138_2 1808 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_3

def innerTail138_1 : List Nat := 991 :: innerTail138_2
def innerCandidates138_1 : List Nat := [1171]
set_option maxRecDepth 4096 in
theorem innerCandidates138_1_checked : candidates edge 4 811 991 innerTail138_2 best.bound = innerCandidates138_1 := by
  unfold candidates
  have row : edge 991 = PrimePairCache.smallRow991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_1 : scan edge (search edge 3) 4 [811] 811 innerTail138_1 best = best := by
  rw [innerTail138_1, scan, if_neg (by decide), innerCandidates138_1_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_2
    (if 3 ≤ innerCandidates138_1.length then search edge 3 [811, 991] innerCandidates138_1 1802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_2

def innerTail138_0 : List Nat := 919 :: innerTail138_1
def innerCandidates138_0 : List Nat := [3643]
set_option maxRecDepth 4096 in
theorem innerCandidates138_0_checked : candidates edge 4 811 919 innerTail138_1 best.bound = innerCandidates138_0 := by
  unfold candidates
  have row : edge 919 = PrimePairCache.smallRow919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan138_0 : scan edge (search edge 3) 4 [811] 811 innerTail138_0 best = best := by
  rw [innerTail138_0, scan, if_neg (by decide), innerCandidates138_0_checked]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_1
    (if 3 ≤ innerCandidates138_0.length then search edge 3 [811, 919] innerCandidates138_0 1730 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan138_1

theorem rootBranch138 : search edge 4 [811] rootCandidates138 811 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [811] 811 innerTail138_0 best = best
  exact innerScan138_0

end ProjectEulerSolutions.PrimePairUpper.Search
