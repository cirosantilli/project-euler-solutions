import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates264 : List Nat := [1913, 2213, 2333, 2393, 2423, 2591, 2843, 3137, 3299, 3389, 3917, 4001, 4013, 4283, 4397, 4523, 6551, 6857, 6869, 7001, 7247, 7823, 7877, 8231, 8297]
set_option maxRecDepth 4096 in
theorem rootCandidates264_checked : candidates edge 5 0 1709 tail265 best.bound = rootCandidates264 := by
  unfold candidates
  have row : edge 1709 = PrimePairCache.smallRow1709.contains := by funext q; rfl
  rw [row]
  decide

def innerTail264_25 : List Nat := []
theorem innerScan264_25 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_25 best = best := rfl

def innerTail264_24 : List Nat := 8297 :: innerTail264_25
def innerCandidates264_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates264_24_checked : candidates edge 4 1709 8297 innerTail264_25 best.bound = innerCandidates264_24 := by
  unfold candidates
  have row : edge 8297 = SearchData.row8297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_24 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_24 best = best := by
  rw [innerTail264_24, scan, if_neg (by decide), innerCandidates264_24_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_25
    (if 3 ≤ innerCandidates264_24.length then search edge 3 [1709, 8297] innerCandidates264_24 10006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_25

def innerTail264_23 : List Nat := 8231 :: innerTail264_24
def innerCandidates264_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates264_23_checked : candidates edge 4 1709 8231 innerTail264_24 best.bound = innerCandidates264_23 := by
  unfold candidates
  have row : edge 8231 = SearchData.row8231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_23 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_23 best = best := by
  rw [innerTail264_23, scan, if_neg (by decide), innerCandidates264_23_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_24
    (if 3 ≤ innerCandidates264_23.length then search edge 3 [1709, 8231] innerCandidates264_23 9940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_24

def innerTail264_22 : List Nat := 7877 :: innerTail264_23
def innerCandidates264_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates264_22_checked : candidates edge 4 1709 7877 innerTail264_23 best.bound = innerCandidates264_22 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_22 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_22 best = best := by
  rw [innerTail264_22, scan, if_neg (by decide), innerCandidates264_22_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_23
    (if 3 ≤ innerCandidates264_22.length then search edge 3 [1709, 7877] innerCandidates264_22 9586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_23

def innerTail264_21 : List Nat := 7823 :: innerTail264_22
def innerCandidates264_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates264_21_checked : candidates edge 4 1709 7823 innerTail264_22 best.bound = innerCandidates264_21 := by
  unfold candidates
  have row : edge 7823 = SearchData.row7823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_21 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_21 best = best := by
  rw [innerTail264_21, scan, if_neg (by decide), innerCandidates264_21_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_22
    (if 3 ≤ innerCandidates264_21.length then search edge 3 [1709, 7823] innerCandidates264_21 9532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_22

def innerTail264_20 : List Nat := 7247 :: innerTail264_21
def innerCandidates264_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates264_20_checked : candidates edge 4 1709 7247 innerTail264_21 best.bound = innerCandidates264_20 := by
  unfold candidates
  have row : edge 7247 = SearchData.row7247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_20 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_20 best = best := by
  rw [innerTail264_20, scan, if_neg (by decide), innerCandidates264_20_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_21
    (if 3 ≤ innerCandidates264_20.length then search edge 3 [1709, 7247] innerCandidates264_20 8956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_21

def innerTail264_19 : List Nat := 7001 :: innerTail264_20
def innerCandidates264_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates264_19_checked : candidates edge 4 1709 7001 innerTail264_20 best.bound = innerCandidates264_19 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_19 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_19 best = best := by
  rw [innerTail264_19, scan, if_neg (by decide), innerCandidates264_19_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_20
    (if 3 ≤ innerCandidates264_19.length then search edge 3 [1709, 7001] innerCandidates264_19 8710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_20

def innerTail264_18 : List Nat := 6869 :: innerTail264_19
def innerCandidates264_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates264_18_checked : candidates edge 4 1709 6869 innerTail264_19 best.bound = innerCandidates264_18 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_18 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_18 best = best := by
  rw [innerTail264_18, scan, if_neg (by decide), innerCandidates264_18_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_19
    (if 3 ≤ innerCandidates264_18.length then search edge 3 [1709, 6869] innerCandidates264_18 8578 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_19

def innerTail264_17 : List Nat := 6857 :: innerTail264_18
def innerCandidates264_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates264_17_checked : candidates edge 4 1709 6857 innerTail264_18 best.bound = innerCandidates264_17 := by
  unfold candidates
  have row : edge 6857 = SearchData.row6857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_17 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_17 best = best := by
  rw [innerTail264_17, scan, if_neg (by decide), innerCandidates264_17_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_18
    (if 3 ≤ innerCandidates264_17.length then search edge 3 [1709, 6857] innerCandidates264_17 8566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_18

def innerTail264_16 : List Nat := 6551 :: innerTail264_17
def innerCandidates264_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates264_16_checked : candidates edge 4 1709 6551 innerTail264_17 best.bound = innerCandidates264_16 := by
  unfold candidates
  have row : edge 6551 = SearchData.row6551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_16 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_16 best = best := by
  rw [innerTail264_16, scan, if_neg (by decide), innerCandidates264_16_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_17
    (if 3 ≤ innerCandidates264_16.length then search edge 3 [1709, 6551] innerCandidates264_16 8260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_17

def innerTail264_15 : List Nat := 4523 :: innerTail264_16
def innerCandidates264_15 : List Nat := [6551]
set_option maxRecDepth 4096 in
theorem innerCandidates264_15_checked : candidates edge 4 1709 4523 innerTail264_16 best.bound = innerCandidates264_15 := by
  unfold candidates
  have row : edge 4523 = SearchData.row4523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_15 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_15 best = best := by
  rw [innerTail264_15, scan, if_neg (by decide), innerCandidates264_15_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_16
    (if 3 ≤ innerCandidates264_15.length then search edge 3 [1709, 4523] innerCandidates264_15 6232 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_16

def innerTail264_14 : List Nat := 4397 :: innerTail264_15
def innerCandidates264_14 : List Nat := [7823]
set_option maxRecDepth 4096 in
theorem innerCandidates264_14_checked : candidates edge 4 1709 4397 innerTail264_15 best.bound = innerCandidates264_14 := by
  unfold candidates
  have row : edge 4397 = SearchData.row4397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_14 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_14 best = best := by
  rw [innerTail264_14, scan, if_neg (by decide), innerCandidates264_14_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_15
    (if 3 ≤ innerCandidates264_14.length then search edge 3 [1709, 4397] innerCandidates264_14 6106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_15

def innerTail264_13 : List Nat := 4283 :: innerTail264_14
def innerCandidates264_13 : List Nat := [4523]
set_option maxRecDepth 4096 in
theorem innerCandidates264_13_checked : candidates edge 4 1709 4283 innerTail264_14 best.bound = innerCandidates264_13 := by
  unfold candidates
  have row : edge 4283 = SearchData.row4283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_13 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_13 best = best := by
  rw [innerTail264_13, scan, if_neg (by decide), innerCandidates264_13_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_14
    (if 3 ≤ innerCandidates264_13.length then search edge 3 [1709, 4283] innerCandidates264_13 5992 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_14

def innerTail264_12 : List Nat := 4013 :: innerTail264_13
def innerCandidates264_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates264_12_checked : candidates edge 4 1709 4013 innerTail264_13 best.bound = innerCandidates264_12 := by
  unfold candidates
  have row : edge 4013 = SearchData.row4013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_12 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_12 best = best := by
  rw [innerTail264_12, scan, if_neg (by decide), innerCandidates264_12_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_13
    (if 3 ≤ innerCandidates264_12.length then search edge 3 [1709, 4013] innerCandidates264_12 5722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_13

def innerTail264_11 : List Nat := 4001 :: innerTail264_12
def innerCandidates264_11 : List Nat := [7877]
set_option maxRecDepth 4096 in
theorem innerCandidates264_11_checked : candidates edge 4 1709 4001 innerTail264_12 best.bound = innerCandidates264_11 := by
  unfold candidates
  have row : edge 4001 = SearchData.row4001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_11 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_11 best = best := by
  rw [innerTail264_11, scan, if_neg (by decide), innerCandidates264_11_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_12
    (if 3 ≤ innerCandidates264_11.length then search edge 3 [1709, 4001] innerCandidates264_11 5710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_12

def innerTail264_10 : List Nat := 3917 :: innerTail264_11
def innerCandidates264_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates264_10_checked : candidates edge 4 1709 3917 innerTail264_11 best.bound = innerCandidates264_10 := by
  unfold candidates
  have row : edge 3917 = SearchData.row3917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_10 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_10 best = best := by
  rw [innerTail264_10, scan, if_neg (by decide), innerCandidates264_10_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_11
    (if 3 ≤ innerCandidates264_10.length then search edge 3 [1709, 3917] innerCandidates264_10 5626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_11

def innerTail264_9 : List Nat := 3389 :: innerTail264_10
def innerCandidates264_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates264_9_checked : candidates edge 4 1709 3389 innerTail264_10 best.bound = innerCandidates264_9 := by
  unfold candidates
  have row : edge 3389 = SearchData.row3389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_9 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_9 best = best := by
  rw [innerTail264_9, scan, if_neg (by decide), innerCandidates264_9_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_10
    (if 3 ≤ innerCandidates264_9.length then search edge 3 [1709, 3389] innerCandidates264_9 5098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_10

def innerTail264_8 : List Nat := 3299 :: innerTail264_9
def innerCandidates264_8 : List Nat := [4523, 6869]
set_option maxRecDepth 4096 in
theorem innerCandidates264_8_checked : candidates edge 4 1709 3299 innerTail264_9 best.bound = innerCandidates264_8 := by
  unfold candidates
  have row : edge 3299 = SearchData.row3299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_8 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_8 best = best := by
  rw [innerTail264_8, scan, if_neg (by decide), innerCandidates264_8_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_9
    (if 3 ≤ innerCandidates264_8.length then search edge 3 [1709, 3299] innerCandidates264_8 5008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_9

def innerTail264_7 : List Nat := 3137 :: innerTail264_8
def innerCandidates264_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates264_7_checked : candidates edge 4 1709 3137 innerTail264_8 best.bound = innerCandidates264_7 := by
  unfold candidates
  have row : edge 3137 = SearchData.row3137.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_7 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_7 best = best := by
  rw [innerTail264_7, scan, if_neg (by decide), innerCandidates264_7_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_8
    (if 3 ≤ innerCandidates264_7.length then search edge 3 [1709, 3137] innerCandidates264_7 4846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_8

def innerTail264_6 : List Nat := 2843 :: innerTail264_7
def innerCandidates264_6 : List Nat := [3389]
set_option maxRecDepth 4096 in
theorem innerCandidates264_6_checked : candidates edge 4 1709 2843 innerTail264_7 best.bound = innerCandidates264_6 := by
  unfold candidates
  have row : edge 2843 = SearchData.row2843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_6 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_6 best = best := by
  rw [innerTail264_6, scan, if_neg (by decide), innerCandidates264_6_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_7
    (if 3 ≤ innerCandidates264_6.length then search edge 3 [1709, 2843] innerCandidates264_6 4552 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_7

def innerTail264_5 : List Nat := 2591 :: innerTail264_6
def innerCandidates264_5 : List Nat := [4283, 8231]
set_option maxRecDepth 4096 in
theorem innerCandidates264_5_checked : candidates edge 4 1709 2591 innerTail264_6 best.bound = innerCandidates264_5 := by
  unfold candidates
  have row : edge 2591 = PrimePairCache.smallRow2591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_5 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_5 best = best := by
  rw [innerTail264_5, scan, if_neg (by decide), innerCandidates264_5_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_6
    (if 3 ≤ innerCandidates264_5.length then search edge 3 [1709, 2591] innerCandidates264_5 4300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_6

def innerTail264_4 : List Nat := 2423 :: innerTail264_5
def innerCandidates264_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates264_4_checked : candidates edge 4 1709 2423 innerTail264_5 best.bound = innerCandidates264_4 := by
  unfold candidates
  have row : edge 2423 = PrimePairCache.smallRow2423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_4 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_4 best = best := by
  rw [innerTail264_4, scan, if_neg (by decide), innerCandidates264_4_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_5
    (if 3 ≤ innerCandidates264_4.length then search edge 3 [1709, 2423] innerCandidates264_4 4132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_5

def innerTail264_3 : List Nat := 2393 :: innerTail264_4
def innerCandidates264_3 : List Nat := [2423]
set_option maxRecDepth 4096 in
theorem innerCandidates264_3_checked : candidates edge 4 1709 2393 innerTail264_4 best.bound = innerCandidates264_3 := by
  unfold candidates
  have row : edge 2393 = PrimePairCache.smallRow2393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_3 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_3 best = best := by
  rw [innerTail264_3, scan, if_neg (by decide), innerCandidates264_3_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_4
    (if 3 ≤ innerCandidates264_3.length then search edge 3 [1709, 2393] innerCandidates264_3 4102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_4

def innerTail264_2 : List Nat := 2333 :: innerTail264_3
def innerCandidates264_2 : List Nat := [2423, 2843, 3137]
set_option maxRecDepth 4096 in
theorem innerCandidates264_2_checked : candidates edge 4 1709 2333 innerTail264_3 best.bound = innerCandidates264_2 := by
  unfold candidates
  have row : edge 2333 = PrimePairCache.smallRow2333.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild264_2 : search edge 3 [1709, 2333] innerCandidates264_2 4042 best = best := by decide

theorem innerScan264_2 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_2 best = best := by
  rw [innerTail264_2, scan, if_neg (by decide), innerCandidates264_2_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_3
    (if 3 ≤ innerCandidates264_2.length then search edge 3 [1709, 2333] innerCandidates264_2 4042 best else best) = best
  rw [if_pos (by decide), innerChild264_2]
  exact innerScan264_3

def innerTail264_1 : List Nat := 2213 :: innerTail264_2
def innerCandidates264_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates264_1_checked : candidates edge 4 1709 2213 innerTail264_2 best.bound = innerCandidates264_1 := by
  unfold candidates
  have row : edge 2213 = PrimePairCache.smallRow2213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan264_1 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_1 best = best := by
  rw [innerTail264_1, scan, if_neg (by decide), innerCandidates264_1_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_2
    (if 3 ≤ innerCandidates264_1.length then search edge 3 [1709, 2213] innerCandidates264_1 3922 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan264_2

def innerTail264_0 : List Nat := 1913 :: innerTail264_1
def innerCandidates264_0 : List Nat := [2213, 2843, 8297]
set_option maxRecDepth 4096 in
theorem innerCandidates264_0_checked : candidates edge 4 1709 1913 innerTail264_1 best.bound = innerCandidates264_0 := by
  unfold candidates
  have row : edge 1913 = PrimePairCache.smallRow1913.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild264_0 : search edge 3 [1709, 1913] innerCandidates264_0 3622 best = best := by decide

theorem innerScan264_0 : scan edge (search edge 3) 4 [1709] 1709 innerTail264_0 best = best := by
  rw [innerTail264_0, scan, if_neg (by decide), innerCandidates264_0_checked]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_1
    (if 3 ≤ innerCandidates264_0.length then search edge 3 [1709, 1913] innerCandidates264_0 3622 best else best) = best
  rw [if_pos (by decide), innerChild264_0]
  exact innerScan264_1

theorem rootBranch264 : search edge 4 [1709] rootCandidates264 1709 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1709] 1709 innerTail264_0 best = best
  exact innerScan264_0

def rootCandidates263 : List Nat := [1759, 1861, 2179, 2203, 2851, 3313, 3373, 3607, 3853, 4177, 4363, 4567, 5659, 5791, 6121, 6571, 6691, 7507, 7549, 7639]
set_option maxRecDepth 4096 in
theorem rootCandidates263_checked : candidates edge 5 0 1699 tail264 best.bound = rootCandidates263 := by
  unfold candidates
  have row : edge 1699 = PrimePairCache.smallRow1699.contains := by funext q; rfl
  rw [row]
  decide

def innerTail263_20 : List Nat := []
theorem innerScan263_20 : scan edge (search edge 3) 4 [1699] 1699 innerTail263_20 best = best := rfl

def innerTail263_19 : List Nat := 7639 :: innerTail263_20
def innerCandidates263_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates263_19_checked : candidates edge 4 1699 7639 innerTail263_20 best.bound = innerCandidates263_19 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan263_19 : scan edge (search edge 3) 4 [1699] 1699 innerTail263_19 best = best := by
  rw [innerTail263_19, scan, if_neg (by decide), innerCandidates263_19_checked]
  change scan edge (search edge 3) 4 [1699] 1699 innerTail263_20
    (if 3 ≤ innerCandidates263_19.length then search edge 3 [1699, 7639] innerCandidates263_19 9338 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan263_20

def innerTail263_18 : List Nat := 7549 :: innerTail263_19
def innerCandidates263_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates263_18_checked : candidates edge 4 1699 7549 innerTail263_19 best.bound = innerCandidates263_18 := by
  unfold candidates
  have row : edge 7549 = SearchData.row7549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan263_18 : scan edge (search edge 3) 4 [1699] 1699 innerTail263_18 best = best := by
  rw [innerTail263_18, scan, if_neg (by decide), innerCandidates263_18_checked]
  change scan edge (search edge 3) 4 [1699] 1699 innerTail263_19
    (if 3 ≤ innerCandidates263_18.length then search edge 3 [1699, 7549] innerCandidates263_18 9248 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan263_19

def innerTail263_17 : List Nat := 7507 :: innerTail263_18
def innerCandidates263_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates263_17_checked : candidates edge 4 1699 7507 innerTail263_18 best.bound = innerCandidates263_17 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan263_17 : scan edge (search edge 3) 4 [1699] 1699 innerTail263_17 best = best := by
  rw [innerTail263_17, scan, if_neg (by decide), innerCandidates263_17_checked]
  change scan edge (search edge 3) 4 [1699] 1699 innerTail263_18
    (if 3 ≤ innerCandidates263_17.length then search edge 3 [1699, 7507] innerCandidates263_17 9206 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan263_18

def innerTail263_16 : List Nat := 6691 :: innerTail263_17
def innerCandidates263_16 : List Nat := [7507]
set_option maxRecDepth 4096 in
theorem innerCandidates263_16_checked : candidates edge 4 1699 6691 innerTail263_17 best.bound = innerCandidates263_16 := by
  unfold candidates
  have row : edge 6691 = SearchData.row6691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan263_16 : scan edge (search edge 3) 4 [1699] 1699 innerTail263_16 best = best := by
  rw [innerTail263_16, scan, if_neg (by decide), innerCandidates263_16_checked]
  change scan edge (search edge 3) 4 [1699] 1699 innerTail263_17
    (if 3 ≤ innerCandidates263_16.length then search edge 3 [1699, 6691] innerCandidates263_16 8390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan263_17

def innerTail263_15 : List Nat := 6571 :: innerTail263_16
def innerCandidates263_15 : List Nat := [7549]
set_option maxRecDepth 4096 in
theorem innerCandidates263_15_checked : candidates edge 4 1699 6571 innerTail263_16 best.bound = innerCandidates263_15 := by
  unfold candidates
  have row : edge 6571 = SearchData.row6571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan263_15 : scan edge (search edge 3) 4 [1699] 1699 innerTail263_15 best = best := by
  rw [innerTail263_15, scan, if_neg (by decide), innerCandidates263_15_checked]
  change scan edge (search edge 3) 4 [1699] 1699 innerTail263_16
    (if 3 ≤ innerCandidates263_15.length then search edge 3 [1699, 6571] innerCandidates263_15 8270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan263_16

def innerTail263_14 : List Nat := 6121 :: innerTail263_15
def innerCandidates263_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates263_14_checked : candidates edge 4 1699 6121 innerTail263_15 best.bound = innerCandidates263_14 := by
  unfold candidates
  have row : edge 6121 = SearchData.row6121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan263_14 : scan edge (search edge 3) 4 [1699] 1699 innerTail263_14 best = best := by
  rw [innerTail263_14, scan, if_neg (by decide), innerCandidates263_14_checked]
  change scan edge (search edge 3) 4 [1699] 1699 innerTail263_15
    (if 3 ≤ innerCandidates263_14.length then search edge 3 [1699, 6121] innerCandidates263_14 7820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan263_15

def innerTail263_13 : List Nat := 5791 :: innerTail263_14
def innerCandidates263_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates263_13_checked : candidates edge 4 1699 5791 innerTail263_14 best.bound = innerCandidates263_13 := by
  unfold candidates
  have row : edge 5791 = SearchData.row5791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan263_13 : scan edge (search edge 3) 4 [1699] 1699 innerTail263_13 best = best := by
  rw [innerTail263_13, scan, if_neg (by decide), innerCandidates263_13_checked]
  change scan edge (search edge 3) 4 [1699] 1699 innerTail263_14
    (if 3 ≤ innerCandidates263_13.length then search edge 3 [1699, 5791] innerCandidates263_13 7490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan263_14

def innerTail263_12 : List Nat := 5659 :: innerTail263_13
def innerCandidates263_12 : List Nat := [6121]
set_option maxRecDepth 4096 in
theorem innerCandidates263_12_checked : candidates edge 4 1699 5659 innerTail263_13 best.bound = innerCandidates263_12 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan263_12 : scan edge (search edge 3) 4 [1699] 1699 innerTail263_12 best = best := by
  rw [innerTail263_12, scan, if_neg (by decide), innerCandidates263_12_checked]
  change scan edge (search edge 3) 4 [1699] 1699 innerTail263_13
    (if 3 ≤ innerCandidates263_12.length then search edge 3 [1699, 5659] innerCandidates263_12 7358 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan263_13

def innerTail263_11 : List Nat := 4567 :: innerTail263_12
def innerCandidates263_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates263_11_checked : candidates edge 4 1699 4567 innerTail263_12 best.bound = innerCandidates263_11 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan263_11 : scan edge (search edge 3) 4 [1699] 1699 innerTail263_11 best = best := by
  rw [innerTail263_11, scan, if_neg (by decide), innerCandidates263_11_checked]
  change scan edge (search edge 3) 4 [1699] 1699 innerTail263_12
    (if 3 ≤ innerCandidates263_11.length then search edge 3 [1699, 4567] innerCandidates263_11 6266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan263_12

def innerTail263_10 : List Nat := 4363 :: innerTail263_11
def innerCandidates263_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates263_10_checked : candidates edge 4 1699 4363 innerTail263_11 best.bound = innerCandidates263_10 := by
  unfold candidates
  have row : edge 4363 = SearchData.row4363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan263_10 : scan edge (search edge 3) 4 [1699] 1699 innerTail263_10 best = best := by
  rw [innerTail263_10, scan, if_neg (by decide), innerCandidates263_10_checked]
  change scan edge (search edge 3) 4 [1699] 1699 innerTail263_11
    (if 3 ≤ innerCandidates263_10.length then search edge 3 [1699, 4363] innerCandidates263_10 6062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan263_11

def innerTail263_9 : List Nat := 4177 :: innerTail263_10
def innerCandidates263_9 : List Nat := [4363, 5791]
set_option maxRecDepth 4096 in
theorem innerCandidates263_9_checked : candidates edge 4 1699 4177 innerTail263_10 best.bound = innerCandidates263_9 := by
  unfold candidates
  have row : edge 4177 = SearchData.row4177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan263_9 : scan edge (search edge 3) 4 [1699] 1699 innerTail263_9 best = best := by
  rw [innerTail263_9, scan, if_neg (by decide), innerCandidates263_9_checked]
  change scan edge (search edge 3) 4 [1699] 1699 innerTail263_10
    (if 3 ≤ innerCandidates263_9.length then search edge 3 [1699, 4177] innerCandidates263_9 5876 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan263_10

def innerTail263_8 : List Nat := 3853 :: innerTail263_9
def innerCandidates263_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates263_8_checked : candidates edge 4 1699 3853 innerTail263_9 best.bound = innerCandidates263_8 := by
  unfold candidates
  have row : edge 3853 = SearchData.row3853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan263_8 : scan edge (search edge 3) 4 [1699] 1699 innerTail263_8 best = best := by
  rw [innerTail263_8, scan, if_neg (by decide), innerCandidates263_8_checked]
  change scan edge (search edge 3) 4 [1699] 1699 innerTail263_9
    (if 3 ≤ innerCandidates263_8.length then search edge 3 [1699, 3853] innerCandidates263_8 5552 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan263_9

def innerTail263_7 : List Nat := 3607 :: innerTail263_8
def innerCandidates263_7 : List Nat := [6571]
set_option maxRecDepth 4096 in
theorem innerCandidates263_7_checked : candidates edge 4 1699 3607 innerTail263_8 best.bound = innerCandidates263_7 := by
  unfold candidates
  have row : edge 3607 = SearchData.row3607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan263_7 : scan edge (search edge 3) 4 [1699] 1699 innerTail263_7 best = best := by
  rw [innerTail263_7, scan, if_neg (by decide), innerCandidates263_7_checked]
  change scan edge (search edge 3) 4 [1699] 1699 innerTail263_8
    (if 3 ≤ innerCandidates263_7.length then search edge 3 [1699, 3607] innerCandidates263_7 5306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan263_8

def innerTail263_6 : List Nat := 3373 :: innerTail263_7
def innerCandidates263_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates263_6_checked : candidates edge 4 1699 3373 innerTail263_7 best.bound = innerCandidates263_6 := by
  unfold candidates
  have row : edge 3373 = SearchData.row3373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan263_6 : scan edge (search edge 3) 4 [1699] 1699 innerTail263_6 best = best := by
  rw [innerTail263_6, scan, if_neg (by decide), innerCandidates263_6_checked]
  change scan edge (search edge 3) 4 [1699] 1699 innerTail263_7
    (if 3 ≤ innerCandidates263_6.length then search edge 3 [1699, 3373] innerCandidates263_6 5072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan263_7

def innerTail263_5 : List Nat := 3313 :: innerTail263_6
def innerCandidates263_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates263_5_checked : candidates edge 4 1699 3313 innerTail263_6 best.bound = innerCandidates263_5 := by
  unfold candidates
  have row : edge 3313 = SearchData.row3313.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan263_5 : scan edge (search edge 3) 4 [1699] 1699 innerTail263_5 best = best := by
  rw [innerTail263_5, scan, if_neg (by decide), innerCandidates263_5_checked]
  change scan edge (search edge 3) 4 [1699] 1699 innerTail263_6
    (if 3 ≤ innerCandidates263_5.length then search edge 3 [1699, 3313] innerCandidates263_5 5012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan263_6

def innerTail263_4 : List Nat := 2851 :: innerTail263_5
def innerCandidates263_4 : List Nat := [3313]
set_option maxRecDepth 4096 in
theorem innerCandidates263_4_checked : candidates edge 4 1699 2851 innerTail263_5 best.bound = innerCandidates263_4 := by
  unfold candidates
  have row : edge 2851 = SearchData.row2851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan263_4 : scan edge (search edge 3) 4 [1699] 1699 innerTail263_4 best = best := by
  rw [innerTail263_4, scan, if_neg (by decide), innerCandidates263_4_checked]
  change scan edge (search edge 3) 4 [1699] 1699 innerTail263_5
    (if 3 ≤ innerCandidates263_4.length then search edge 3 [1699, 2851] innerCandidates263_4 4550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan263_5

def innerTail263_3 : List Nat := 2203 :: innerTail263_4
def innerCandidates263_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates263_3_checked : candidates edge 4 1699 2203 innerTail263_4 best.bound = innerCandidates263_3 := by
  unfold candidates
  have row : edge 2203 = PrimePairCache.smallRow2203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan263_3 : scan edge (search edge 3) 4 [1699] 1699 innerTail263_3 best = best := by
  rw [innerTail263_3, scan, if_neg (by decide), innerCandidates263_3_checked]
  change scan edge (search edge 3) 4 [1699] 1699 innerTail263_4
    (if 3 ≤ innerCandidates263_3.length then search edge 3 [1699, 2203] innerCandidates263_3 3902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan263_4

def innerTail263_2 : List Nat := 2179 :: innerTail263_3
def innerCandidates263_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates263_2_checked : candidates edge 4 1699 2179 innerTail263_3 best.bound = innerCandidates263_2 := by
  unfold candidates
  have row : edge 2179 = PrimePairCache.smallRow2179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan263_2 : scan edge (search edge 3) 4 [1699] 1699 innerTail263_2 best = best := by
  rw [innerTail263_2, scan, if_neg (by decide), innerCandidates263_2_checked]
  change scan edge (search edge 3) 4 [1699] 1699 innerTail263_3
    (if 3 ≤ innerCandidates263_2.length then search edge 3 [1699, 2179] innerCandidates263_2 3878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan263_3

def innerTail263_1 : List Nat := 1861 :: innerTail263_2
def innerCandidates263_1 : List Nat := [6121]
set_option maxRecDepth 4096 in
theorem innerCandidates263_1_checked : candidates edge 4 1699 1861 innerTail263_2 best.bound = innerCandidates263_1 := by
  unfold candidates
  have row : edge 1861 = PrimePairCache.smallRow1861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan263_1 : scan edge (search edge 3) 4 [1699] 1699 innerTail263_1 best = best := by
  rw [innerTail263_1, scan, if_neg (by decide), innerCandidates263_1_checked]
  change scan edge (search edge 3) 4 [1699] 1699 innerTail263_2
    (if 3 ≤ innerCandidates263_1.length then search edge 3 [1699, 1861] innerCandidates263_1 3560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan263_2

def innerTail263_0 : List Nat := 1759 :: innerTail263_1
def innerCandidates263_0 : List Nat := [4177]
set_option maxRecDepth 4096 in
theorem innerCandidates263_0_checked : candidates edge 4 1699 1759 innerTail263_1 best.bound = innerCandidates263_0 := by
  unfold candidates
  have row : edge 1759 = PrimePairCache.smallRow1759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan263_0 : scan edge (search edge 3) 4 [1699] 1699 innerTail263_0 best = best := by
  rw [innerTail263_0, scan, if_neg (by decide), innerCandidates263_0_checked]
  change scan edge (search edge 3) 4 [1699] 1699 innerTail263_1
    (if 3 ≤ innerCandidates263_0.length then search edge 3 [1699, 1759] innerCandidates263_0 3458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan263_1

theorem rootBranch263 : search edge 4 [1699] rootCandidates263 1699 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1699] 1699 innerTail263_0 best = best
  exact innerScan263_0

def rootCandidates262 : List Nat := [1979, 2027, 2741, 3041, 3527, 4931, 5087, 5099, 5477, 6689, 7193, 7643, 7901, 8237]
set_option maxRecDepth 4096 in
theorem rootCandidates262_checked : candidates edge 5 0 1697 tail263 best.bound = rootCandidates262 := by
  unfold candidates
  have row : edge 1697 = PrimePairCache.smallRow1697.contains := by funext q; rfl
  rw [row]
  decide

def innerTail262_14 : List Nat := []
theorem innerScan262_14 : scan edge (search edge 3) 4 [1697] 1697 innerTail262_14 best = best := rfl

def innerTail262_13 : List Nat := 8237 :: innerTail262_14
def innerCandidates262_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates262_13_checked : candidates edge 4 1697 8237 innerTail262_14 best.bound = innerCandidates262_13 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan262_13 : scan edge (search edge 3) 4 [1697] 1697 innerTail262_13 best = best := by
  rw [innerTail262_13, scan, if_neg (by decide), innerCandidates262_13_checked]
  change scan edge (search edge 3) 4 [1697] 1697 innerTail262_14
    (if 3 ≤ innerCandidates262_13.length then search edge 3 [1697, 8237] innerCandidates262_13 9934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan262_14

def innerTail262_12 : List Nat := 7901 :: innerTail262_13
def innerCandidates262_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates262_12_checked : candidates edge 4 1697 7901 innerTail262_13 best.bound = innerCandidates262_12 := by
  unfold candidates
  have row : edge 7901 = SearchData.row7901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan262_12 : scan edge (search edge 3) 4 [1697] 1697 innerTail262_12 best = best := by
  rw [innerTail262_12, scan, if_neg (by decide), innerCandidates262_12_checked]
  change scan edge (search edge 3) 4 [1697] 1697 innerTail262_13
    (if 3 ≤ innerCandidates262_12.length then search edge 3 [1697, 7901] innerCandidates262_12 9598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan262_13

def innerTail262_11 : List Nat := 7643 :: innerTail262_12
def innerCandidates262_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates262_11_checked : candidates edge 4 1697 7643 innerTail262_12 best.bound = innerCandidates262_11 := by
  unfold candidates
  have row : edge 7643 = SearchData.row7643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan262_11 : scan edge (search edge 3) 4 [1697] 1697 innerTail262_11 best = best := by
  rw [innerTail262_11, scan, if_neg (by decide), innerCandidates262_11_checked]
  change scan edge (search edge 3) 4 [1697] 1697 innerTail262_12
    (if 3 ≤ innerCandidates262_11.length then search edge 3 [1697, 7643] innerCandidates262_11 9340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan262_12

def innerTail262_10 : List Nat := 7193 :: innerTail262_11
def innerCandidates262_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates262_10_checked : candidates edge 4 1697 7193 innerTail262_11 best.bound = innerCandidates262_10 := by
  unfold candidates
  have row : edge 7193 = SearchData.row7193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan262_10 : scan edge (search edge 3) 4 [1697] 1697 innerTail262_10 best = best := by
  rw [innerTail262_10, scan, if_neg (by decide), innerCandidates262_10_checked]
  change scan edge (search edge 3) 4 [1697] 1697 innerTail262_11
    (if 3 ≤ innerCandidates262_10.length then search edge 3 [1697, 7193] innerCandidates262_10 8890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan262_11

def innerTail262_9 : List Nat := 6689 :: innerTail262_10
def innerCandidates262_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates262_9_checked : candidates edge 4 1697 6689 innerTail262_10 best.bound = innerCandidates262_9 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan262_9 : scan edge (search edge 3) 4 [1697] 1697 innerTail262_9 best = best := by
  rw [innerTail262_9, scan, if_neg (by decide), innerCandidates262_9_checked]
  change scan edge (search edge 3) 4 [1697] 1697 innerTail262_10
    (if 3 ≤ innerCandidates262_9.length then search edge 3 [1697, 6689] innerCandidates262_9 8386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan262_10

def innerTail262_8 : List Nat := 5477 :: innerTail262_9
def innerCandidates262_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates262_8_checked : candidates edge 4 1697 5477 innerTail262_9 best.bound = innerCandidates262_8 := by
  unfold candidates
  have row : edge 5477 = SearchData.row5477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan262_8 : scan edge (search edge 3) 4 [1697] 1697 innerTail262_8 best = best := by
  rw [innerTail262_8, scan, if_neg (by decide), innerCandidates262_8_checked]
  change scan edge (search edge 3) 4 [1697] 1697 innerTail262_9
    (if 3 ≤ innerCandidates262_8.length then search edge 3 [1697, 5477] innerCandidates262_8 7174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan262_9

def innerTail262_7 : List Nat := 5099 :: innerTail262_8
def innerCandidates262_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates262_7_checked : candidates edge 4 1697 5099 innerTail262_8 best.bound = innerCandidates262_7 := by
  unfold candidates
  have row : edge 5099 = SearchData.row5099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan262_7 : scan edge (search edge 3) 4 [1697] 1697 innerTail262_7 best = best := by
  rw [innerTail262_7, scan, if_neg (by decide), innerCandidates262_7_checked]
  change scan edge (search edge 3) 4 [1697] 1697 innerTail262_8
    (if 3 ≤ innerCandidates262_7.length then search edge 3 [1697, 5099] innerCandidates262_7 6796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan262_8

def innerTail262_6 : List Nat := 5087 :: innerTail262_7
def innerCandidates262_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates262_6_checked : candidates edge 4 1697 5087 innerTail262_7 best.bound = innerCandidates262_6 := by
  unfold candidates
  have row : edge 5087 = SearchData.row5087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan262_6 : scan edge (search edge 3) 4 [1697] 1697 innerTail262_6 best = best := by
  rw [innerTail262_6, scan, if_neg (by decide), innerCandidates262_6_checked]
  change scan edge (search edge 3) 4 [1697] 1697 innerTail262_7
    (if 3 ≤ innerCandidates262_6.length then search edge 3 [1697, 5087] innerCandidates262_6 6784 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan262_7

def innerTail262_5 : List Nat := 4931 :: innerTail262_6
def innerCandidates262_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates262_5_checked : candidates edge 4 1697 4931 innerTail262_6 best.bound = innerCandidates262_5 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan262_5 : scan edge (search edge 3) 4 [1697] 1697 innerTail262_5 best = best := by
  rw [innerTail262_5, scan, if_neg (by decide), innerCandidates262_5_checked]
  change scan edge (search edge 3) 4 [1697] 1697 innerTail262_6
    (if 3 ≤ innerCandidates262_5.length then search edge 3 [1697, 4931] innerCandidates262_5 6628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan262_6

def innerTail262_4 : List Nat := 3527 :: innerTail262_5
def innerCandidates262_4 : List Nat := [7901]
set_option maxRecDepth 4096 in
theorem innerCandidates262_4_checked : candidates edge 4 1697 3527 innerTail262_5 best.bound = innerCandidates262_4 := by
  unfold candidates
  have row : edge 3527 = SearchData.row3527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan262_4 : scan edge (search edge 3) 4 [1697] 1697 innerTail262_4 best = best := by
  rw [innerTail262_4, scan, if_neg (by decide), innerCandidates262_4_checked]
  change scan edge (search edge 3) 4 [1697] 1697 innerTail262_5
    (if 3 ≤ innerCandidates262_4.length then search edge 3 [1697, 3527] innerCandidates262_4 5224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan262_5

def innerTail262_3 : List Nat := 3041 :: innerTail262_4
def innerCandidates262_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates262_3_checked : candidates edge 4 1697 3041 innerTail262_4 best.bound = innerCandidates262_3 := by
  unfold candidates
  have row : edge 3041 = SearchData.row3041.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan262_3 : scan edge (search edge 3) 4 [1697] 1697 innerTail262_3 best = best := by
  rw [innerTail262_3, scan, if_neg (by decide), innerCandidates262_3_checked]
  change scan edge (search edge 3) 4 [1697] 1697 innerTail262_4
    (if 3 ≤ innerCandidates262_3.length then search edge 3 [1697, 3041] innerCandidates262_3 4738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan262_4

def innerTail262_2 : List Nat := 2741 :: innerTail262_3
def innerCandidates262_2 : List Nat := [6689]
set_option maxRecDepth 4096 in
theorem innerCandidates262_2_checked : candidates edge 4 1697 2741 innerTail262_3 best.bound = innerCandidates262_2 := by
  unfold candidates
  have row : edge 2741 = SearchData.row2741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan262_2 : scan edge (search edge 3) 4 [1697] 1697 innerTail262_2 best = best := by
  rw [innerTail262_2, scan, if_neg (by decide), innerCandidates262_2_checked]
  change scan edge (search edge 3) 4 [1697] 1697 innerTail262_3
    (if 3 ≤ innerCandidates262_2.length then search edge 3 [1697, 2741] innerCandidates262_2 4438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan262_3

def innerTail262_1 : List Nat := 2027 :: innerTail262_2
def innerCandidates262_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates262_1_checked : candidates edge 4 1697 2027 innerTail262_2 best.bound = innerCandidates262_1 := by
  unfold candidates
  have row : edge 2027 = PrimePairCache.smallRow2027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan262_1 : scan edge (search edge 3) 4 [1697] 1697 innerTail262_1 best = best := by
  rw [innerTail262_1, scan, if_neg (by decide), innerCandidates262_1_checked]
  change scan edge (search edge 3) 4 [1697] 1697 innerTail262_2
    (if 3 ≤ innerCandidates262_1.length then search edge 3 [1697, 2027] innerCandidates262_1 3724 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan262_2

def innerTail262_0 : List Nat := 1979 :: innerTail262_1
def innerCandidates262_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates262_0_checked : candidates edge 4 1697 1979 innerTail262_1 best.bound = innerCandidates262_0 := by
  unfold candidates
  have row : edge 1979 = PrimePairCache.smallRow1979.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan262_0 : scan edge (search edge 3) 4 [1697] 1697 innerTail262_0 best = best := by
  rw [innerTail262_0, scan, if_neg (by decide), innerCandidates262_0_checked]
  change scan edge (search edge 3) 4 [1697] 1697 innerTail262_1
    (if 3 ≤ innerCandidates262_0.length then search edge 3 [1697, 1979] innerCandidates262_0 3676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan262_1

theorem rootBranch262 : search edge 4 [1697] rootCandidates262 1697 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1697] 1697 innerTail262_0 best = best
  exact innerScan262_0

end ProjectEulerSolutions.PrimePairUpper.Search
