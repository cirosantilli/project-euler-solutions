import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates118 : List Nat := [769, 877, 883, 1237, 1447, 1489, 1777, 1933, 1993, 2539, 2791, 2857, 2953, 3331, 3361, 3823, 3931, 4759, 5503, 5641, 6151, 6529, 6781, 6871, 7159, 7873]
set_option maxRecDepth 4096 in
theorem rootCandidates118_checked : candidates edge 5 0 661 tail119 best.bound = rootCandidates118 := by
  unfold candidates
  have row : edge 661 = PrimePairCache.smallRow661.contains := by funext q; rfl
  rw [row]
  decide

def innerTail118_26 : List Nat := []
theorem innerScan118_26 : scan edge (search edge 3) 4 [661] 661 innerTail118_26 best = best := rfl

def innerTail118_25 : List Nat := 7873 :: innerTail118_26
def innerCandidates118_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates118_25_checked : candidates edge 4 661 7873 innerTail118_26 best.bound = innerCandidates118_25 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_25 : scan edge (search edge 3) 4 [661] 661 innerTail118_25 best = best := by
  rw [innerTail118_25, scan, if_neg (by decide), innerCandidates118_25_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_26
    (if 3 ≤ innerCandidates118_25.length then search edge 3 [661, 7873] innerCandidates118_25 8534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_26

def innerTail118_24 : List Nat := 7159 :: innerTail118_25
def innerCandidates118_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates118_24_checked : candidates edge 4 661 7159 innerTail118_25 best.bound = innerCandidates118_24 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_24 : scan edge (search edge 3) 4 [661] 661 innerTail118_24 best = best := by
  rw [innerTail118_24, scan, if_neg (by decide), innerCandidates118_24_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_25
    (if 3 ≤ innerCandidates118_24.length then search edge 3 [661, 7159] innerCandidates118_24 7820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_25

def innerTail118_23 : List Nat := 6871 :: innerTail118_24
def innerCandidates118_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates118_23_checked : candidates edge 4 661 6871 innerTail118_24 best.bound = innerCandidates118_23 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_23 : scan edge (search edge 3) 4 [661] 661 innerTail118_23 best = best := by
  rw [innerTail118_23, scan, if_neg (by decide), innerCandidates118_23_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_24
    (if 3 ≤ innerCandidates118_23.length then search edge 3 [661, 6871] innerCandidates118_23 7532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_24

def innerTail118_22 : List Nat := 6781 :: innerTail118_23
def innerCandidates118_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates118_22_checked : candidates edge 4 661 6781 innerTail118_23 best.bound = innerCandidates118_22 := by
  unfold candidates
  have row : edge 6781 = SearchData.row6781.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_22 : scan edge (search edge 3) 4 [661] 661 innerTail118_22 best = best := by
  rw [innerTail118_22, scan, if_neg (by decide), innerCandidates118_22_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_23
    (if 3 ≤ innerCandidates118_22.length then search edge 3 [661, 6781] innerCandidates118_22 7442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_23

def innerTail118_21 : List Nat := 6529 :: innerTail118_22
def innerCandidates118_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates118_21_checked : candidates edge 4 661 6529 innerTail118_22 best.bound = innerCandidates118_21 := by
  unfold candidates
  have row : edge 6529 = SearchData.row6529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_21 : scan edge (search edge 3) 4 [661] 661 innerTail118_21 best = best := by
  rw [innerTail118_21, scan, if_neg (by decide), innerCandidates118_21_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_22
    (if 3 ≤ innerCandidates118_21.length then search edge 3 [661, 6529] innerCandidates118_21 7190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_22

def innerTail118_20 : List Nat := 6151 :: innerTail118_21
def innerCandidates118_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates118_20_checked : candidates edge 4 661 6151 innerTail118_21 best.bound = innerCandidates118_20 := by
  unfold candidates
  have row : edge 6151 = SearchData.row6151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_20 : scan edge (search edge 3) 4 [661] 661 innerTail118_20 best = best := by
  rw [innerTail118_20, scan, if_neg (by decide), innerCandidates118_20_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_21
    (if 3 ≤ innerCandidates118_20.length then search edge 3 [661, 6151] innerCandidates118_20 6812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_21

def innerTail118_19 : List Nat := 5641 :: innerTail118_20
def innerCandidates118_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates118_19_checked : candidates edge 4 661 5641 innerTail118_20 best.bound = innerCandidates118_19 := by
  unfold candidates
  have row : edge 5641 = SearchData.row5641.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_19 : scan edge (search edge 3) 4 [661] 661 innerTail118_19 best = best := by
  rw [innerTail118_19, scan, if_neg (by decide), innerCandidates118_19_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_20
    (if 3 ≤ innerCandidates118_19.length then search edge 3 [661, 5641] innerCandidates118_19 6302 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_20

def innerTail118_18 : List Nat := 5503 :: innerTail118_19
def innerCandidates118_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates118_18_checked : candidates edge 4 661 5503 innerTail118_19 best.bound = innerCandidates118_18 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_18 : scan edge (search edge 3) 4 [661] 661 innerTail118_18 best = best := by
  rw [innerTail118_18, scan, if_neg (by decide), innerCandidates118_18_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_19
    (if 3 ≤ innerCandidates118_18.length then search edge 3 [661, 5503] innerCandidates118_18 6164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_19

def innerTail118_17 : List Nat := 4759 :: innerTail118_18
def innerCandidates118_17 : List Nat := [6151]
set_option maxRecDepth 4096 in
theorem innerCandidates118_17_checked : candidates edge 4 661 4759 innerTail118_18 best.bound = innerCandidates118_17 := by
  unfold candidates
  have row : edge 4759 = SearchData.row4759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_17 : scan edge (search edge 3) 4 [661] 661 innerTail118_17 best = best := by
  rw [innerTail118_17, scan, if_neg (by decide), innerCandidates118_17_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_18
    (if 3 ≤ innerCandidates118_17.length then search edge 3 [661, 4759] innerCandidates118_17 5420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_18

def innerTail118_16 : List Nat := 3931 :: innerTail118_17
def innerCandidates118_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates118_16_checked : candidates edge 4 661 3931 innerTail118_17 best.bound = innerCandidates118_16 := by
  unfold candidates
  have row : edge 3931 = SearchData.row3931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_16 : scan edge (search edge 3) 4 [661] 661 innerTail118_16 best = best := by
  rw [innerTail118_16, scan, if_neg (by decide), innerCandidates118_16_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_17
    (if 3 ≤ innerCandidates118_16.length then search edge 3 [661, 3931] innerCandidates118_16 4592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_17

def innerTail118_15 : List Nat := 3823 :: innerTail118_16
def innerCandidates118_15 : List Nat := [4759]
set_option maxRecDepth 4096 in
theorem innerCandidates118_15_checked : candidates edge 4 661 3823 innerTail118_16 best.bound = innerCandidates118_15 := by
  unfold candidates
  have row : edge 3823 = SearchData.row3823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_15 : scan edge (search edge 3) 4 [661] 661 innerTail118_15 best = best := by
  rw [innerTail118_15, scan, if_neg (by decide), innerCandidates118_15_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_16
    (if 3 ≤ innerCandidates118_15.length then search edge 3 [661, 3823] innerCandidates118_15 4484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_16

def innerTail118_14 : List Nat := 3361 :: innerTail118_15
def innerCandidates118_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates118_14_checked : candidates edge 4 661 3361 innerTail118_15 best.bound = innerCandidates118_14 := by
  unfold candidates
  have row : edge 3361 = SearchData.row3361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_14 : scan edge (search edge 3) 4 [661] 661 innerTail118_14 best = best := by
  rw [innerTail118_14, scan, if_neg (by decide), innerCandidates118_14_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_15
    (if 3 ≤ innerCandidates118_14.length then search edge 3 [661, 3361] innerCandidates118_14 4022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_15

def innerTail118_13 : List Nat := 3331 :: innerTail118_14
def innerCandidates118_13 : List Nat := [7159]
set_option maxRecDepth 4096 in
theorem innerCandidates118_13_checked : candidates edge 4 661 3331 innerTail118_14 best.bound = innerCandidates118_13 := by
  unfold candidates
  have row : edge 3331 = SearchData.row3331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_13 : scan edge (search edge 3) 4 [661] 661 innerTail118_13 best = best := by
  rw [innerTail118_13, scan, if_neg (by decide), innerCandidates118_13_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_14
    (if 3 ≤ innerCandidates118_13.length then search edge 3 [661, 3331] innerCandidates118_13 3992 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_14

def innerTail118_12 : List Nat := 2953 :: innerTail118_13
def innerCandidates118_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates118_12_checked : candidates edge 4 661 2953 innerTail118_13 best.bound = innerCandidates118_12 := by
  unfold candidates
  have row : edge 2953 = SearchData.row2953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_12 : scan edge (search edge 3) 4 [661] 661 innerTail118_12 best = best := by
  rw [innerTail118_12, scan, if_neg (by decide), innerCandidates118_12_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_13
    (if 3 ≤ innerCandidates118_12.length then search edge 3 [661, 2953] innerCandidates118_12 3614 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_13

def innerTail118_11 : List Nat := 2857 :: innerTail118_12
def innerCandidates118_11 : List Nat := [5641]
set_option maxRecDepth 4096 in
theorem innerCandidates118_11_checked : candidates edge 4 661 2857 innerTail118_12 best.bound = innerCandidates118_11 := by
  unfold candidates
  have row : edge 2857 = SearchData.row2857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_11 : scan edge (search edge 3) 4 [661] 661 innerTail118_11 best = best := by
  rw [innerTail118_11, scan, if_neg (by decide), innerCandidates118_11_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_12
    (if 3 ≤ innerCandidates118_11.length then search edge 3 [661, 2857] innerCandidates118_11 3518 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_12

def innerTail118_10 : List Nat := 2791 :: innerTail118_11
def innerCandidates118_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates118_10_checked : candidates edge 4 661 2791 innerTail118_11 best.bound = innerCandidates118_10 := by
  unfold candidates
  have row : edge 2791 = SearchData.row2791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_10 : scan edge (search edge 3) 4 [661] 661 innerTail118_10 best = best := by
  rw [innerTail118_10, scan, if_neg (by decide), innerCandidates118_10_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_11
    (if 3 ≤ innerCandidates118_10.length then search edge 3 [661, 2791] innerCandidates118_10 3452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_11

def innerTail118_9 : List Nat := 2539 :: innerTail118_10
def innerCandidates118_9 : List Nat := [7159]
set_option maxRecDepth 4096 in
theorem innerCandidates118_9_checked : candidates edge 4 661 2539 innerTail118_10 best.bound = innerCandidates118_9 := by
  unfold candidates
  have row : edge 2539 = PrimePairCache.smallRow2539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_9 : scan edge (search edge 3) 4 [661] 661 innerTail118_9 best = best := by
  rw [innerTail118_9, scan, if_neg (by decide), innerCandidates118_9_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_10
    (if 3 ≤ innerCandidates118_9.length then search edge 3 [661, 2539] innerCandidates118_9 3200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_10

def innerTail118_8 : List Nat := 1993 :: innerTail118_9
def innerCandidates118_8 : List Nat := [6151, 6781]
set_option maxRecDepth 4096 in
theorem innerCandidates118_8_checked : candidates edge 4 661 1993 innerTail118_9 best.bound = innerCandidates118_8 := by
  unfold candidates
  have row : edge 1993 = PrimePairCache.smallRow1993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_8 : scan edge (search edge 3) 4 [661] 661 innerTail118_8 best = best := by
  rw [innerTail118_8, scan, if_neg (by decide), innerCandidates118_8_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_9
    (if 3 ≤ innerCandidates118_8.length then search edge 3 [661, 1993] innerCandidates118_8 2654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_9

def innerTail118_7 : List Nat := 1933 :: innerTail118_8
def innerCandidates118_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates118_7_checked : candidates edge 4 661 1933 innerTail118_8 best.bound = innerCandidates118_7 := by
  unfold candidates
  have row : edge 1933 = PrimePairCache.smallRow1933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_7 : scan edge (search edge 3) 4 [661] 661 innerTail118_7 best = best := by
  rw [innerTail118_7, scan, if_neg (by decide), innerCandidates118_7_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_8
    (if 3 ≤ innerCandidates118_7.length then search edge 3 [661, 1933] innerCandidates118_7 2594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_8

def innerTail118_6 : List Nat := 1777 :: innerTail118_7
def innerCandidates118_6 : List Nat := [2539, 6151]
set_option maxRecDepth 4096 in
theorem innerCandidates118_6_checked : candidates edge 4 661 1777 innerTail118_7 best.bound = innerCandidates118_6 := by
  unfold candidates
  have row : edge 1777 = PrimePairCache.smallRow1777.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_6 : scan edge (search edge 3) 4 [661] 661 innerTail118_6 best = best := by
  rw [innerTail118_6, scan, if_neg (by decide), innerCandidates118_6_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_7
    (if 3 ≤ innerCandidates118_6.length then search edge 3 [661, 1777] innerCandidates118_6 2438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_7

def innerTail118_5 : List Nat := 1489 :: innerTail118_6
def innerCandidates118_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates118_5_checked : candidates edge 4 661 1489 innerTail118_6 best.bound = innerCandidates118_5 := by
  unfold candidates
  have row : edge 1489 = PrimePairCache.smallRow1489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_5 : scan edge (search edge 3) 4 [661] 661 innerTail118_5 best = best := by
  rw [innerTail118_5, scan, if_neg (by decide), innerCandidates118_5_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_6
    (if 3 ≤ innerCandidates118_5.length then search edge 3 [661, 1489] innerCandidates118_5 2150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_6

def innerTail118_4 : List Nat := 1447 :: innerTail118_5
def innerCandidates118_4 : List Nat := [1993, 5641]
set_option maxRecDepth 4096 in
theorem innerCandidates118_4_checked : candidates edge 4 661 1447 innerTail118_5 best.bound = innerCandidates118_4 := by
  unfold candidates
  have row : edge 1447 = PrimePairCache.smallRow1447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_4 : scan edge (search edge 3) 4 [661] 661 innerTail118_4 best = best := by
  rw [innerTail118_4, scan, if_neg (by decide), innerCandidates118_4_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_5
    (if 3 ≤ innerCandidates118_4.length then search edge 3 [661, 1447] innerCandidates118_4 2108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_5

def innerTail118_3 : List Nat := 1237 :: innerTail118_4
def innerCandidates118_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates118_3_checked : candidates edge 4 661 1237 innerTail118_4 best.bound = innerCandidates118_3 := by
  unfold candidates
  have row : edge 1237 = PrimePairCache.smallRow1237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_3 : scan edge (search edge 3) 4 [661] 661 innerTail118_3 best = best := by
  rw [innerTail118_3, scan, if_neg (by decide), innerCandidates118_3_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_4
    (if 3 ≤ innerCandidates118_3.length then search edge 3 [661, 1237] innerCandidates118_3 1898 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_4

def innerTail118_2 : List Nat := 883 :: innerTail118_3
def innerCandidates118_2 : List Nat := [6151]
set_option maxRecDepth 4096 in
theorem innerCandidates118_2_checked : candidates edge 4 661 883 innerTail118_3 best.bound = innerCandidates118_2 := by
  unfold candidates
  have row : edge 883 = PrimePairCache.smallRow883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_2 : scan edge (search edge 3) 4 [661] 661 innerTail118_2 best = best := by
  rw [innerTail118_2, scan, if_neg (by decide), innerCandidates118_2_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_3
    (if 3 ≤ innerCandidates118_2.length then search edge 3 [661, 883] innerCandidates118_2 1544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_3

def innerTail118_1 : List Nat := 877 :: innerTail118_2
def innerCandidates118_1 : List Nat := [883, 3823, 6529]
set_option maxRecDepth 4096 in
theorem innerCandidates118_1_checked : candidates edge 4 661 877 innerTail118_2 best.bound = innerCandidates118_1 := by
  unfold candidates
  have row : edge 877 = PrimePairCache.smallRow877.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild118_1 : search edge 3 [661, 877] innerCandidates118_1 1538 best = best := by decide

theorem innerScan118_1 : scan edge (search edge 3) 4 [661] 661 innerTail118_1 best = best := by
  rw [innerTail118_1, scan, if_neg (by decide), innerCandidates118_1_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_2
    (if 3 ≤ innerCandidates118_1.length then search edge 3 [661, 877] innerCandidates118_1 1538 best else best) = best
  rw [if_pos (by decide), innerChild118_1]
  exact innerScan118_2

def innerTail118_0 : List Nat := 769 :: innerTail118_1
def innerCandidates118_0 : List Nat := [5503]
set_option maxRecDepth 4096 in
theorem innerCandidates118_0_checked : candidates edge 4 661 769 innerTail118_1 best.bound = innerCandidates118_0 := by
  unfold candidates
  have row : edge 769 = PrimePairCache.smallRow769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan118_0 : scan edge (search edge 3) 4 [661] 661 innerTail118_0 best = best := by
  rw [innerTail118_0, scan, if_neg (by decide), innerCandidates118_0_checked]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_1
    (if 3 ≤ innerCandidates118_0.length then search edge 3 [661, 769] innerCandidates118_0 1430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan118_1

theorem rootBranch118 : search edge 4 [661] rootCandidates118 661 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [661] 661 innerTail118_0 best = best
  exact innerScan118_0

def rootCandidates117 : List Nat := [947, 983, 1493, 2393, 2741, 3023, 3329, 3371, 3407, 3593, 4127, 4139, 4799, 5009, 6089, 7307, 7517, 7583, 8087]
set_option maxRecDepth 4096 in
theorem rootCandidates117_checked : candidates edge 5 0 659 tail118 best.bound = rootCandidates117 := by
  unfold candidates
  have row : edge 659 = PrimePairCache.smallRow659.contains := by funext q; rfl
  rw [row]
  decide

def innerTail117_19 : List Nat := []
theorem innerScan117_19 : scan edge (search edge 3) 4 [659] 659 innerTail117_19 best = best := rfl

def innerTail117_18 : List Nat := 8087 :: innerTail117_19
def innerCandidates117_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates117_18_checked : candidates edge 4 659 8087 innerTail117_19 best.bound = innerCandidates117_18 := by
  unfold candidates
  have row : edge 8087 = SearchData.row8087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan117_18 : scan edge (search edge 3) 4 [659] 659 innerTail117_18 best = best := by
  rw [innerTail117_18, scan, if_neg (by decide), innerCandidates117_18_checked]
  change scan edge (search edge 3) 4 [659] 659 innerTail117_19
    (if 3 ≤ innerCandidates117_18.length then search edge 3 [659, 8087] innerCandidates117_18 8746 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan117_19

def innerTail117_17 : List Nat := 7583 :: innerTail117_18
def innerCandidates117_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates117_17_checked : candidates edge 4 659 7583 innerTail117_18 best.bound = innerCandidates117_17 := by
  unfold candidates
  have row : edge 7583 = SearchData.row7583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan117_17 : scan edge (search edge 3) 4 [659] 659 innerTail117_17 best = best := by
  rw [innerTail117_17, scan, if_neg (by decide), innerCandidates117_17_checked]
  change scan edge (search edge 3) 4 [659] 659 innerTail117_18
    (if 3 ≤ innerCandidates117_17.length then search edge 3 [659, 7583] innerCandidates117_17 8242 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan117_18

def innerTail117_16 : List Nat := 7517 :: innerTail117_17
def innerCandidates117_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates117_16_checked : candidates edge 4 659 7517 innerTail117_17 best.bound = innerCandidates117_16 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan117_16 : scan edge (search edge 3) 4 [659] 659 innerTail117_16 best = best := by
  rw [innerTail117_16, scan, if_neg (by decide), innerCandidates117_16_checked]
  change scan edge (search edge 3) 4 [659] 659 innerTail117_17
    (if 3 ≤ innerCandidates117_16.length then search edge 3 [659, 7517] innerCandidates117_16 8176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan117_17

def innerTail117_15 : List Nat := 7307 :: innerTail117_16
def innerCandidates117_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates117_15_checked : candidates edge 4 659 7307 innerTail117_16 best.bound = innerCandidates117_15 := by
  unfold candidates
  have row : edge 7307 = SearchData.row7307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan117_15 : scan edge (search edge 3) 4 [659] 659 innerTail117_15 best = best := by
  rw [innerTail117_15, scan, if_neg (by decide), innerCandidates117_15_checked]
  change scan edge (search edge 3) 4 [659] 659 innerTail117_16
    (if 3 ≤ innerCandidates117_15.length then search edge 3 [659, 7307] innerCandidates117_15 7966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan117_16

def innerTail117_14 : List Nat := 6089 :: innerTail117_15
def innerCandidates117_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates117_14_checked : candidates edge 4 659 6089 innerTail117_15 best.bound = innerCandidates117_14 := by
  unfold candidates
  have row : edge 6089 = SearchData.row6089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan117_14 : scan edge (search edge 3) 4 [659] 659 innerTail117_14 best = best := by
  rw [innerTail117_14, scan, if_neg (by decide), innerCandidates117_14_checked]
  change scan edge (search edge 3) 4 [659] 659 innerTail117_15
    (if 3 ≤ innerCandidates117_14.length then search edge 3 [659, 6089] innerCandidates117_14 6748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan117_15

def innerTail117_13 : List Nat := 5009 :: innerTail117_14
def innerCandidates117_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates117_13_checked : candidates edge 4 659 5009 innerTail117_14 best.bound = innerCandidates117_13 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan117_13 : scan edge (search edge 3) 4 [659] 659 innerTail117_13 best = best := by
  rw [innerTail117_13, scan, if_neg (by decide), innerCandidates117_13_checked]
  change scan edge (search edge 3) 4 [659] 659 innerTail117_14
    (if 3 ≤ innerCandidates117_13.length then search edge 3 [659, 5009] innerCandidates117_13 5668 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan117_14

def innerTail117_12 : List Nat := 4799 :: innerTail117_13
def innerCandidates117_12 : List Nat := [7583]
set_option maxRecDepth 4096 in
theorem innerCandidates117_12_checked : candidates edge 4 659 4799 innerTail117_13 best.bound = innerCandidates117_12 := by
  unfold candidates
  have row : edge 4799 = SearchData.row4799.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan117_12 : scan edge (search edge 3) 4 [659] 659 innerTail117_12 best = best := by
  rw [innerTail117_12, scan, if_neg (by decide), innerCandidates117_12_checked]
  change scan edge (search edge 3) 4 [659] 659 innerTail117_13
    (if 3 ≤ innerCandidates117_12.length then search edge 3 [659, 4799] innerCandidates117_12 5458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan117_13

def innerTail117_11 : List Nat := 4139 :: innerTail117_12
def innerCandidates117_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates117_11_checked : candidates edge 4 659 4139 innerTail117_12 best.bound = innerCandidates117_11 := by
  unfold candidates
  have row : edge 4139 = SearchData.row4139.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan117_11 : scan edge (search edge 3) 4 [659] 659 innerTail117_11 best = best := by
  rw [innerTail117_11, scan, if_neg (by decide), innerCandidates117_11_checked]
  change scan edge (search edge 3) 4 [659] 659 innerTail117_12
    (if 3 ≤ innerCandidates117_11.length then search edge 3 [659, 4139] innerCandidates117_11 4798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan117_12

def innerTail117_10 : List Nat := 4127 :: innerTail117_11
def innerCandidates117_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates117_10_checked : candidates edge 4 659 4127 innerTail117_11 best.bound = innerCandidates117_10 := by
  unfold candidates
  have row : edge 4127 = SearchData.row4127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan117_10 : scan edge (search edge 3) 4 [659] 659 innerTail117_10 best = best := by
  rw [innerTail117_10, scan, if_neg (by decide), innerCandidates117_10_checked]
  change scan edge (search edge 3) 4 [659] 659 innerTail117_11
    (if 3 ≤ innerCandidates117_10.length then search edge 3 [659, 4127] innerCandidates117_10 4786 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan117_11

def innerTail117_9 : List Nat := 3593 :: innerTail117_10
def innerCandidates117_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates117_9_checked : candidates edge 4 659 3593 innerTail117_10 best.bound = innerCandidates117_9 := by
  unfold candidates
  have row : edge 3593 = SearchData.row3593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan117_9 : scan edge (search edge 3) 4 [659] 659 innerTail117_9 best = best := by
  rw [innerTail117_9, scan, if_neg (by decide), innerCandidates117_9_checked]
  change scan edge (search edge 3) 4 [659] 659 innerTail117_10
    (if 3 ≤ innerCandidates117_9.length then search edge 3 [659, 3593] innerCandidates117_9 4252 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan117_10

def innerTail117_8 : List Nat := 3407 :: innerTail117_9
def innerCandidates117_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates117_8_checked : candidates edge 4 659 3407 innerTail117_9 best.bound = innerCandidates117_8 := by
  unfold candidates
  have row : edge 3407 = SearchData.row3407.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan117_8 : scan edge (search edge 3) 4 [659] 659 innerTail117_8 best = best := by
  rw [innerTail117_8, scan, if_neg (by decide), innerCandidates117_8_checked]
  change scan edge (search edge 3) 4 [659] 659 innerTail117_9
    (if 3 ≤ innerCandidates117_8.length then search edge 3 [659, 3407] innerCandidates117_8 4066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan117_9

def innerTail117_7 : List Nat := 3371 :: innerTail117_8
def innerCandidates117_7 : List Nat := [4799, 5009]
set_option maxRecDepth 4096 in
theorem innerCandidates117_7_checked : candidates edge 4 659 3371 innerTail117_8 best.bound = innerCandidates117_7 := by
  unfold candidates
  have row : edge 3371 = SearchData.row3371.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan117_7 : scan edge (search edge 3) 4 [659] 659 innerTail117_7 best = best := by
  rw [innerTail117_7, scan, if_neg (by decide), innerCandidates117_7_checked]
  change scan edge (search edge 3) 4 [659] 659 innerTail117_8
    (if 3 ≤ innerCandidates117_7.length then search edge 3 [659, 3371] innerCandidates117_7 4030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan117_8

def innerTail117_6 : List Nat := 3329 :: innerTail117_7
def innerCandidates117_6 : List Nat := [4127]
set_option maxRecDepth 4096 in
theorem innerCandidates117_6_checked : candidates edge 4 659 3329 innerTail117_7 best.bound = innerCandidates117_6 := by
  unfold candidates
  have row : edge 3329 = SearchData.row3329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan117_6 : scan edge (search edge 3) 4 [659] 659 innerTail117_6 best = best := by
  rw [innerTail117_6, scan, if_neg (by decide), innerCandidates117_6_checked]
  change scan edge (search edge 3) 4 [659] 659 innerTail117_7
    (if 3 ≤ innerCandidates117_6.length then search edge 3 [659, 3329] innerCandidates117_6 3988 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan117_7

def innerTail117_5 : List Nat := 3023 :: innerTail117_6
def innerCandidates117_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates117_5_checked : candidates edge 4 659 3023 innerTail117_6 best.bound = innerCandidates117_5 := by
  unfold candidates
  have row : edge 3023 = SearchData.row3023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan117_5 : scan edge (search edge 3) 4 [659] 659 innerTail117_5 best = best := by
  rw [innerTail117_5, scan, if_neg (by decide), innerCandidates117_5_checked]
  change scan edge (search edge 3) 4 [659] 659 innerTail117_6
    (if 3 ≤ innerCandidates117_5.length then search edge 3 [659, 3023] innerCandidates117_5 3682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan117_6

def innerTail117_4 : List Nat := 2741 :: innerTail117_5
def innerCandidates117_4 : List Nat := [3329, 4799]
set_option maxRecDepth 4096 in
theorem innerCandidates117_4_checked : candidates edge 4 659 2741 innerTail117_5 best.bound = innerCandidates117_4 := by
  unfold candidates
  have row : edge 2741 = SearchData.row2741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan117_4 : scan edge (search edge 3) 4 [659] 659 innerTail117_4 best = best := by
  rw [innerTail117_4, scan, if_neg (by decide), innerCandidates117_4_checked]
  change scan edge (search edge 3) 4 [659] 659 innerTail117_5
    (if 3 ≤ innerCandidates117_4.length then search edge 3 [659, 2741] innerCandidates117_4 3400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan117_5

def innerTail117_3 : List Nat := 2393 :: innerTail117_4
def innerCandidates117_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates117_3_checked : candidates edge 4 659 2393 innerTail117_4 best.bound = innerCandidates117_3 := by
  unfold candidates
  have row : edge 2393 = PrimePairCache.smallRow2393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan117_3 : scan edge (search edge 3) 4 [659] 659 innerTail117_3 best = best := by
  rw [innerTail117_3, scan, if_neg (by decide), innerCandidates117_3_checked]
  change scan edge (search edge 3) 4 [659] 659 innerTail117_4
    (if 3 ≤ innerCandidates117_3.length then search edge 3 [659, 2393] innerCandidates117_3 3052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan117_4

def innerTail117_2 : List Nat := 1493 :: innerTail117_3
def innerCandidates117_2 : List Nat := [3329]
set_option maxRecDepth 4096 in
theorem innerCandidates117_2_checked : candidates edge 4 659 1493 innerTail117_3 best.bound = innerCandidates117_2 := by
  unfold candidates
  have row : edge 1493 = PrimePairCache.smallRow1493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan117_2 : scan edge (search edge 3) 4 [659] 659 innerTail117_2 best = best := by
  rw [innerTail117_2, scan, if_neg (by decide), innerCandidates117_2_checked]
  change scan edge (search edge 3) 4 [659] 659 innerTail117_3
    (if 3 ≤ innerCandidates117_2.length then search edge 3 [659, 1493] innerCandidates117_2 2152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan117_3

def innerTail117_1 : List Nat := 983 :: innerTail117_2
def innerCandidates117_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates117_1_checked : candidates edge 4 659 983 innerTail117_2 best.bound = innerCandidates117_1 := by
  unfold candidates
  have row : edge 983 = PrimePairCache.smallRow983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan117_1 : scan edge (search edge 3) 4 [659] 659 innerTail117_1 best = best := by
  rw [innerTail117_1, scan, if_neg (by decide), innerCandidates117_1_checked]
  change scan edge (search edge 3) 4 [659] 659 innerTail117_2
    (if 3 ≤ innerCandidates117_1.length then search edge 3 [659, 983] innerCandidates117_1 1642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan117_2

def innerTail117_0 : List Nat := 947 :: innerTail117_1
def innerCandidates117_0 : List Nat := [5009]
set_option maxRecDepth 4096 in
theorem innerCandidates117_0_checked : candidates edge 4 659 947 innerTail117_1 best.bound = innerCandidates117_0 := by
  unfold candidates
  have row : edge 947 = PrimePairCache.smallRow947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan117_0 : scan edge (search edge 3) 4 [659] 659 innerTail117_0 best = best := by
  rw [innerTail117_0, scan, if_neg (by decide), innerCandidates117_0_checked]
  change scan edge (search edge 3) 4 [659] 659 innerTail117_1
    (if 3 ≤ innerCandidates117_0.length then search edge 3 [659, 947] innerCandidates117_0 1606 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan117_1

theorem rootBranch117 : search edge 4 [659] rootCandidates117 659 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [659] 659 innerTail117_0 best = best
  exact innerScan117_0

end ProjectEulerSolutions.PrimePairUpper.Search
