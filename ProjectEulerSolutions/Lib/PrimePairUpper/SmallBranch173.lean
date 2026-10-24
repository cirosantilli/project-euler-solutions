import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates164 : List Nat := [1009, 1171, 1609, 1753, 1861, 2017, 2029, 2269, 2689, 2719, 2803, 3361, 3571, 3613, 3727, 4153, 5881, 6091, 6163, 7027, 7549, 7759, 8017]
set_option maxRecDepth 4096 in
theorem rootCandidates164_checked : candidates edge 5 0 991 tail165 best.bound = rootCandidates164 := by
  unfold candidates
  have row : edge 991 = PrimePairCache.smallRow991.contains := by funext q; rfl
  rw [row]
  decide

def innerTail164_23 : List Nat := []
theorem innerScan164_23 : scan edge (search edge 3) 4 [991] 991 innerTail164_23 best = best := rfl

def innerTail164_22 : List Nat := 8017 :: innerTail164_23
def innerCandidates164_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates164_22_checked : candidates edge 4 991 8017 innerTail164_23 best.bound = innerCandidates164_22 := by
  unfold candidates
  have row : edge 8017 = SearchData.row8017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan164_22 : scan edge (search edge 3) 4 [991] 991 innerTail164_22 best = best := by
  rw [innerTail164_22, scan, if_neg (by decide), innerCandidates164_22_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_23
    (if 3 ≤ innerCandidates164_22.length then search edge 3 [991, 8017] innerCandidates164_22 9008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan164_23

def innerTail164_21 : List Nat := 7759 :: innerTail164_22
def innerCandidates164_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates164_21_checked : candidates edge 4 991 7759 innerTail164_22 best.bound = innerCandidates164_21 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan164_21 : scan edge (search edge 3) 4 [991] 991 innerTail164_21 best = best := by
  rw [innerTail164_21, scan, if_neg (by decide), innerCandidates164_21_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_22
    (if 3 ≤ innerCandidates164_21.length then search edge 3 [991, 7759] innerCandidates164_21 8750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan164_22

def innerTail164_20 : List Nat := 7549 :: innerTail164_21
def innerCandidates164_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates164_20_checked : candidates edge 4 991 7549 innerTail164_21 best.bound = innerCandidates164_20 := by
  unfold candidates
  have row : edge 7549 = SearchData.row7549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan164_20 : scan edge (search edge 3) 4 [991] 991 innerTail164_20 best = best := by
  rw [innerTail164_20, scan, if_neg (by decide), innerCandidates164_20_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_21
    (if 3 ≤ innerCandidates164_20.length then search edge 3 [991, 7549] innerCandidates164_20 8540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan164_21

def innerTail164_19 : List Nat := 7027 :: innerTail164_20
def innerCandidates164_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates164_19_checked : candidates edge 4 991 7027 innerTail164_20 best.bound = innerCandidates164_19 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan164_19 : scan edge (search edge 3) 4 [991] 991 innerTail164_19 best = best := by
  rw [innerTail164_19, scan, if_neg (by decide), innerCandidates164_19_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_20
    (if 3 ≤ innerCandidates164_19.length then search edge 3 [991, 7027] innerCandidates164_19 8018 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan164_20

def innerTail164_18 : List Nat := 6163 :: innerTail164_19
def innerCandidates164_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates164_18_checked : candidates edge 4 991 6163 innerTail164_19 best.bound = innerCandidates164_18 := by
  unfold candidates
  have row : edge 6163 = SearchData.row6163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan164_18 : scan edge (search edge 3) 4 [991] 991 innerTail164_18 best = best := by
  rw [innerTail164_18, scan, if_neg (by decide), innerCandidates164_18_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_19
    (if 3 ≤ innerCandidates164_18.length then search edge 3 [991, 6163] innerCandidates164_18 7154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan164_19

def innerTail164_17 : List Nat := 6091 :: innerTail164_18
def innerCandidates164_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates164_17_checked : candidates edge 4 991 6091 innerTail164_18 best.bound = innerCandidates164_17 := by
  unfold candidates
  have row : edge 6091 = SearchData.row6091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan164_17 : scan edge (search edge 3) 4 [991] 991 innerTail164_17 best = best := by
  rw [innerTail164_17, scan, if_neg (by decide), innerCandidates164_17_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_18
    (if 3 ≤ innerCandidates164_17.length then search edge 3 [991, 6091] innerCandidates164_17 7082 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan164_18

def innerTail164_16 : List Nat := 5881 :: innerTail164_17
def innerCandidates164_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates164_16_checked : candidates edge 4 991 5881 innerTail164_17 best.bound = innerCandidates164_16 := by
  unfold candidates
  have row : edge 5881 = SearchData.row5881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan164_16 : scan edge (search edge 3) 4 [991] 991 innerTail164_16 best = best := by
  rw [innerTail164_16, scan, if_neg (by decide), innerCandidates164_16_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_17
    (if 3 ≤ innerCandidates164_16.length then search edge 3 [991, 5881] innerCandidates164_16 6872 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan164_17

def innerTail164_15 : List Nat := 4153 :: innerTail164_16
def innerCandidates164_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates164_15_checked : candidates edge 4 991 4153 innerTail164_16 best.bound = innerCandidates164_15 := by
  unfold candidates
  have row : edge 4153 = SearchData.row4153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan164_15 : scan edge (search edge 3) 4 [991] 991 innerTail164_15 best = best := by
  rw [innerTail164_15, scan, if_neg (by decide), innerCandidates164_15_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_16
    (if 3 ≤ innerCandidates164_15.length then search edge 3 [991, 4153] innerCandidates164_15 5144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan164_16

def innerTail164_14 : List Nat := 3727 :: innerTail164_15
def innerCandidates164_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates164_14_checked : candidates edge 4 991 3727 innerTail164_15 best.bound = innerCandidates164_14 := by
  unfold candidates
  have row : edge 3727 = SearchData.row3727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan164_14 : scan edge (search edge 3) 4 [991] 991 innerTail164_14 best = best := by
  rw [innerTail164_14, scan, if_neg (by decide), innerCandidates164_14_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_15
    (if 3 ≤ innerCandidates164_14.length then search edge 3 [991, 3727] innerCandidates164_14 4718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan164_15

def innerTail164_13 : List Nat := 3613 :: innerTail164_14
def innerCandidates164_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates164_13_checked : candidates edge 4 991 3613 innerTail164_14 best.bound = innerCandidates164_13 := by
  unfold candidates
  have row : edge 3613 = SearchData.row3613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan164_13 : scan edge (search edge 3) 4 [991] 991 innerTail164_13 best = best := by
  rw [innerTail164_13, scan, if_neg (by decide), innerCandidates164_13_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_14
    (if 3 ≤ innerCandidates164_13.length then search edge 3 [991, 3613] innerCandidates164_13 4604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan164_14

def innerTail164_12 : List Nat := 3571 :: innerTail164_13
def innerCandidates164_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates164_12_checked : candidates edge 4 991 3571 innerTail164_13 best.bound = innerCandidates164_12 := by
  unfold candidates
  have row : edge 3571 = SearchData.row3571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan164_12 : scan edge (search edge 3) 4 [991] 991 innerTail164_12 best = best := by
  rw [innerTail164_12, scan, if_neg (by decide), innerCandidates164_12_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_13
    (if 3 ≤ innerCandidates164_12.length then search edge 3 [991, 3571] innerCandidates164_12 4562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan164_13

def innerTail164_11 : List Nat := 3361 :: innerTail164_12
def innerCandidates164_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates164_11_checked : candidates edge 4 991 3361 innerTail164_12 best.bound = innerCandidates164_11 := by
  unfold candidates
  have row : edge 3361 = SearchData.row3361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan164_11 : scan edge (search edge 3) 4 [991] 991 innerTail164_11 best = best := by
  rw [innerTail164_11, scan, if_neg (by decide), innerCandidates164_11_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_12
    (if 3 ≤ innerCandidates164_11.length then search edge 3 [991, 3361] innerCandidates164_11 4352 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan164_12

def innerTail164_10 : List Nat := 2803 :: innerTail164_11
def innerCandidates164_10 : List Nat := [7549]
set_option maxRecDepth 4096 in
theorem innerCandidates164_10_checked : candidates edge 4 991 2803 innerTail164_11 best.bound = innerCandidates164_10 := by
  unfold candidates
  have row : edge 2803 = SearchData.row2803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan164_10 : scan edge (search edge 3) 4 [991] 991 innerTail164_10 best = best := by
  rw [innerTail164_10, scan, if_neg (by decide), innerCandidates164_10_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_11
    (if 3 ≤ innerCandidates164_10.length then search edge 3 [991, 2803] innerCandidates164_10 3794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan164_11

def innerTail164_9 : List Nat := 2719 :: innerTail164_10
def innerCandidates164_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates164_9_checked : candidates edge 4 991 2719 innerTail164_10 best.bound = innerCandidates164_9 := by
  unfold candidates
  have row : edge 2719 = SearchData.row2719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan164_9 : scan edge (search edge 3) 4 [991] 991 innerTail164_9 best = best := by
  rw [innerTail164_9, scan, if_neg (by decide), innerCandidates164_9_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_10
    (if 3 ≤ innerCandidates164_9.length then search edge 3 [991, 2719] innerCandidates164_9 3710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan164_10

def innerTail164_8 : List Nat := 2689 :: innerTail164_9
def innerCandidates164_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates164_8_checked : candidates edge 4 991 2689 innerTail164_9 best.bound = innerCandidates164_8 := by
  unfold candidates
  have row : edge 2689 = SearchData.row2689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan164_8 : scan edge (search edge 3) 4 [991] 991 innerTail164_8 best = best := by
  rw [innerTail164_8, scan, if_neg (by decide), innerCandidates164_8_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_9
    (if 3 ≤ innerCandidates164_8.length then search edge 3 [991, 2689] innerCandidates164_8 3680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan164_9

def innerTail164_7 : List Nat := 2269 :: innerTail164_8
def innerCandidates164_7 : List Nat := [3613, 7027]
set_option maxRecDepth 4096 in
theorem innerCandidates164_7_checked : candidates edge 4 991 2269 innerTail164_8 best.bound = innerCandidates164_7 := by
  unfold candidates
  have row : edge 2269 = PrimePairCache.smallRow2269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan164_7 : scan edge (search edge 3) 4 [991] 991 innerTail164_7 best = best := by
  rw [innerTail164_7, scan, if_neg (by decide), innerCandidates164_7_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_8
    (if 3 ≤ innerCandidates164_7.length then search edge 3 [991, 2269] innerCandidates164_7 3260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan164_8

def innerTail164_6 : List Nat := 2029 :: innerTail164_7
def innerCandidates164_6 : List Nat := [7027]
set_option maxRecDepth 4096 in
theorem innerCandidates164_6_checked : candidates edge 4 991 2029 innerTail164_7 best.bound = innerCandidates164_6 := by
  unfold candidates
  have row : edge 2029 = PrimePairCache.smallRow2029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan164_6 : scan edge (search edge 3) 4 [991] 991 innerTail164_6 best = best := by
  rw [innerTail164_6, scan, if_neg (by decide), innerCandidates164_6_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_7
    (if 3 ≤ innerCandidates164_6.length then search edge 3 [991, 2029] innerCandidates164_6 3020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan164_7

def innerTail164_5 : List Nat := 2017 :: innerTail164_6
def innerCandidates164_5 : List Nat := [2689]
set_option maxRecDepth 4096 in
theorem innerCandidates164_5_checked : candidates edge 4 991 2017 innerTail164_6 best.bound = innerCandidates164_5 := by
  unfold candidates
  have row : edge 2017 = PrimePairCache.smallRow2017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan164_5 : scan edge (search edge 3) 4 [991] 991 innerTail164_5 best = best := by
  rw [innerTail164_5, scan, if_neg (by decide), innerCandidates164_5_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_6
    (if 3 ≤ innerCandidates164_5.length then search edge 3 [991, 2017] innerCandidates164_5 3008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan164_6

def innerTail164_4 : List Nat := 1861 :: innerTail164_5
def innerCandidates164_4 : List Nat := [5881]
set_option maxRecDepth 4096 in
theorem innerCandidates164_4_checked : candidates edge 4 991 1861 innerTail164_5 best.bound = innerCandidates164_4 := by
  unfold candidates
  have row : edge 1861 = PrimePairCache.smallRow1861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan164_4 : scan edge (search edge 3) 4 [991] 991 innerTail164_4 best = best := by
  rw [innerTail164_4, scan, if_neg (by decide), innerCandidates164_4_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_5
    (if 3 ≤ innerCandidates164_4.length then search edge 3 [991, 1861] innerCandidates164_4 2852 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan164_5

def innerTail164_3 : List Nat := 1753 :: innerTail164_4
def innerCandidates164_3 : List Nat := [7549]
set_option maxRecDepth 4096 in
theorem innerCandidates164_3_checked : candidates edge 4 991 1753 innerTail164_4 best.bound = innerCandidates164_3 := by
  unfold candidates
  have row : edge 1753 = PrimePairCache.smallRow1753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan164_3 : scan edge (search edge 3) 4 [991] 991 innerTail164_3 best = best := by
  rw [innerTail164_3, scan, if_neg (by decide), innerCandidates164_3_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_4
    (if 3 ≤ innerCandidates164_3.length then search edge 3 [991, 1753] innerCandidates164_3 2744 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan164_4

def innerTail164_2 : List Nat := 1609 :: innerTail164_3
def innerCandidates164_2 : List Nat := [2017, 2803, 3613]
set_option maxRecDepth 4096 in
theorem innerCandidates164_2_checked : candidates edge 4 991 1609 innerTail164_3 best.bound = innerCandidates164_2 := by
  unfold candidates
  have row : edge 1609 = PrimePairCache.smallRow1609.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild164_2 : search edge 3 [991, 1609] innerCandidates164_2 2600 best = best := by decide

theorem innerScan164_2 : scan edge (search edge 3) 4 [991] 991 innerTail164_2 best = best := by
  rw [innerTail164_2, scan, if_neg (by decide), innerCandidates164_2_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_3
    (if 3 ≤ innerCandidates164_2.length then search edge 3 [991, 1609] innerCandidates164_2 2600 best else best) = best
  rw [if_pos (by decide), innerChild164_2]
  exact innerScan164_3

def innerTail164_1 : List Nat := 1171 :: innerTail164_2
def innerCandidates164_1 : List Nat := [5881]
set_option maxRecDepth 4096 in
theorem innerCandidates164_1_checked : candidates edge 4 991 1171 innerTail164_2 best.bound = innerCandidates164_1 := by
  unfold candidates
  have row : edge 1171 = PrimePairCache.smallRow1171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan164_1 : scan edge (search edge 3) 4 [991] 991 innerTail164_1 best = best := by
  rw [innerTail164_1, scan, if_neg (by decide), innerCandidates164_1_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_2
    (if 3 ≤ innerCandidates164_1.length then search edge 3 [991, 1171] innerCandidates164_1 2162 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan164_2

def innerTail164_0 : List Nat := 1009 :: innerTail164_1
def innerCandidates164_0 : List Nat := [3613, 7027]
set_option maxRecDepth 4096 in
theorem innerCandidates164_0_checked : candidates edge 4 991 1009 innerTail164_1 best.bound = innerCandidates164_0 := by
  unfold candidates
  have row : edge 1009 = PrimePairCache.smallRow1009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan164_0 : scan edge (search edge 3) 4 [991] 991 innerTail164_0 best = best := by
  rw [innerTail164_0, scan, if_neg (by decide), innerCandidates164_0_checked]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_1
    (if 3 ≤ innerCandidates164_0.length then search edge 3 [991, 1009] innerCandidates164_0 2000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan164_1

theorem rootBranch164 : search edge 4 [991] rootCandidates164 991 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [991] 991 innerTail164_0 best = best
  exact innerScan164_0

def rootCandidates163 : List Nat := [1091, 1259, 1637, 2129, 2267, 2399, 3221, 3347, 3617, 4073, 4871, 5171, 6389, 6719, 6917, 7109, 7151, 7481]
set_option maxRecDepth 4096 in
theorem rootCandidates163_checked : candidates edge 5 0 983 tail164 best.bound = rootCandidates163 := by
  unfold candidates
  have row : edge 983 = PrimePairCache.smallRow983.contains := by funext q; rfl
  rw [row]
  decide

def innerTail163_18 : List Nat := []
theorem innerScan163_18 : scan edge (search edge 3) 4 [983] 983 innerTail163_18 best = best := rfl

def innerTail163_17 : List Nat := 7481 :: innerTail163_18
def innerCandidates163_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates163_17_checked : candidates edge 4 983 7481 innerTail163_18 best.bound = innerCandidates163_17 := by
  unfold candidates
  have row : edge 7481 = SearchData.row7481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan163_17 : scan edge (search edge 3) 4 [983] 983 innerTail163_17 best = best := by
  rw [innerTail163_17, scan, if_neg (by decide), innerCandidates163_17_checked]
  change scan edge (search edge 3) 4 [983] 983 innerTail163_18
    (if 3 ≤ innerCandidates163_17.length then search edge 3 [983, 7481] innerCandidates163_17 8464 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan163_18

def innerTail163_16 : List Nat := 7151 :: innerTail163_17
def innerCandidates163_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates163_16_checked : candidates edge 4 983 7151 innerTail163_17 best.bound = innerCandidates163_16 := by
  unfold candidates
  have row : edge 7151 = SearchData.row7151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan163_16 : scan edge (search edge 3) 4 [983] 983 innerTail163_16 best = best := by
  rw [innerTail163_16, scan, if_neg (by decide), innerCandidates163_16_checked]
  change scan edge (search edge 3) 4 [983] 983 innerTail163_17
    (if 3 ≤ innerCandidates163_16.length then search edge 3 [983, 7151] innerCandidates163_16 8134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan163_17

def innerTail163_15 : List Nat := 7109 :: innerTail163_16
def innerCandidates163_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates163_15_checked : candidates edge 4 983 7109 innerTail163_16 best.bound = innerCandidates163_15 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan163_15 : scan edge (search edge 3) 4 [983] 983 innerTail163_15 best = best := by
  rw [innerTail163_15, scan, if_neg (by decide), innerCandidates163_15_checked]
  change scan edge (search edge 3) 4 [983] 983 innerTail163_16
    (if 3 ≤ innerCandidates163_15.length then search edge 3 [983, 7109] innerCandidates163_15 8092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan163_16

def innerTail163_14 : List Nat := 6917 :: innerTail163_15
def innerCandidates163_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates163_14_checked : candidates edge 4 983 6917 innerTail163_15 best.bound = innerCandidates163_14 := by
  unfold candidates
  have row : edge 6917 = SearchData.row6917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan163_14 : scan edge (search edge 3) 4 [983] 983 innerTail163_14 best = best := by
  rw [innerTail163_14, scan, if_neg (by decide), innerCandidates163_14_checked]
  change scan edge (search edge 3) 4 [983] 983 innerTail163_15
    (if 3 ≤ innerCandidates163_14.length then search edge 3 [983, 6917] innerCandidates163_14 7900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan163_15

def innerTail163_13 : List Nat := 6719 :: innerTail163_14
def innerCandidates163_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates163_13_checked : candidates edge 4 983 6719 innerTail163_14 best.bound = innerCandidates163_13 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan163_13 : scan edge (search edge 3) 4 [983] 983 innerTail163_13 best = best := by
  rw [innerTail163_13, scan, if_neg (by decide), innerCandidates163_13_checked]
  change scan edge (search edge 3) 4 [983] 983 innerTail163_14
    (if 3 ≤ innerCandidates163_13.length then search edge 3 [983, 6719] innerCandidates163_13 7702 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan163_14

def innerTail163_12 : List Nat := 6389 :: innerTail163_13
def innerCandidates163_12 : List Nat := [7481]
set_option maxRecDepth 4096 in
theorem innerCandidates163_12_checked : candidates edge 4 983 6389 innerTail163_13 best.bound = innerCandidates163_12 := by
  unfold candidates
  have row : edge 6389 = SearchData.row6389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan163_12 : scan edge (search edge 3) 4 [983] 983 innerTail163_12 best = best := by
  rw [innerTail163_12, scan, if_neg (by decide), innerCandidates163_12_checked]
  change scan edge (search edge 3) 4 [983] 983 innerTail163_13
    (if 3 ≤ innerCandidates163_12.length then search edge 3 [983, 6389] innerCandidates163_12 7372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan163_13

def innerTail163_11 : List Nat := 5171 :: innerTail163_12
def innerCandidates163_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates163_11_checked : candidates edge 4 983 5171 innerTail163_12 best.bound = innerCandidates163_11 := by
  unfold candidates
  have row : edge 5171 = SearchData.row5171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan163_11 : scan edge (search edge 3) 4 [983] 983 innerTail163_11 best = best := by
  rw [innerTail163_11, scan, if_neg (by decide), innerCandidates163_11_checked]
  change scan edge (search edge 3) 4 [983] 983 innerTail163_12
    (if 3 ≤ innerCandidates163_11.length then search edge 3 [983, 5171] innerCandidates163_11 6154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan163_12

def innerTail163_10 : List Nat := 4871 :: innerTail163_11
def innerCandidates163_10 : List Nat := [6389, 6719]
set_option maxRecDepth 4096 in
theorem innerCandidates163_10_checked : candidates edge 4 983 4871 innerTail163_11 best.bound = innerCandidates163_10 := by
  unfold candidates
  have row : edge 4871 = SearchData.row4871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan163_10 : scan edge (search edge 3) 4 [983] 983 innerTail163_10 best = best := by
  rw [innerTail163_10, scan, if_neg (by decide), innerCandidates163_10_checked]
  change scan edge (search edge 3) 4 [983] 983 innerTail163_11
    (if 3 ≤ innerCandidates163_10.length then search edge 3 [983, 4871] innerCandidates163_10 5854 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan163_11

def innerTail163_9 : List Nat := 4073 :: innerTail163_10
def innerCandidates163_9 : List Nat := [6389]
set_option maxRecDepth 4096 in
theorem innerCandidates163_9_checked : candidates edge 4 983 4073 innerTail163_10 best.bound = innerCandidates163_9 := by
  unfold candidates
  have row : edge 4073 = SearchData.row4073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan163_9 : scan edge (search edge 3) 4 [983] 983 innerTail163_9 best = best := by
  rw [innerTail163_9, scan, if_neg (by decide), innerCandidates163_9_checked]
  change scan edge (search edge 3) 4 [983] 983 innerTail163_10
    (if 3 ≤ innerCandidates163_9.length then search edge 3 [983, 4073] innerCandidates163_9 5056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan163_10

def innerTail163_8 : List Nat := 3617 :: innerTail163_9
def innerCandidates163_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates163_8_checked : candidates edge 4 983 3617 innerTail163_9 best.bound = innerCandidates163_8 := by
  unfold candidates
  have row : edge 3617 = SearchData.row3617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan163_8 : scan edge (search edge 3) 4 [983] 983 innerTail163_8 best = best := by
  rw [innerTail163_8, scan, if_neg (by decide), innerCandidates163_8_checked]
  change scan edge (search edge 3) 4 [983] 983 innerTail163_9
    (if 3 ≤ innerCandidates163_8.length then search edge 3 [983, 3617] innerCandidates163_8 4600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan163_9

def innerTail163_7 : List Nat := 3347 :: innerTail163_8
def innerCandidates163_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates163_7_checked : candidates edge 4 983 3347 innerTail163_8 best.bound = innerCandidates163_7 := by
  unfold candidates
  have row : edge 3347 = SearchData.row3347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan163_7 : scan edge (search edge 3) 4 [983] 983 innerTail163_7 best = best := by
  rw [innerTail163_7, scan, if_neg (by decide), innerCandidates163_7_checked]
  change scan edge (search edge 3) 4 [983] 983 innerTail163_8
    (if 3 ≤ innerCandidates163_7.length then search edge 3 [983, 3347] innerCandidates163_7 4330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan163_8

def innerTail163_6 : List Nat := 3221 :: innerTail163_7
def innerCandidates163_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates163_6_checked : candidates edge 4 983 3221 innerTail163_7 best.bound = innerCandidates163_6 := by
  unfold candidates
  have row : edge 3221 = SearchData.row3221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan163_6 : scan edge (search edge 3) 4 [983] 983 innerTail163_6 best = best := by
  rw [innerTail163_6, scan, if_neg (by decide), innerCandidates163_6_checked]
  change scan edge (search edge 3) 4 [983] 983 innerTail163_7
    (if 3 ≤ innerCandidates163_6.length then search edge 3 [983, 3221] innerCandidates163_6 4204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan163_7

def innerTail163_5 : List Nat := 2399 :: innerTail163_6
def innerCandidates163_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates163_5_checked : candidates edge 4 983 2399 innerTail163_6 best.bound = innerCandidates163_5 := by
  unfold candidates
  have row : edge 2399 = PrimePairCache.smallRow2399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan163_5 : scan edge (search edge 3) 4 [983] 983 innerTail163_5 best = best := by
  rw [innerTail163_5, scan, if_neg (by decide), innerCandidates163_5_checked]
  change scan edge (search edge 3) 4 [983] 983 innerTail163_6
    (if 3 ≤ innerCandidates163_5.length then search edge 3 [983, 2399] innerCandidates163_5 3382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan163_6

def innerTail163_4 : List Nat := 2267 :: innerTail163_5
def innerCandidates163_4 : List Nat := [6389, 6917]
set_option maxRecDepth 4096 in
theorem innerCandidates163_4_checked : candidates edge 4 983 2267 innerTail163_5 best.bound = innerCandidates163_4 := by
  unfold candidates
  have row : edge 2267 = PrimePairCache.smallRow2267.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan163_4 : scan edge (search edge 3) 4 [983] 983 innerTail163_4 best = best := by
  rw [innerTail163_4, scan, if_neg (by decide), innerCandidates163_4_checked]
  change scan edge (search edge 3) 4 [983] 983 innerTail163_5
    (if 3 ≤ innerCandidates163_4.length then search edge 3 [983, 2267] innerCandidates163_4 3250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan163_5

def innerTail163_3 : List Nat := 2129 :: innerTail163_4
def innerCandidates163_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates163_3_checked : candidates edge 4 983 2129 innerTail163_4 best.bound = innerCandidates163_3 := by
  unfold candidates
  have row : edge 2129 = PrimePairCache.smallRow2129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan163_3 : scan edge (search edge 3) 4 [983] 983 innerTail163_3 best = best := by
  rw [innerTail163_3, scan, if_neg (by decide), innerCandidates163_3_checked]
  change scan edge (search edge 3) 4 [983] 983 innerTail163_4
    (if 3 ≤ innerCandidates163_3.length then search edge 3 [983, 2129] innerCandidates163_3 3112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan163_4

def innerTail163_2 : List Nat := 1637 :: innerTail163_3
def innerCandidates163_2 : List Nat := [4073]
set_option maxRecDepth 4096 in
theorem innerCandidates163_2_checked : candidates edge 4 983 1637 innerTail163_3 best.bound = innerCandidates163_2 := by
  unfold candidates
  have row : edge 1637 = PrimePairCache.smallRow1637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan163_2 : scan edge (search edge 3) 4 [983] 983 innerTail163_2 best = best := by
  rw [innerTail163_2, scan, if_neg (by decide), innerCandidates163_2_checked]
  change scan edge (search edge 3) 4 [983] 983 innerTail163_3
    (if 3 ≤ innerCandidates163_2.length then search edge 3 [983, 1637] innerCandidates163_2 2620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan163_3

def innerTail163_1 : List Nat := 1259 :: innerTail163_2
def innerCandidates163_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates163_1_checked : candidates edge 4 983 1259 innerTail163_2 best.bound = innerCandidates163_1 := by
  unfold candidates
  have row : edge 1259 = PrimePairCache.smallRow1259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan163_1 : scan edge (search edge 3) 4 [983] 983 innerTail163_1 best = best := by
  rw [innerTail163_1, scan, if_neg (by decide), innerCandidates163_1_checked]
  change scan edge (search edge 3) 4 [983] 983 innerTail163_2
    (if 3 ≤ innerCandidates163_1.length then search edge 3 [983, 1259] innerCandidates163_1 2242 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan163_2

def innerTail163_0 : List Nat := 1091 :: innerTail163_1
def innerCandidates163_0 : List Nat := [2267]
set_option maxRecDepth 4096 in
theorem innerCandidates163_0_checked : candidates edge 4 983 1091 innerTail163_1 best.bound = innerCandidates163_0 := by
  unfold candidates
  have row : edge 1091 = PrimePairCache.smallRow1091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan163_0 : scan edge (search edge 3) 4 [983] 983 innerTail163_0 best = best := by
  rw [innerTail163_0, scan, if_neg (by decide), innerCandidates163_0_checked]
  change scan edge (search edge 3) 4 [983] 983 innerTail163_1
    (if 3 ≤ innerCandidates163_0.length then search edge 3 [983, 1091] innerCandidates163_0 2074 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan163_1

theorem rootBranch163 : search edge 4 [983] rootCandidates163 983 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [983] 983 innerTail163_0 best = best
  exact innerScan163_0

def rootCandidates162 : List Nat := [1217, 1259, 1409, 2039, 2129, 2141, 2753, 3557, 3593, 3767, 3851, 4049, 5021, 5039, 7853, 8297]
set_option maxRecDepth 4096 in
theorem rootCandidates162_checked : candidates edge 5 0 977 tail163 best.bound = rootCandidates162 := by
  unfold candidates
  have row : edge 977 = PrimePairCache.smallRow977.contains := by funext q; rfl
  rw [row]
  decide

def innerTail162_16 : List Nat := []
theorem innerScan162_16 : scan edge (search edge 3) 4 [977] 977 innerTail162_16 best = best := rfl

def innerTail162_15 : List Nat := 8297 :: innerTail162_16
def innerCandidates162_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates162_15_checked : candidates edge 4 977 8297 innerTail162_16 best.bound = innerCandidates162_15 := by
  unfold candidates
  have row : edge 8297 = SearchData.row8297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan162_15 : scan edge (search edge 3) 4 [977] 977 innerTail162_15 best = best := by
  rw [innerTail162_15, scan, if_neg (by decide), innerCandidates162_15_checked]
  change scan edge (search edge 3) 4 [977] 977 innerTail162_16
    (if 3 ≤ innerCandidates162_15.length then search edge 3 [977, 8297] innerCandidates162_15 9274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan162_16

def innerTail162_14 : List Nat := 7853 :: innerTail162_15
def innerCandidates162_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates162_14_checked : candidates edge 4 977 7853 innerTail162_15 best.bound = innerCandidates162_14 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan162_14 : scan edge (search edge 3) 4 [977] 977 innerTail162_14 best = best := by
  rw [innerTail162_14, scan, if_neg (by decide), innerCandidates162_14_checked]
  change scan edge (search edge 3) 4 [977] 977 innerTail162_15
    (if 3 ≤ innerCandidates162_14.length then search edge 3 [977, 7853] innerCandidates162_14 8830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan162_15

def innerTail162_13 : List Nat := 5039 :: innerTail162_14
def innerCandidates162_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates162_13_checked : candidates edge 4 977 5039 innerTail162_14 best.bound = innerCandidates162_13 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan162_13 : scan edge (search edge 3) 4 [977] 977 innerTail162_13 best = best := by
  rw [innerTail162_13, scan, if_neg (by decide), innerCandidates162_13_checked]
  change scan edge (search edge 3) 4 [977] 977 innerTail162_14
    (if 3 ≤ innerCandidates162_13.length then search edge 3 [977, 5039] innerCandidates162_13 6016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan162_14

def innerTail162_12 : List Nat := 5021 :: innerTail162_13
def innerCandidates162_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates162_12_checked : candidates edge 4 977 5021 innerTail162_13 best.bound = innerCandidates162_12 := by
  unfold candidates
  have row : edge 5021 = SearchData.row5021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan162_12 : scan edge (search edge 3) 4 [977] 977 innerTail162_12 best = best := by
  rw [innerTail162_12, scan, if_neg (by decide), innerCandidates162_12_checked]
  change scan edge (search edge 3) 4 [977] 977 innerTail162_13
    (if 3 ≤ innerCandidates162_12.length then search edge 3 [977, 5021] innerCandidates162_12 5998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan162_13

def innerTail162_11 : List Nat := 4049 :: innerTail162_12
def innerCandidates162_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates162_11_checked : candidates edge 4 977 4049 innerTail162_12 best.bound = innerCandidates162_11 := by
  unfold candidates
  have row : edge 4049 = SearchData.row4049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan162_11 : scan edge (search edge 3) 4 [977] 977 innerTail162_11 best = best := by
  rw [innerTail162_11, scan, if_neg (by decide), innerCandidates162_11_checked]
  change scan edge (search edge 3) 4 [977] 977 innerTail162_12
    (if 3 ≤ innerCandidates162_11.length then search edge 3 [977, 4049] innerCandidates162_11 5026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan162_12

def innerTail162_10 : List Nat := 3851 :: innerTail162_11
def innerCandidates162_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates162_10_checked : candidates edge 4 977 3851 innerTail162_11 best.bound = innerCandidates162_10 := by
  unfold candidates
  have row : edge 3851 = SearchData.row3851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan162_10 : scan edge (search edge 3) 4 [977] 977 innerTail162_10 best = best := by
  rw [innerTail162_10, scan, if_neg (by decide), innerCandidates162_10_checked]
  change scan edge (search edge 3) 4 [977] 977 innerTail162_11
    (if 3 ≤ innerCandidates162_10.length then search edge 3 [977, 3851] innerCandidates162_10 4828 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan162_11

def innerTail162_9 : List Nat := 3767 :: innerTail162_10
def innerCandidates162_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates162_9_checked : candidates edge 4 977 3767 innerTail162_10 best.bound = innerCandidates162_9 := by
  unfold candidates
  have row : edge 3767 = SearchData.row3767.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan162_9 : scan edge (search edge 3) 4 [977] 977 innerTail162_9 best = best := by
  rw [innerTail162_9, scan, if_neg (by decide), innerCandidates162_9_checked]
  change scan edge (search edge 3) 4 [977] 977 innerTail162_10
    (if 3 ≤ innerCandidates162_9.length then search edge 3 [977, 3767] innerCandidates162_9 4744 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan162_10

def innerTail162_8 : List Nat := 3593 :: innerTail162_9
def innerCandidates162_8 : List Nat := [5021, 8297]
set_option maxRecDepth 4096 in
theorem innerCandidates162_8_checked : candidates edge 4 977 3593 innerTail162_9 best.bound = innerCandidates162_8 := by
  unfold candidates
  have row : edge 3593 = SearchData.row3593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan162_8 : scan edge (search edge 3) 4 [977] 977 innerTail162_8 best = best := by
  rw [innerTail162_8, scan, if_neg (by decide), innerCandidates162_8_checked]
  change scan edge (search edge 3) 4 [977] 977 innerTail162_9
    (if 3 ≤ innerCandidates162_8.length then search edge 3 [977, 3593] innerCandidates162_8 4570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan162_9

def innerTail162_7 : List Nat := 3557 :: innerTail162_8
def innerCandidates162_7 : List Nat := [5021]
set_option maxRecDepth 4096 in
theorem innerCandidates162_7_checked : candidates edge 4 977 3557 innerTail162_8 best.bound = innerCandidates162_7 := by
  unfold candidates
  have row : edge 3557 = SearchData.row3557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan162_7 : scan edge (search edge 3) 4 [977] 977 innerTail162_7 best = best := by
  rw [innerTail162_7, scan, if_neg (by decide), innerCandidates162_7_checked]
  change scan edge (search edge 3) 4 [977] 977 innerTail162_8
    (if 3 ≤ innerCandidates162_7.length then search edge 3 [977, 3557] innerCandidates162_7 4534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan162_8

def innerTail162_6 : List Nat := 2753 :: innerTail162_7
def innerCandidates162_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates162_6_checked : candidates edge 4 977 2753 innerTail162_7 best.bound = innerCandidates162_6 := by
  unfold candidates
  have row : edge 2753 = SearchData.row2753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan162_6 : scan edge (search edge 3) 4 [977] 977 innerTail162_6 best = best := by
  rw [innerTail162_6, scan, if_neg (by decide), innerCandidates162_6_checked]
  change scan edge (search edge 3) 4 [977] 977 innerTail162_7
    (if 3 ≤ innerCandidates162_6.length then search edge 3 [977, 2753] innerCandidates162_6 3730 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan162_7

def innerTail162_5 : List Nat := 2141 :: innerTail162_6
def innerCandidates162_5 : List Nat := [4049]
set_option maxRecDepth 4096 in
theorem innerCandidates162_5_checked : candidates edge 4 977 2141 innerTail162_6 best.bound = innerCandidates162_5 := by
  unfold candidates
  have row : edge 2141 = PrimePairCache.smallRow2141.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan162_5 : scan edge (search edge 3) 4 [977] 977 innerTail162_5 best = best := by
  rw [innerTail162_5, scan, if_neg (by decide), innerCandidates162_5_checked]
  change scan edge (search edge 3) 4 [977] 977 innerTail162_6
    (if 3 ≤ innerCandidates162_5.length then search edge 3 [977, 2141] innerCandidates162_5 3118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan162_6

def innerTail162_4 : List Nat := 2129 :: innerTail162_5
def innerCandidates162_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates162_4_checked : candidates edge 4 977 2129 innerTail162_5 best.bound = innerCandidates162_4 := by
  unfold candidates
  have row : edge 2129 = PrimePairCache.smallRow2129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan162_4 : scan edge (search edge 3) 4 [977] 977 innerTail162_4 best = best := by
  rw [innerTail162_4, scan, if_neg (by decide), innerCandidates162_4_checked]
  change scan edge (search edge 3) 4 [977] 977 innerTail162_5
    (if 3 ≤ innerCandidates162_4.length then search edge 3 [977, 2129] innerCandidates162_4 3106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan162_5

def innerTail162_3 : List Nat := 2039 :: innerTail162_4
def innerCandidates162_3 : List Nat := [3767]
set_option maxRecDepth 4096 in
theorem innerCandidates162_3_checked : candidates edge 4 977 2039 innerTail162_4 best.bound = innerCandidates162_3 := by
  unfold candidates
  have row : edge 2039 = PrimePairCache.smallRow2039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan162_3 : scan edge (search edge 3) 4 [977] 977 innerTail162_3 best = best := by
  rw [innerTail162_3, scan, if_neg (by decide), innerCandidates162_3_checked]
  change scan edge (search edge 3) 4 [977] 977 innerTail162_4
    (if 3 ≤ innerCandidates162_3.length then search edge 3 [977, 2039] innerCandidates162_3 3016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan162_4

def innerTail162_2 : List Nat := 1409 :: innerTail162_3
def innerCandidates162_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates162_2_checked : candidates edge 4 977 1409 innerTail162_3 best.bound = innerCandidates162_2 := by
  unfold candidates
  have row : edge 1409 = PrimePairCache.smallRow1409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan162_2 : scan edge (search edge 3) 4 [977] 977 innerTail162_2 best = best := by
  rw [innerTail162_2, scan, if_neg (by decide), innerCandidates162_2_checked]
  change scan edge (search edge 3) 4 [977] 977 innerTail162_3
    (if 3 ≤ innerCandidates162_2.length then search edge 3 [977, 1409] innerCandidates162_2 2386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan162_3

def innerTail162_1 : List Nat := 1259 :: innerTail162_2
def innerCandidates162_1 : List Nat := [5039]
set_option maxRecDepth 4096 in
theorem innerCandidates162_1_checked : candidates edge 4 977 1259 innerTail162_2 best.bound = innerCandidates162_1 := by
  unfold candidates
  have row : edge 1259 = PrimePairCache.smallRow1259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan162_1 : scan edge (search edge 3) 4 [977] 977 innerTail162_1 best = best := by
  rw [innerTail162_1, scan, if_neg (by decide), innerCandidates162_1_checked]
  change scan edge (search edge 3) 4 [977] 977 innerTail162_2
    (if 3 ≤ innerCandidates162_1.length then search edge 3 [977, 1259] innerCandidates162_1 2236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan162_2

def innerTail162_0 : List Nat := 1217 :: innerTail162_1
def innerCandidates162_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates162_0_checked : candidates edge 4 977 1217 innerTail162_1 best.bound = innerCandidates162_0 := by
  unfold candidates
  have row : edge 1217 = PrimePairCache.smallRow1217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan162_0 : scan edge (search edge 3) 4 [977] 977 innerTail162_0 best = best := by
  rw [innerTail162_0, scan, if_neg (by decide), innerCandidates162_0_checked]
  change scan edge (search edge 3) 4 [977] 977 innerTail162_1
    (if 3 ≤ innerCandidates162_0.length then search edge 3 [977, 1217] innerCandidates162_0 2194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan162_1

theorem rootBranch162 : search edge 4 [977] rootCandidates162 977 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [977] 977 innerTail162_0 best = best
  exact innerScan162_0

end ProjectEulerSolutions.PrimePairUpper.Search
