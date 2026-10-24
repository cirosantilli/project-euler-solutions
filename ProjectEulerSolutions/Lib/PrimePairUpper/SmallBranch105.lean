import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates375 : List Nat := [2617, 2713, 2719, 2887, 2971, 3169, 3307, 3373, 3709, 3847, 5011, 5101, 5281, 5443, 5683, 7027, 7039, 7573, 7753, 7927, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates375_checked : candidates edge 5 0 2593 tail376 best.bound = rootCandidates375 := by
  unfold candidates
  have row : edge 2593 = PrimePairCache.smallRow2593.contains := by funext q; rfl
  rw [row]
  decide

def innerTail375_21 : List Nat := []
theorem innerScan375_21 : scan edge (search edge 3) 4 [2593] 2593 innerTail375_21 best = best := rfl

def innerTail375_20 : List Nat := 8293 :: innerTail375_21
def innerCandidates375_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates375_20_checked : candidates edge 4 2593 8293 innerTail375_21 best.bound = innerCandidates375_20 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan375_20 : scan edge (search edge 3) 4 [2593] 2593 innerTail375_20 best = best := by
  rw [innerTail375_20, scan, if_neg (by decide), innerCandidates375_20_checked]
  change scan edge (search edge 3) 4 [2593] 2593 innerTail375_21
    (if 3 ≤ innerCandidates375_20.length then search edge 3 [2593, 8293] innerCandidates375_20 10886 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan375_21

def innerTail375_19 : List Nat := 7927 :: innerTail375_20
def innerCandidates375_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates375_19_checked : candidates edge 4 2593 7927 innerTail375_20 best.bound = innerCandidates375_19 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan375_19 : scan edge (search edge 3) 4 [2593] 2593 innerTail375_19 best = best := by
  rw [innerTail375_19, scan, if_neg (by decide), innerCandidates375_19_checked]
  change scan edge (search edge 3) 4 [2593] 2593 innerTail375_20
    (if 3 ≤ innerCandidates375_19.length then search edge 3 [2593, 7927] innerCandidates375_19 10520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan375_20

def innerTail375_18 : List Nat := 7753 :: innerTail375_19
def innerCandidates375_18 : List Nat := [7927]
set_option maxRecDepth 4096 in
theorem innerCandidates375_18_checked : candidates edge 4 2593 7753 innerTail375_19 best.bound = innerCandidates375_18 := by
  unfold candidates
  have row : edge 7753 = SearchData.row7753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan375_18 : scan edge (search edge 3) 4 [2593] 2593 innerTail375_18 best = best := by
  rw [innerTail375_18, scan, if_neg (by decide), innerCandidates375_18_checked]
  change scan edge (search edge 3) 4 [2593] 2593 innerTail375_19
    (if 3 ≤ innerCandidates375_18.length then search edge 3 [2593, 7753] innerCandidates375_18 10346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan375_19

def innerTail375_17 : List Nat := 7573 :: innerTail375_18
def innerCandidates375_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates375_17_checked : candidates edge 4 2593 7573 innerTail375_18 best.bound = innerCandidates375_17 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan375_17 : scan edge (search edge 3) 4 [2593] 2593 innerTail375_17 best = best := by
  rw [innerTail375_17, scan, if_neg (by decide), innerCandidates375_17_checked]
  change scan edge (search edge 3) 4 [2593] 2593 innerTail375_18
    (if 3 ≤ innerCandidates375_17.length then search edge 3 [2593, 7573] innerCandidates375_17 10166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan375_18

def innerTail375_16 : List Nat := 7039 :: innerTail375_17
def innerCandidates375_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates375_16_checked : candidates edge 4 2593 7039 innerTail375_17 best.bound = innerCandidates375_16 := by
  unfold candidates
  have row : edge 7039 = SearchData.row7039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan375_16 : scan edge (search edge 3) 4 [2593] 2593 innerTail375_16 best = best := by
  rw [innerTail375_16, scan, if_neg (by decide), innerCandidates375_16_checked]
  change scan edge (search edge 3) 4 [2593] 2593 innerTail375_17
    (if 3 ≤ innerCandidates375_16.length then search edge 3 [2593, 7039] innerCandidates375_16 9632 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan375_17

def innerTail375_15 : List Nat := 7027 :: innerTail375_16
def innerCandidates375_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates375_15_checked : candidates edge 4 2593 7027 innerTail375_16 best.bound = innerCandidates375_15 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan375_15 : scan edge (search edge 3) 4 [2593] 2593 innerTail375_15 best = best := by
  rw [innerTail375_15, scan, if_neg (by decide), innerCandidates375_15_checked]
  change scan edge (search edge 3) 4 [2593] 2593 innerTail375_16
    (if 3 ≤ innerCandidates375_15.length then search edge 3 [2593, 7027] innerCandidates375_15 9620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan375_16

def innerTail375_14 : List Nat := 5683 :: innerTail375_15
def innerCandidates375_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates375_14_checked : candidates edge 4 2593 5683 innerTail375_15 best.bound = innerCandidates375_14 := by
  unfold candidates
  have row : edge 5683 = SearchData.row5683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan375_14 : scan edge (search edge 3) 4 [2593] 2593 innerTail375_14 best = best := by
  rw [innerTail375_14, scan, if_neg (by decide), innerCandidates375_14_checked]
  change scan edge (search edge 3) 4 [2593] 2593 innerTail375_15
    (if 3 ≤ innerCandidates375_14.length then search edge 3 [2593, 5683] innerCandidates375_14 8276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan375_15

def innerTail375_13 : List Nat := 5443 :: innerTail375_14
def innerCandidates375_13 : List Nat := [7039]
set_option maxRecDepth 4096 in
theorem innerCandidates375_13_checked : candidates edge 4 2593 5443 innerTail375_14 best.bound = innerCandidates375_13 := by
  unfold candidates
  have row : edge 5443 = SearchData.row5443.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan375_13 : scan edge (search edge 3) 4 [2593] 2593 innerTail375_13 best = best := by
  rw [innerTail375_13, scan, if_neg (by decide), innerCandidates375_13_checked]
  change scan edge (search edge 3) 4 [2593] 2593 innerTail375_14
    (if 3 ≤ innerCandidates375_13.length then search edge 3 [2593, 5443] innerCandidates375_13 8036 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan375_14

def innerTail375_12 : List Nat := 5281 :: innerTail375_13
def innerCandidates375_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates375_12_checked : candidates edge 4 2593 5281 innerTail375_13 best.bound = innerCandidates375_12 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan375_12 : scan edge (search edge 3) 4 [2593] 2593 innerTail375_12 best = best := by
  rw [innerTail375_12, scan, if_neg (by decide), innerCandidates375_12_checked]
  change scan edge (search edge 3) 4 [2593] 2593 innerTail375_13
    (if 3 ≤ innerCandidates375_12.length then search edge 3 [2593, 5281] innerCandidates375_12 7874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan375_13

def innerTail375_11 : List Nat := 5101 :: innerTail375_12
def innerCandidates375_11 : List Nat := [8293]
set_option maxRecDepth 4096 in
theorem innerCandidates375_11_checked : candidates edge 4 2593 5101 innerTail375_12 best.bound = innerCandidates375_11 := by
  unfold candidates
  have row : edge 5101 = SearchData.row5101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan375_11 : scan edge (search edge 3) 4 [2593] 2593 innerTail375_11 best = best := by
  rw [innerTail375_11, scan, if_neg (by decide), innerCandidates375_11_checked]
  change scan edge (search edge 3) 4 [2593] 2593 innerTail375_12
    (if 3 ≤ innerCandidates375_11.length then search edge 3 [2593, 5101] innerCandidates375_11 7694 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan375_12

def innerTail375_10 : List Nat := 5011 :: innerTail375_11
def innerCandidates375_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates375_10_checked : candidates edge 4 2593 5011 innerTail375_11 best.bound = innerCandidates375_10 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan375_10 : scan edge (search edge 3) 4 [2593] 2593 innerTail375_10 best = best := by
  rw [innerTail375_10, scan, if_neg (by decide), innerCandidates375_10_checked]
  change scan edge (search edge 3) 4 [2593] 2593 innerTail375_11
    (if 3 ≤ innerCandidates375_10.length then search edge 3 [2593, 5011] innerCandidates375_10 7604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan375_11

def innerTail375_9 : List Nat := 3847 :: innerTail375_10
def innerCandidates375_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates375_9_checked : candidates edge 4 2593 3847 innerTail375_10 best.bound = innerCandidates375_9 := by
  unfold candidates
  have row : edge 3847 = SearchData.row3847.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan375_9 : scan edge (search edge 3) 4 [2593] 2593 innerTail375_9 best = best := by
  rw [innerTail375_9, scan, if_neg (by decide), innerCandidates375_9_checked]
  change scan edge (search edge 3) 4 [2593] 2593 innerTail375_10
    (if 3 ≤ innerCandidates375_9.length then search edge 3 [2593, 3847] innerCandidates375_9 6440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan375_10

def innerTail375_8 : List Nat := 3709 :: innerTail375_9
def innerCandidates375_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates375_8_checked : candidates edge 4 2593 3709 innerTail375_9 best.bound = innerCandidates375_8 := by
  unfold candidates
  have row : edge 3709 = SearchData.row3709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan375_8 : scan edge (search edge 3) 4 [2593] 2593 innerTail375_8 best = best := by
  rw [innerTail375_8, scan, if_neg (by decide), innerCandidates375_8_checked]
  change scan edge (search edge 3) 4 [2593] 2593 innerTail375_9
    (if 3 ≤ innerCandidates375_8.length then search edge 3 [2593, 3709] innerCandidates375_8 6302 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan375_9

def innerTail375_7 : List Nat := 3373 :: innerTail375_8
def innerCandidates375_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates375_7_checked : candidates edge 4 2593 3373 innerTail375_8 best.bound = innerCandidates375_7 := by
  unfold candidates
  have row : edge 3373 = SearchData.row3373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan375_7 : scan edge (search edge 3) 4 [2593] 2593 innerTail375_7 best = best := by
  rw [innerTail375_7, scan, if_neg (by decide), innerCandidates375_7_checked]
  change scan edge (search edge 3) 4 [2593] 2593 innerTail375_8
    (if 3 ≤ innerCandidates375_7.length then search edge 3 [2593, 3373] innerCandidates375_7 5966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan375_8

def innerTail375_6 : List Nat := 3307 :: innerTail375_7
def innerCandidates375_6 : List Nat := [5443, 7039]
set_option maxRecDepth 4096 in
theorem innerCandidates375_6_checked : candidates edge 4 2593 3307 innerTail375_7 best.bound = innerCandidates375_6 := by
  unfold candidates
  have row : edge 3307 = SearchData.row3307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan375_6 : scan edge (search edge 3) 4 [2593] 2593 innerTail375_6 best = best := by
  rw [innerTail375_6, scan, if_neg (by decide), innerCandidates375_6_checked]
  change scan edge (search edge 3) 4 [2593] 2593 innerTail375_7
    (if 3 ≤ innerCandidates375_6.length then search edge 3 [2593, 3307] innerCandidates375_6 5900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan375_7

def innerTail375_5 : List Nat := 3169 :: innerTail375_6
def innerCandidates375_5 : List Nat := [5101]
set_option maxRecDepth 4096 in
theorem innerCandidates375_5_checked : candidates edge 4 2593 3169 innerTail375_6 best.bound = innerCandidates375_5 := by
  unfold candidates
  have row : edge 3169 = SearchData.row3169.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan375_5 : scan edge (search edge 3) 4 [2593] 2593 innerTail375_5 best = best := by
  rw [innerTail375_5, scan, if_neg (by decide), innerCandidates375_5_checked]
  change scan edge (search edge 3) 4 [2593] 2593 innerTail375_6
    (if 3 ≤ innerCandidates375_5.length then search edge 3 [2593, 3169] innerCandidates375_5 5762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan375_6

def innerTail375_4 : List Nat := 2971 :: innerTail375_5
def innerCandidates375_4 : List Nat := [3709, 7573]
set_option maxRecDepth 4096 in
theorem innerCandidates375_4_checked : candidates edge 4 2593 2971 innerTail375_5 best.bound = innerCandidates375_4 := by
  unfold candidates
  have row : edge 2971 = SearchData.row2971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan375_4 : scan edge (search edge 3) 4 [2593] 2593 innerTail375_4 best = best := by
  rw [innerTail375_4, scan, if_neg (by decide), innerCandidates375_4_checked]
  change scan edge (search edge 3) 4 [2593] 2593 innerTail375_5
    (if 3 ≤ innerCandidates375_4.length then search edge 3 [2593, 2971] innerCandidates375_4 5564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan375_5

def innerTail375_3 : List Nat := 2887 :: innerTail375_4
def innerCandidates375_3 : List Nat := [7573]
set_option maxRecDepth 4096 in
theorem innerCandidates375_3_checked : candidates edge 4 2593 2887 innerTail375_4 best.bound = innerCandidates375_3 := by
  unfold candidates
  have row : edge 2887 = SearchData.row2887.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan375_3 : scan edge (search edge 3) 4 [2593] 2593 innerTail375_3 best = best := by
  rw [innerTail375_3, scan, if_neg (by decide), innerCandidates375_3_checked]
  change scan edge (search edge 3) 4 [2593] 2593 innerTail375_4
    (if 3 ≤ innerCandidates375_3.length then search edge 3 [2593, 2887] innerCandidates375_3 5480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan375_4

def innerTail375_2 : List Nat := 2719 :: innerTail375_3
def innerCandidates375_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates375_2_checked : candidates edge 4 2593 2719 innerTail375_3 best.bound = innerCandidates375_2 := by
  unfold candidates
  have row : edge 2719 = SearchData.row2719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan375_2 : scan edge (search edge 3) 4 [2593] 2593 innerTail375_2 best = best := by
  rw [innerTail375_2, scan, if_neg (by decide), innerCandidates375_2_checked]
  change scan edge (search edge 3) 4 [2593] 2593 innerTail375_3
    (if 3 ≤ innerCandidates375_2.length then search edge 3 [2593, 2719] innerCandidates375_2 5312 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan375_3

def innerTail375_1 : List Nat := 2713 :: innerTail375_2
def innerCandidates375_1 : List Nat := [5443]
set_option maxRecDepth 4096 in
theorem innerCandidates375_1_checked : candidates edge 4 2593 2713 innerTail375_2 best.bound = innerCandidates375_1 := by
  unfold candidates
  have row : edge 2713 = SearchData.row2713.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan375_1 : scan edge (search edge 3) 4 [2593] 2593 innerTail375_1 best = best := by
  rw [innerTail375_1, scan, if_neg (by decide), innerCandidates375_1_checked]
  change scan edge (search edge 3) 4 [2593] 2593 innerTail375_2
    (if 3 ≤ innerCandidates375_1.length then search edge 3 [2593, 2713] innerCandidates375_1 5306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan375_2

def innerTail375_0 : List Nat := 2617 :: innerTail375_1
def innerCandidates375_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates375_0_checked : candidates edge 4 2593 2617 innerTail375_1 best.bound = innerCandidates375_0 := by
  unfold candidates
  have row : edge 2617 = PrimePairCache.smallRow2617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan375_0 : scan edge (search edge 3) 4 [2593] 2593 innerTail375_0 best = best := by
  rw [innerTail375_0, scan, if_neg (by decide), innerCandidates375_0_checked]
  change scan edge (search edge 3) 4 [2593] 2593 innerTail375_1
    (if 3 ≤ innerCandidates375_0.length then search edge 3 [2593, 2617] innerCandidates375_0 5210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan375_1

theorem rootBranch375 : search edge 4 [2593] rootCandidates375 2593 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2593] 2593 innerTail375_0 best = best
  exact innerScan375_0

def rootCandidates374 : List Nat := [2633, 3257, 3527, 4073, 4211, 4283, 4937, 5741, 5807, 5849, 6197, 6299, 7193, 7499, 7937, 8231]
set_option maxRecDepth 4096 in
theorem rootCandidates374_checked : candidates edge 5 0 2591 tail375 best.bound = rootCandidates374 := by
  unfold candidates
  have row : edge 2591 = PrimePairCache.smallRow2591.contains := by funext q; rfl
  rw [row]
  decide

def innerTail374_16 : List Nat := []
theorem innerScan374_16 : scan edge (search edge 3) 4 [2591] 2591 innerTail374_16 best = best := rfl

def innerTail374_15 : List Nat := 8231 :: innerTail374_16
def innerCandidates374_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates374_15_checked : candidates edge 4 2591 8231 innerTail374_16 best.bound = innerCandidates374_15 := by
  unfold candidates
  have row : edge 8231 = SearchData.row8231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan374_15 : scan edge (search edge 3) 4 [2591] 2591 innerTail374_15 best = best := by
  rw [innerTail374_15, scan, if_neg (by decide), innerCandidates374_15_checked]
  change scan edge (search edge 3) 4 [2591] 2591 innerTail374_16
    (if 3 ≤ innerCandidates374_15.length then search edge 3 [2591, 8231] innerCandidates374_15 10822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan374_16

def innerTail374_14 : List Nat := 7937 :: innerTail374_15
def innerCandidates374_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates374_14_checked : candidates edge 4 2591 7937 innerTail374_15 best.bound = innerCandidates374_14 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan374_14 : scan edge (search edge 3) 4 [2591] 2591 innerTail374_14 best = best := by
  rw [innerTail374_14, scan, if_neg (by decide), innerCandidates374_14_checked]
  change scan edge (search edge 3) 4 [2591] 2591 innerTail374_15
    (if 3 ≤ innerCandidates374_14.length then search edge 3 [2591, 7937] innerCandidates374_14 10528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan374_15

def innerTail374_13 : List Nat := 7499 :: innerTail374_14
def innerCandidates374_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates374_13_checked : candidates edge 4 2591 7499 innerTail374_14 best.bound = innerCandidates374_13 := by
  unfold candidates
  have row : edge 7499 = SearchData.row7499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan374_13 : scan edge (search edge 3) 4 [2591] 2591 innerTail374_13 best = best := by
  rw [innerTail374_13, scan, if_neg (by decide), innerCandidates374_13_checked]
  change scan edge (search edge 3) 4 [2591] 2591 innerTail374_14
    (if 3 ≤ innerCandidates374_13.length then search edge 3 [2591, 7499] innerCandidates374_13 10090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan374_14

def innerTail374_12 : List Nat := 7193 :: innerTail374_13
def innerCandidates374_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates374_12_checked : candidates edge 4 2591 7193 innerTail374_13 best.bound = innerCandidates374_12 := by
  unfold candidates
  have row : edge 7193 = SearchData.row7193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan374_12 : scan edge (search edge 3) 4 [2591] 2591 innerTail374_12 best = best := by
  rw [innerTail374_12, scan, if_neg (by decide), innerCandidates374_12_checked]
  change scan edge (search edge 3) 4 [2591] 2591 innerTail374_13
    (if 3 ≤ innerCandidates374_12.length then search edge 3 [2591, 7193] innerCandidates374_12 9784 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan374_13

def innerTail374_11 : List Nat := 6299 :: innerTail374_12
def innerCandidates374_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates374_11_checked : candidates edge 4 2591 6299 innerTail374_12 best.bound = innerCandidates374_11 := by
  unfold candidates
  have row : edge 6299 = SearchData.row6299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan374_11 : scan edge (search edge 3) 4 [2591] 2591 innerTail374_11 best = best := by
  rw [innerTail374_11, scan, if_neg (by decide), innerCandidates374_11_checked]
  change scan edge (search edge 3) 4 [2591] 2591 innerTail374_12
    (if 3 ≤ innerCandidates374_11.length then search edge 3 [2591, 6299] innerCandidates374_11 8890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan374_12

def innerTail374_10 : List Nat := 6197 :: innerTail374_11
def innerCandidates374_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates374_10_checked : candidates edge 4 2591 6197 innerTail374_11 best.bound = innerCandidates374_10 := by
  unfold candidates
  have row : edge 6197 = SearchData.row6197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan374_10 : scan edge (search edge 3) 4 [2591] 2591 innerTail374_10 best = best := by
  rw [innerTail374_10, scan, if_neg (by decide), innerCandidates374_10_checked]
  change scan edge (search edge 3) 4 [2591] 2591 innerTail374_11
    (if 3 ≤ innerCandidates374_10.length then search edge 3 [2591, 6197] innerCandidates374_10 8788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan374_11

def innerTail374_9 : List Nat := 5849 :: innerTail374_10
def innerCandidates374_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates374_9_checked : candidates edge 4 2591 5849 innerTail374_10 best.bound = innerCandidates374_9 := by
  unfold candidates
  have row : edge 5849 = SearchData.row5849.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan374_9 : scan edge (search edge 3) 4 [2591] 2591 innerTail374_9 best = best := by
  rw [innerTail374_9, scan, if_neg (by decide), innerCandidates374_9_checked]
  change scan edge (search edge 3) 4 [2591] 2591 innerTail374_10
    (if 3 ≤ innerCandidates374_9.length then search edge 3 [2591, 5849] innerCandidates374_9 8440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan374_10

def innerTail374_8 : List Nat := 5807 :: innerTail374_9
def innerCandidates374_8 : List Nat := [7499]
set_option maxRecDepth 4096 in
theorem innerCandidates374_8_checked : candidates edge 4 2591 5807 innerTail374_9 best.bound = innerCandidates374_8 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan374_8 : scan edge (search edge 3) 4 [2591] 2591 innerTail374_8 best = best := by
  rw [innerTail374_8, scan, if_neg (by decide), innerCandidates374_8_checked]
  change scan edge (search edge 3) 4 [2591] 2591 innerTail374_9
    (if 3 ≤ innerCandidates374_8.length then search edge 3 [2591, 5807] innerCandidates374_8 8398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan374_9

def innerTail374_7 : List Nat := 5741 :: innerTail374_8
def innerCandidates374_7 : List Nat := [6299]
set_option maxRecDepth 4096 in
theorem innerCandidates374_7_checked : candidates edge 4 2591 5741 innerTail374_8 best.bound = innerCandidates374_7 := by
  unfold candidates
  have row : edge 5741 = SearchData.row5741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan374_7 : scan edge (search edge 3) 4 [2591] 2591 innerTail374_7 best = best := by
  rw [innerTail374_7, scan, if_neg (by decide), innerCandidates374_7_checked]
  change scan edge (search edge 3) 4 [2591] 2591 innerTail374_8
    (if 3 ≤ innerCandidates374_7.length then search edge 3 [2591, 5741] innerCandidates374_7 8332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan374_8

def innerTail374_6 : List Nat := 4937 :: innerTail374_7
def innerCandidates374_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates374_6_checked : candidates edge 4 2591 4937 innerTail374_7 best.bound = innerCandidates374_6 := by
  unfold candidates
  have row : edge 4937 = SearchData.row4937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan374_6 : scan edge (search edge 3) 4 [2591] 2591 innerTail374_6 best = best := by
  rw [innerTail374_6, scan, if_neg (by decide), innerCandidates374_6_checked]
  change scan edge (search edge 3) 4 [2591] 2591 innerTail374_7
    (if 3 ≤ innerCandidates374_6.length then search edge 3 [2591, 4937] innerCandidates374_6 7528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan374_7

def innerTail374_5 : List Nat := 4283 :: innerTail374_6
def innerCandidates374_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates374_5_checked : candidates edge 4 2591 4283 innerTail374_6 best.bound = innerCandidates374_5 := by
  unfold candidates
  have row : edge 4283 = SearchData.row4283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan374_5 : scan edge (search edge 3) 4 [2591] 2591 innerTail374_5 best = best := by
  rw [innerTail374_5, scan, if_neg (by decide), innerCandidates374_5_checked]
  change scan edge (search edge 3) 4 [2591] 2591 innerTail374_6
    (if 3 ≤ innerCandidates374_5.length then search edge 3 [2591, 4283] innerCandidates374_5 6874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan374_6

def innerTail374_4 : List Nat := 4211 :: innerTail374_5
def innerCandidates374_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates374_4_checked : candidates edge 4 2591 4211 innerTail374_5 best.bound = innerCandidates374_4 := by
  unfold candidates
  have row : edge 4211 = SearchData.row4211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan374_4 : scan edge (search edge 3) 4 [2591] 2591 innerTail374_4 best = best := by
  rw [innerTail374_4, scan, if_neg (by decide), innerCandidates374_4_checked]
  change scan edge (search edge 3) 4 [2591] 2591 innerTail374_5
    (if 3 ≤ innerCandidates374_4.length then search edge 3 [2591, 4211] innerCandidates374_4 6802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan374_5

def innerTail374_3 : List Nat := 4073 :: innerTail374_4
def innerCandidates374_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates374_3_checked : candidates edge 4 2591 4073 innerTail374_4 best.bound = innerCandidates374_3 := by
  unfold candidates
  have row : edge 4073 = SearchData.row4073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan374_3 : scan edge (search edge 3) 4 [2591] 2591 innerTail374_3 best = best := by
  rw [innerTail374_3, scan, if_neg (by decide), innerCandidates374_3_checked]
  change scan edge (search edge 3) 4 [2591] 2591 innerTail374_4
    (if 3 ≤ innerCandidates374_3.length then search edge 3 [2591, 4073] innerCandidates374_3 6664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan374_4

def innerTail374_2 : List Nat := 3527 :: innerTail374_3
def innerCandidates374_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates374_2_checked : candidates edge 4 2591 3527 innerTail374_3 best.bound = innerCandidates374_2 := by
  unfold candidates
  have row : edge 3527 = SearchData.row3527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan374_2 : scan edge (search edge 3) 4 [2591] 2591 innerTail374_2 best = best := by
  rw [innerTail374_2, scan, if_neg (by decide), innerCandidates374_2_checked]
  change scan edge (search edge 3) 4 [2591] 2591 innerTail374_3
    (if 3 ≤ innerCandidates374_2.length then search edge 3 [2591, 3527] innerCandidates374_2 6118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan374_3

def innerTail374_1 : List Nat := 3257 :: innerTail374_2
def innerCandidates374_1 : List Nat := [4937]
set_option maxRecDepth 4096 in
theorem innerCandidates374_1_checked : candidates edge 4 2591 3257 innerTail374_2 best.bound = innerCandidates374_1 := by
  unfold candidates
  have row : edge 3257 = SearchData.row3257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan374_1 : scan edge (search edge 3) 4 [2591] 2591 innerTail374_1 best = best := by
  rw [innerTail374_1, scan, if_neg (by decide), innerCandidates374_1_checked]
  change scan edge (search edge 3) 4 [2591] 2591 innerTail374_2
    (if 3 ≤ innerCandidates374_1.length then search edge 3 [2591, 3257] innerCandidates374_1 5848 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan374_2

def innerTail374_0 : List Nat := 2633 :: innerTail374_1
def innerCandidates374_0 : List Nat := [5807, 7193]
set_option maxRecDepth 4096 in
theorem innerCandidates374_0_checked : candidates edge 4 2591 2633 innerTail374_1 best.bound = innerCandidates374_0 := by
  unfold candidates
  have row : edge 2633 = PrimePairCache.smallRow2633.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan374_0 : scan edge (search edge 3) 4 [2591] 2591 innerTail374_0 best = best := by
  rw [innerTail374_0, scan, if_neg (by decide), innerCandidates374_0_checked]
  change scan edge (search edge 3) 4 [2591] 2591 innerTail374_1
    (if 3 ≤ innerCandidates374_0.length then search edge 3 [2591, 2633] innerCandidates374_0 5224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan374_1

theorem rootBranch374 : search edge 4 [2591] rootCandidates374 2591 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2591] 2591 innerTail374_0 best = best
  exact innerScan374_0

def rootCandidates373 : List Nat := [3359, 3623, 4007, 4421, 4871, 4931, 5639, 5843, 6131, 7229, 7349, 7727]
set_option maxRecDepth 4096 in
theorem rootCandidates373_checked : candidates edge 5 0 2579 tail374 best.bound = rootCandidates373 := by
  unfold candidates
  have row : edge 2579 = PrimePairCache.smallRow2579.contains := by funext q; rfl
  rw [row]
  decide

def innerTail373_12 : List Nat := []
theorem innerScan373_12 : scan edge (search edge 3) 4 [2579] 2579 innerTail373_12 best = best := rfl

def innerTail373_11 : List Nat := 7727 :: innerTail373_12
def innerCandidates373_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates373_11_checked : candidates edge 4 2579 7727 innerTail373_12 best.bound = innerCandidates373_11 := by
  unfold candidates
  have row : edge 7727 = SearchData.row7727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan373_11 : scan edge (search edge 3) 4 [2579] 2579 innerTail373_11 best = best := by
  rw [innerTail373_11, scan, if_neg (by decide), innerCandidates373_11_checked]
  change scan edge (search edge 3) 4 [2579] 2579 innerTail373_12
    (if 3 ≤ innerCandidates373_11.length then search edge 3 [2579, 7727] innerCandidates373_11 10306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan373_12

def innerTail373_10 : List Nat := 7349 :: innerTail373_11
def innerCandidates373_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates373_10_checked : candidates edge 4 2579 7349 innerTail373_11 best.bound = innerCandidates373_10 := by
  unfold candidates
  have row : edge 7349 = SearchData.row7349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan373_10 : scan edge (search edge 3) 4 [2579] 2579 innerTail373_10 best = best := by
  rw [innerTail373_10, scan, if_neg (by decide), innerCandidates373_10_checked]
  change scan edge (search edge 3) 4 [2579] 2579 innerTail373_11
    (if 3 ≤ innerCandidates373_10.length then search edge 3 [2579, 7349] innerCandidates373_10 9928 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan373_11

def innerTail373_9 : List Nat := 7229 :: innerTail373_10
def innerCandidates373_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates373_9_checked : candidates edge 4 2579 7229 innerTail373_10 best.bound = innerCandidates373_9 := by
  unfold candidates
  have row : edge 7229 = SearchData.row7229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan373_9 : scan edge (search edge 3) 4 [2579] 2579 innerTail373_9 best = best := by
  rw [innerTail373_9, scan, if_neg (by decide), innerCandidates373_9_checked]
  change scan edge (search edge 3) 4 [2579] 2579 innerTail373_10
    (if 3 ≤ innerCandidates373_9.length then search edge 3 [2579, 7229] innerCandidates373_9 9808 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan373_10

def innerTail373_8 : List Nat := 6131 :: innerTail373_9
def innerCandidates373_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates373_8_checked : candidates edge 4 2579 6131 innerTail373_9 best.bound = innerCandidates373_8 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan373_8 : scan edge (search edge 3) 4 [2579] 2579 innerTail373_8 best = best := by
  rw [innerTail373_8, scan, if_neg (by decide), innerCandidates373_8_checked]
  change scan edge (search edge 3) 4 [2579] 2579 innerTail373_9
    (if 3 ≤ innerCandidates373_8.length then search edge 3 [2579, 6131] innerCandidates373_8 8710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan373_9

def innerTail373_7 : List Nat := 5843 :: innerTail373_8
def innerCandidates373_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates373_7_checked : candidates edge 4 2579 5843 innerTail373_8 best.bound = innerCandidates373_7 := by
  unfold candidates
  have row : edge 5843 = SearchData.row5843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan373_7 : scan edge (search edge 3) 4 [2579] 2579 innerTail373_7 best = best := by
  rw [innerTail373_7, scan, if_neg (by decide), innerCandidates373_7_checked]
  change scan edge (search edge 3) 4 [2579] 2579 innerTail373_8
    (if 3 ≤ innerCandidates373_7.length then search edge 3 [2579, 5843] innerCandidates373_7 8422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan373_8

def innerTail373_6 : List Nat := 5639 :: innerTail373_7
def innerCandidates373_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates373_6_checked : candidates edge 4 2579 5639 innerTail373_7 best.bound = innerCandidates373_6 := by
  unfold candidates
  have row : edge 5639 = SearchData.row5639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan373_6 : scan edge (search edge 3) 4 [2579] 2579 innerTail373_6 best = best := by
  rw [innerTail373_6, scan, if_neg (by decide), innerCandidates373_6_checked]
  change scan edge (search edge 3) 4 [2579] 2579 innerTail373_7
    (if 3 ≤ innerCandidates373_6.length then search edge 3 [2579, 5639] innerCandidates373_6 8218 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan373_7

def innerTail373_5 : List Nat := 4931 :: innerTail373_6
def innerCandidates373_5 : List Nat := [5843, 7349]
set_option maxRecDepth 4096 in
theorem innerCandidates373_5_checked : candidates edge 4 2579 4931 innerTail373_6 best.bound = innerCandidates373_5 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan373_5 : scan edge (search edge 3) 4 [2579] 2579 innerTail373_5 best = best := by
  rw [innerTail373_5, scan, if_neg (by decide), innerCandidates373_5_checked]
  change scan edge (search edge 3) 4 [2579] 2579 innerTail373_6
    (if 3 ≤ innerCandidates373_5.length then search edge 3 [2579, 4931] innerCandidates373_5 7510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan373_6

def innerTail373_4 : List Nat := 4871 :: innerTail373_5
def innerCandidates373_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates373_4_checked : candidates edge 4 2579 4871 innerTail373_5 best.bound = innerCandidates373_4 := by
  unfold candidates
  have row : edge 4871 = SearchData.row4871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan373_4 : scan edge (search edge 3) 4 [2579] 2579 innerTail373_4 best = best := by
  rw [innerTail373_4, scan, if_neg (by decide), innerCandidates373_4_checked]
  change scan edge (search edge 3) 4 [2579] 2579 innerTail373_5
    (if 3 ≤ innerCandidates373_4.length then search edge 3 [2579, 4871] innerCandidates373_4 7450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan373_5

def innerTail373_3 : List Nat := 4421 :: innerTail373_4
def innerCandidates373_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates373_3_checked : candidates edge 4 2579 4421 innerTail373_4 best.bound = innerCandidates373_3 := by
  unfold candidates
  have row : edge 4421 = SearchData.row4421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan373_3 : scan edge (search edge 3) 4 [2579] 2579 innerTail373_3 best = best := by
  rw [innerTail373_3, scan, if_neg (by decide), innerCandidates373_3_checked]
  change scan edge (search edge 3) 4 [2579] 2579 innerTail373_4
    (if 3 ≤ innerCandidates373_3.length then search edge 3 [2579, 4421] innerCandidates373_3 7000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan373_4

def innerTail373_2 : List Nat := 4007 :: innerTail373_3
def innerCandidates373_2 : List Nat := [5639]
set_option maxRecDepth 4096 in
theorem innerCandidates373_2_checked : candidates edge 4 2579 4007 innerTail373_3 best.bound = innerCandidates373_2 := by
  unfold candidates
  have row : edge 4007 = SearchData.row4007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan373_2 : scan edge (search edge 3) 4 [2579] 2579 innerTail373_2 best = best := by
  rw [innerTail373_2, scan, if_neg (by decide), innerCandidates373_2_checked]
  change scan edge (search edge 3) 4 [2579] 2579 innerTail373_3
    (if 3 ≤ innerCandidates373_2.length then search edge 3 [2579, 4007] innerCandidates373_2 6586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan373_3

def innerTail373_1 : List Nat := 3623 :: innerTail373_2
def innerCandidates373_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates373_1_checked : candidates edge 4 2579 3623 innerTail373_2 best.bound = innerCandidates373_1 := by
  unfold candidates
  have row : edge 3623 = SearchData.row3623.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan373_1 : scan edge (search edge 3) 4 [2579] 2579 innerTail373_1 best = best := by
  rw [innerTail373_1, scan, if_neg (by decide), innerCandidates373_1_checked]
  change scan edge (search edge 3) 4 [2579] 2579 innerTail373_2
    (if 3 ≤ innerCandidates373_1.length then search edge 3 [2579, 3623] innerCandidates373_1 6202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan373_2

def innerTail373_0 : List Nat := 3359 :: innerTail373_1
def innerCandidates373_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates373_0_checked : candidates edge 4 2579 3359 innerTail373_1 best.bound = innerCandidates373_0 := by
  unfold candidates
  have row : edge 3359 = SearchData.row3359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan373_0 : scan edge (search edge 3) 4 [2579] 2579 innerTail373_0 best = best := by
  rw [innerTail373_0, scan, if_neg (by decide), innerCandidates373_0_checked]
  change scan edge (search edge 3) 4 [2579] 2579 innerTail373_1
    (if 3 ≤ innerCandidates373_0.length then search edge 3 [2579, 3359] innerCandidates373_0 5938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan373_1

theorem rootBranch373 : search edge 4 [2579] rootCandidates373 2579 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2579] 2579 innerTail373_0 best = best
  exact innerScan373_0

def rootCandidates372 : List Nat := [2683, 3313, 4243, 4639, 4657, 4957, 5749, 6763, 6871, 6997, 7369, 7687, 7789, 7873, 8191]
set_option maxRecDepth 4096 in
theorem rootCandidates372_checked : candidates edge 5 0 2557 tail373 best.bound = rootCandidates372 := by
  unfold candidates
  have row : edge 2557 = PrimePairCache.smallRow2557.contains := by funext q; rfl
  rw [row]
  decide

def innerTail372_15 : List Nat := []
theorem innerScan372_15 : scan edge (search edge 3) 4 [2557] 2557 innerTail372_15 best = best := rfl

def innerTail372_14 : List Nat := 8191 :: innerTail372_15
def innerCandidates372_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates372_14_checked : candidates edge 4 2557 8191 innerTail372_15 best.bound = innerCandidates372_14 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan372_14 : scan edge (search edge 3) 4 [2557] 2557 innerTail372_14 best = best := by
  rw [innerTail372_14, scan, if_neg (by decide), innerCandidates372_14_checked]
  change scan edge (search edge 3) 4 [2557] 2557 innerTail372_15
    (if 3 ≤ innerCandidates372_14.length then search edge 3 [2557, 8191] innerCandidates372_14 10748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan372_15

def innerTail372_13 : List Nat := 7873 :: innerTail372_14
def innerCandidates372_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates372_13_checked : candidates edge 4 2557 7873 innerTail372_14 best.bound = innerCandidates372_13 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan372_13 : scan edge (search edge 3) 4 [2557] 2557 innerTail372_13 best = best := by
  rw [innerTail372_13, scan, if_neg (by decide), innerCandidates372_13_checked]
  change scan edge (search edge 3) 4 [2557] 2557 innerTail372_14
    (if 3 ≤ innerCandidates372_13.length then search edge 3 [2557, 7873] innerCandidates372_13 10430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan372_14

def innerTail372_12 : List Nat := 7789 :: innerTail372_13
def innerCandidates372_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates372_12_checked : candidates edge 4 2557 7789 innerTail372_13 best.bound = innerCandidates372_12 := by
  unfold candidates
  have row : edge 7789 = SearchData.row7789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan372_12 : scan edge (search edge 3) 4 [2557] 2557 innerTail372_12 best = best := by
  rw [innerTail372_12, scan, if_neg (by decide), innerCandidates372_12_checked]
  change scan edge (search edge 3) 4 [2557] 2557 innerTail372_13
    (if 3 ≤ innerCandidates372_12.length then search edge 3 [2557, 7789] innerCandidates372_12 10346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan372_13

def innerTail372_11 : List Nat := 7687 :: innerTail372_12
def innerCandidates372_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates372_11_checked : candidates edge 4 2557 7687 innerTail372_12 best.bound = innerCandidates372_11 := by
  unfold candidates
  have row : edge 7687 = SearchData.row7687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan372_11 : scan edge (search edge 3) 4 [2557] 2557 innerTail372_11 best = best := by
  rw [innerTail372_11, scan, if_neg (by decide), innerCandidates372_11_checked]
  change scan edge (search edge 3) 4 [2557] 2557 innerTail372_12
    (if 3 ≤ innerCandidates372_11.length then search edge 3 [2557, 7687] innerCandidates372_11 10244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan372_12

def innerTail372_10 : List Nat := 7369 :: innerTail372_11
def innerCandidates372_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates372_10_checked : candidates edge 4 2557 7369 innerTail372_11 best.bound = innerCandidates372_10 := by
  unfold candidates
  have row : edge 7369 = SearchData.row7369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan372_10 : scan edge (search edge 3) 4 [2557] 2557 innerTail372_10 best = best := by
  rw [innerTail372_10, scan, if_neg (by decide), innerCandidates372_10_checked]
  change scan edge (search edge 3) 4 [2557] 2557 innerTail372_11
    (if 3 ≤ innerCandidates372_10.length then search edge 3 [2557, 7369] innerCandidates372_10 9926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan372_11

def innerTail372_9 : List Nat := 6997 :: innerTail372_10
def innerCandidates372_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates372_9_checked : candidates edge 4 2557 6997 innerTail372_10 best.bound = innerCandidates372_9 := by
  unfold candidates
  have row : edge 6997 = SearchData.row6997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan372_9 : scan edge (search edge 3) 4 [2557] 2557 innerTail372_9 best = best := by
  rw [innerTail372_9, scan, if_neg (by decide), innerCandidates372_9_checked]
  change scan edge (search edge 3) 4 [2557] 2557 innerTail372_10
    (if 3 ≤ innerCandidates372_9.length then search edge 3 [2557, 6997] innerCandidates372_9 9554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan372_10

def innerTail372_8 : List Nat := 6871 :: innerTail372_9
def innerCandidates372_8 : List Nat := [8191]
set_option maxRecDepth 4096 in
theorem innerCandidates372_8_checked : candidates edge 4 2557 6871 innerTail372_9 best.bound = innerCandidates372_8 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan372_8 : scan edge (search edge 3) 4 [2557] 2557 innerTail372_8 best = best := by
  rw [innerTail372_8, scan, if_neg (by decide), innerCandidates372_8_checked]
  change scan edge (search edge 3) 4 [2557] 2557 innerTail372_9
    (if 3 ≤ innerCandidates372_8.length then search edge 3 [2557, 6871] innerCandidates372_8 9428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan372_9

def innerTail372_7 : List Nat := 6763 :: innerTail372_8
def innerCandidates372_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates372_7_checked : candidates edge 4 2557 6763 innerTail372_8 best.bound = innerCandidates372_7 := by
  unfold candidates
  have row : edge 6763 = SearchData.row6763.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan372_7 : scan edge (search edge 3) 4 [2557] 2557 innerTail372_7 best = best := by
  rw [innerTail372_7, scan, if_neg (by decide), innerCandidates372_7_checked]
  change scan edge (search edge 3) 4 [2557] 2557 innerTail372_8
    (if 3 ≤ innerCandidates372_7.length then search edge 3 [2557, 6763] innerCandidates372_7 9320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan372_8

def innerTail372_6 : List Nat := 5749 :: innerTail372_7
def innerCandidates372_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates372_6_checked : candidates edge 4 2557 5749 innerTail372_7 best.bound = innerCandidates372_6 := by
  unfold candidates
  have row : edge 5749 = SearchData.row5749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan372_6 : scan edge (search edge 3) 4 [2557] 2557 innerTail372_6 best = best := by
  rw [innerTail372_6, scan, if_neg (by decide), innerCandidates372_6_checked]
  change scan edge (search edge 3) 4 [2557] 2557 innerTail372_7
    (if 3 ≤ innerCandidates372_6.length then search edge 3 [2557, 5749] innerCandidates372_6 8306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan372_7

def innerTail372_5 : List Nat := 4957 :: innerTail372_6
def innerCandidates372_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates372_5_checked : candidates edge 4 2557 4957 innerTail372_6 best.bound = innerCandidates372_5 := by
  unfold candidates
  have row : edge 4957 = SearchData.row4957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan372_5 : scan edge (search edge 3) 4 [2557] 2557 innerTail372_5 best = best := by
  rw [innerTail372_5, scan, if_neg (by decide), innerCandidates372_5_checked]
  change scan edge (search edge 3) 4 [2557] 2557 innerTail372_6
    (if 3 ≤ innerCandidates372_5.length then search edge 3 [2557, 4957] innerCandidates372_5 7514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan372_6

def innerTail372_4 : List Nat := 4657 :: innerTail372_5
def innerCandidates372_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates372_4_checked : candidates edge 4 2557 4657 innerTail372_5 best.bound = innerCandidates372_4 := by
  unfold candidates
  have row : edge 4657 = SearchData.row4657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan372_4 : scan edge (search edge 3) 4 [2557] 2557 innerTail372_4 best = best := by
  rw [innerTail372_4, scan, if_neg (by decide), innerCandidates372_4_checked]
  change scan edge (search edge 3) 4 [2557] 2557 innerTail372_5
    (if 3 ≤ innerCandidates372_4.length then search edge 3 [2557, 4657] innerCandidates372_4 7214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan372_5

def innerTail372_3 : List Nat := 4639 :: innerTail372_4
def innerCandidates372_3 : List Nat := [5749]
set_option maxRecDepth 4096 in
theorem innerCandidates372_3_checked : candidates edge 4 2557 4639 innerTail372_4 best.bound = innerCandidates372_3 := by
  unfold candidates
  have row : edge 4639 = SearchData.row4639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan372_3 : scan edge (search edge 3) 4 [2557] 2557 innerTail372_3 best = best := by
  rw [innerTail372_3, scan, if_neg (by decide), innerCandidates372_3_checked]
  change scan edge (search edge 3) 4 [2557] 2557 innerTail372_4
    (if 3 ≤ innerCandidates372_3.length then search edge 3 [2557, 4639] innerCandidates372_3 7196 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan372_4

def innerTail372_2 : List Nat := 4243 :: innerTail372_3
def innerCandidates372_2 : List Nat := [6997]
set_option maxRecDepth 4096 in
theorem innerCandidates372_2_checked : candidates edge 4 2557 4243 innerTail372_3 best.bound = innerCandidates372_2 := by
  unfold candidates
  have row : edge 4243 = SearchData.row4243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan372_2 : scan edge (search edge 3) 4 [2557] 2557 innerTail372_2 best = best := by
  rw [innerTail372_2, scan, if_neg (by decide), innerCandidates372_2_checked]
  change scan edge (search edge 3) 4 [2557] 2557 innerTail372_3
    (if 3 ≤ innerCandidates372_2.length then search edge 3 [2557, 4243] innerCandidates372_2 6800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan372_3

def innerTail372_1 : List Nat := 3313 :: innerTail372_2
def innerCandidates372_1 : List Nat := [6871]
set_option maxRecDepth 4096 in
theorem innerCandidates372_1_checked : candidates edge 4 2557 3313 innerTail372_2 best.bound = innerCandidates372_1 := by
  unfold candidates
  have row : edge 3313 = SearchData.row3313.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan372_1 : scan edge (search edge 3) 4 [2557] 2557 innerTail372_1 best = best := by
  rw [innerTail372_1, scan, if_neg (by decide), innerCandidates372_1_checked]
  change scan edge (search edge 3) 4 [2557] 2557 innerTail372_2
    (if 3 ≤ innerCandidates372_1.length then search edge 3 [2557, 3313] innerCandidates372_1 5870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan372_2

def innerTail372_0 : List Nat := 2683 :: innerTail372_1
def innerCandidates372_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates372_0_checked : candidates edge 4 2557 2683 innerTail372_1 best.bound = innerCandidates372_0 := by
  unfold candidates
  have row : edge 2683 = SearchData.row2683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan372_0 : scan edge (search edge 3) 4 [2557] 2557 innerTail372_0 best = best := by
  rw [innerTail372_0, scan, if_neg (by decide), innerCandidates372_0_checked]
  change scan edge (search edge 3) 4 [2557] 2557 innerTail372_1
    (if 3 ≤ innerCandidates372_0.length then search edge 3 [2557, 2683] innerCandidates372_0 5240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan372_1

theorem rootBranch372 : search edge 4 [2557] rootCandidates372 2557 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2557] 2557 innerTail372_0 best = best
  exact innerScan372_0

end ProjectEulerSolutions.PrimePairUpper.Search
