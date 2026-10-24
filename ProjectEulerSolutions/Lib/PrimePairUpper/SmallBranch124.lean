import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates306 : List Nat := [2357, 2417, 2591, 3299, 3323, 3491, 3767, 3917, 4079, 4523, 4799, 4937, 6053, 6203, 6689, 7673]
set_option maxRecDepth 4096 in
theorem rootCandidates306_checked : candidates edge 5 0 2039 tail307 best.bound = rootCandidates306 := by
  unfold candidates
  have row : edge 2039 = PrimePairCache.smallRow2039.contains := by funext q; rfl
  rw [row]
  decide

def innerTail306_16 : List Nat := []
theorem innerScan306_16 : scan edge (search edge 3) 4 [2039] 2039 innerTail306_16 best = best := rfl

def innerTail306_15 : List Nat := 7673 :: innerTail306_16
def innerCandidates306_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates306_15_checked : candidates edge 4 2039 7673 innerTail306_16 best.bound = innerCandidates306_15 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan306_15 : scan edge (search edge 3) 4 [2039] 2039 innerTail306_15 best = best := by
  rw [innerTail306_15, scan, if_neg (by decide), innerCandidates306_15_checked]
  change scan edge (search edge 3) 4 [2039] 2039 innerTail306_16
    (if 3 ≤ innerCandidates306_15.length then search edge 3 [2039, 7673] innerCandidates306_15 9712 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan306_16

def innerTail306_14 : List Nat := 6689 :: innerTail306_15
def innerCandidates306_14 : List Nat := [7673]
set_option maxRecDepth 4096 in
theorem innerCandidates306_14_checked : candidates edge 4 2039 6689 innerTail306_15 best.bound = innerCandidates306_14 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan306_14 : scan edge (search edge 3) 4 [2039] 2039 innerTail306_14 best = best := by
  rw [innerTail306_14, scan, if_neg (by decide), innerCandidates306_14_checked]
  change scan edge (search edge 3) 4 [2039] 2039 innerTail306_15
    (if 3 ≤ innerCandidates306_14.length then search edge 3 [2039, 6689] innerCandidates306_14 8728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan306_15

def innerTail306_13 : List Nat := 6203 :: innerTail306_14
def innerCandidates306_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates306_13_checked : candidates edge 4 2039 6203 innerTail306_14 best.bound = innerCandidates306_13 := by
  unfold candidates
  have row : edge 6203 = SearchData.row6203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan306_13 : scan edge (search edge 3) 4 [2039] 2039 innerTail306_13 best = best := by
  rw [innerTail306_13, scan, if_neg (by decide), innerCandidates306_13_checked]
  change scan edge (search edge 3) 4 [2039] 2039 innerTail306_14
    (if 3 ≤ innerCandidates306_13.length then search edge 3 [2039, 6203] innerCandidates306_13 8242 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan306_14

def innerTail306_12 : List Nat := 6053 :: innerTail306_13
def innerCandidates306_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates306_12_checked : candidates edge 4 2039 6053 innerTail306_13 best.bound = innerCandidates306_12 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan306_12 : scan edge (search edge 3) 4 [2039] 2039 innerTail306_12 best = best := by
  rw [innerTail306_12, scan, if_neg (by decide), innerCandidates306_12_checked]
  change scan edge (search edge 3) 4 [2039] 2039 innerTail306_13
    (if 3 ≤ innerCandidates306_12.length then search edge 3 [2039, 6053] innerCandidates306_12 8092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan306_13

def innerTail306_11 : List Nat := 4937 :: innerTail306_12
def innerCandidates306_11 : List Nat := [6053]
set_option maxRecDepth 4096 in
theorem innerCandidates306_11_checked : candidates edge 4 2039 4937 innerTail306_12 best.bound = innerCandidates306_11 := by
  unfold candidates
  have row : edge 4937 = SearchData.row4937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan306_11 : scan edge (search edge 3) 4 [2039] 2039 innerTail306_11 best = best := by
  rw [innerTail306_11, scan, if_neg (by decide), innerCandidates306_11_checked]
  change scan edge (search edge 3) 4 [2039] 2039 innerTail306_12
    (if 3 ≤ innerCandidates306_11.length then search edge 3 [2039, 4937] innerCandidates306_11 6976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan306_12

def innerTail306_10 : List Nat := 4799 :: innerTail306_11
def innerCandidates306_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates306_10_checked : candidates edge 4 2039 4799 innerTail306_11 best.bound = innerCandidates306_10 := by
  unfold candidates
  have row : edge 4799 = SearchData.row4799.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan306_10 : scan edge (search edge 3) 4 [2039] 2039 innerTail306_10 best = best := by
  rw [innerTail306_10, scan, if_neg (by decide), innerCandidates306_10_checked]
  change scan edge (search edge 3) 4 [2039] 2039 innerTail306_11
    (if 3 ≤ innerCandidates306_10.length then search edge 3 [2039, 4799] innerCandidates306_10 6838 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan306_11

def innerTail306_9 : List Nat := 4523 :: innerTail306_10
def innerCandidates306_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates306_9_checked : candidates edge 4 2039 4523 innerTail306_10 best.bound = innerCandidates306_9 := by
  unfold candidates
  have row : edge 4523 = SearchData.row4523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan306_9 : scan edge (search edge 3) 4 [2039] 2039 innerTail306_9 best = best := by
  rw [innerTail306_9, scan, if_neg (by decide), innerCandidates306_9_checked]
  change scan edge (search edge 3) 4 [2039] 2039 innerTail306_10
    (if 3 ≤ innerCandidates306_9.length then search edge 3 [2039, 4523] innerCandidates306_9 6562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan306_10

def innerTail306_8 : List Nat := 4079 :: innerTail306_9
def innerCandidates306_8 : List Nat := [6053]
set_option maxRecDepth 4096 in
theorem innerCandidates306_8_checked : candidates edge 4 2039 4079 innerTail306_9 best.bound = innerCandidates306_8 := by
  unfold candidates
  have row : edge 4079 = SearchData.row4079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan306_8 : scan edge (search edge 3) 4 [2039] 2039 innerTail306_8 best = best := by
  rw [innerTail306_8, scan, if_neg (by decide), innerCandidates306_8_checked]
  change scan edge (search edge 3) 4 [2039] 2039 innerTail306_9
    (if 3 ≤ innerCandidates306_8.length then search edge 3 [2039, 4079] innerCandidates306_8 6118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan306_9

def innerTail306_7 : List Nat := 3917 :: innerTail306_8
def innerCandidates306_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates306_7_checked : candidates edge 4 2039 3917 innerTail306_8 best.bound = innerCandidates306_7 := by
  unfold candidates
  have row : edge 3917 = SearchData.row3917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan306_7 : scan edge (search edge 3) 4 [2039] 2039 innerTail306_7 best = best := by
  rw [innerTail306_7, scan, if_neg (by decide), innerCandidates306_7_checked]
  change scan edge (search edge 3) 4 [2039] 2039 innerTail306_8
    (if 3 ≤ innerCandidates306_7.length then search edge 3 [2039, 3917] innerCandidates306_7 5956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan306_8

def innerTail306_6 : List Nat := 3767 :: innerTail306_7
def innerCandidates306_6 : List Nat := [4523, 4937]
set_option maxRecDepth 4096 in
theorem innerCandidates306_6_checked : candidates edge 4 2039 3767 innerTail306_7 best.bound = innerCandidates306_6 := by
  unfold candidates
  have row : edge 3767 = SearchData.row3767.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan306_6 : scan edge (search edge 3) 4 [2039] 2039 innerTail306_6 best = best := by
  rw [innerTail306_6, scan, if_neg (by decide), innerCandidates306_6_checked]
  change scan edge (search edge 3) 4 [2039] 2039 innerTail306_7
    (if 3 ≤ innerCandidates306_6.length then search edge 3 [2039, 3767] innerCandidates306_6 5806 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan306_7

def innerTail306_5 : List Nat := 3491 :: innerTail306_6
def innerCandidates306_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates306_5_checked : candidates edge 4 2039 3491 innerTail306_6 best.bound = innerCandidates306_5 := by
  unfold candidates
  have row : edge 3491 = SearchData.row3491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan306_5 : scan edge (search edge 3) 4 [2039] 2039 innerTail306_5 best = best := by
  rw [innerTail306_5, scan, if_neg (by decide), innerCandidates306_5_checked]
  change scan edge (search edge 3) 4 [2039] 2039 innerTail306_6
    (if 3 ≤ innerCandidates306_5.length then search edge 3 [2039, 3491] innerCandidates306_5 5530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan306_6

def innerTail306_4 : List Nat := 3323 :: innerTail306_5
def innerCandidates306_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates306_4_checked : candidates edge 4 2039 3323 innerTail306_5 best.bound = innerCandidates306_4 := by
  unfold candidates
  have row : edge 3323 = SearchData.row3323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan306_4 : scan edge (search edge 3) 4 [2039] 2039 innerTail306_4 best = best := by
  rw [innerTail306_4, scan, if_neg (by decide), innerCandidates306_4_checked]
  change scan edge (search edge 3) 4 [2039] 2039 innerTail306_5
    (if 3 ≤ innerCandidates306_4.length then search edge 3 [2039, 3323] innerCandidates306_4 5362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan306_5

def innerTail306_3 : List Nat := 3299 :: innerTail306_4
def innerCandidates306_3 : List Nat := [3767, 4523, 6203]
set_option maxRecDepth 4096 in
theorem innerCandidates306_3_checked : candidates edge 4 2039 3299 innerTail306_4 best.bound = innerCandidates306_3 := by
  unfold candidates
  have row : edge 3299 = SearchData.row3299.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild306_3 : search edge 3 [2039, 3299] innerCandidates306_3 5338 best = best := by decide

theorem innerScan306_3 : scan edge (search edge 3) 4 [2039] 2039 innerTail306_3 best = best := by
  rw [innerTail306_3, scan, if_neg (by decide), innerCandidates306_3_checked]
  change scan edge (search edge 3) 4 [2039] 2039 innerTail306_4
    (if 3 ≤ innerCandidates306_3.length then search edge 3 [2039, 3299] innerCandidates306_3 5338 best else best) = best
  rw [if_pos (by decide), innerChild306_3]
  exact innerScan306_4

def innerTail306_2 : List Nat := 2591 :: innerTail306_3
def innerCandidates306_2 : List Nat := [4937]
set_option maxRecDepth 4096 in
theorem innerCandidates306_2_checked : candidates edge 4 2039 2591 innerTail306_3 best.bound = innerCandidates306_2 := by
  unfold candidates
  have row : edge 2591 = PrimePairCache.smallRow2591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan306_2 : scan edge (search edge 3) 4 [2039] 2039 innerTail306_2 best = best := by
  rw [innerTail306_2, scan, if_neg (by decide), innerCandidates306_2_checked]
  change scan edge (search edge 3) 4 [2039] 2039 innerTail306_3
    (if 3 ≤ innerCandidates306_2.length then search edge 3 [2039, 2591] innerCandidates306_2 4630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan306_3

def innerTail306_1 : List Nat := 2417 :: innerTail306_2
def innerCandidates306_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates306_1_checked : candidates edge 4 2039 2417 innerTail306_2 best.bound = innerCandidates306_1 := by
  unfold candidates
  have row : edge 2417 = PrimePairCache.smallRow2417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan306_1 : scan edge (search edge 3) 4 [2039] 2039 innerTail306_1 best = best := by
  rw [innerTail306_1, scan, if_neg (by decide), innerCandidates306_1_checked]
  change scan edge (search edge 3) 4 [2039] 2039 innerTail306_2
    (if 3 ≤ innerCandidates306_1.length then search edge 3 [2039, 2417] innerCandidates306_1 4456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan306_2

def innerTail306_0 : List Nat := 2357 :: innerTail306_1
def innerCandidates306_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates306_0_checked : candidates edge 4 2039 2357 innerTail306_1 best.bound = innerCandidates306_0 := by
  unfold candidates
  have row : edge 2357 = PrimePairCache.smallRow2357.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan306_0 : scan edge (search edge 3) 4 [2039] 2039 innerTail306_0 best = best := by
  rw [innerTail306_0, scan, if_neg (by decide), innerCandidates306_0_checked]
  change scan edge (search edge 3) 4 [2039] 2039 innerTail306_1
    (if 3 ≤ innerCandidates306_0.length then search edge 3 [2039, 2357] innerCandidates306_0 4396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan306_1

theorem rootBranch306 : search edge 4 [2039] rootCandidates306 2039 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2039] 2039 innerTail306_0 best = best
  exact innerScan306_0

def rootCandidates305 : List Nat := [2281, 2341, 2791, 3739, 4507, 5101, 5227, 5281, 5419, 5689, 5701, 6121, 6247, 7027, 7213, 7309, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates305_checked : candidates edge 5 0 2029 tail306 best.bound = rootCandidates305 := by
  unfold candidates
  have row : edge 2029 = PrimePairCache.smallRow2029.contains := by funext q; rfl
  rw [row]
  decide

def innerTail305_17 : List Nat := []
theorem innerScan305_17 : scan edge (search edge 3) 4 [2029] 2029 innerTail305_17 best = best := rfl

def innerTail305_16 : List Nat := 8293 :: innerTail305_17
def innerCandidates305_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates305_16_checked : candidates edge 4 2029 8293 innerTail305_17 best.bound = innerCandidates305_16 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan305_16 : scan edge (search edge 3) 4 [2029] 2029 innerTail305_16 best = best := by
  rw [innerTail305_16, scan, if_neg (by decide), innerCandidates305_16_checked]
  change scan edge (search edge 3) 4 [2029] 2029 innerTail305_17
    (if 3 ≤ innerCandidates305_16.length then search edge 3 [2029, 8293] innerCandidates305_16 10322 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan305_17

def innerTail305_15 : List Nat := 7309 :: innerTail305_16
def innerCandidates305_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates305_15_checked : candidates edge 4 2029 7309 innerTail305_16 best.bound = innerCandidates305_15 := by
  unfold candidates
  have row : edge 7309 = SearchData.row7309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan305_15 : scan edge (search edge 3) 4 [2029] 2029 innerTail305_15 best = best := by
  rw [innerTail305_15, scan, if_neg (by decide), innerCandidates305_15_checked]
  change scan edge (search edge 3) 4 [2029] 2029 innerTail305_16
    (if 3 ≤ innerCandidates305_15.length then search edge 3 [2029, 7309] innerCandidates305_15 9338 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan305_16

def innerTail305_14 : List Nat := 7213 :: innerTail305_15
def innerCandidates305_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates305_14_checked : candidates edge 4 2029 7213 innerTail305_15 best.bound = innerCandidates305_14 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan305_14 : scan edge (search edge 3) 4 [2029] 2029 innerTail305_14 best = best := by
  rw [innerTail305_14, scan, if_neg (by decide), innerCandidates305_14_checked]
  change scan edge (search edge 3) 4 [2029] 2029 innerTail305_15
    (if 3 ≤ innerCandidates305_14.length then search edge 3 [2029, 7213] innerCandidates305_14 9242 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan305_15

def innerTail305_13 : List Nat := 7027 :: innerTail305_14
def innerCandidates305_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates305_13_checked : candidates edge 4 2029 7027 innerTail305_14 best.bound = innerCandidates305_13 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan305_13 : scan edge (search edge 3) 4 [2029] 2029 innerTail305_13 best = best := by
  rw [innerTail305_13, scan, if_neg (by decide), innerCandidates305_13_checked]
  change scan edge (search edge 3) 4 [2029] 2029 innerTail305_14
    (if 3 ≤ innerCandidates305_13.length then search edge 3 [2029, 7027] innerCandidates305_13 9056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan305_14

def innerTail305_12 : List Nat := 6247 :: innerTail305_13
def innerCandidates305_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates305_12_checked : candidates edge 4 2029 6247 innerTail305_13 best.bound = innerCandidates305_12 := by
  unfold candidates
  have row : edge 6247 = SearchData.row6247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan305_12 : scan edge (search edge 3) 4 [2029] 2029 innerTail305_12 best = best := by
  rw [innerTail305_12, scan, if_neg (by decide), innerCandidates305_12_checked]
  change scan edge (search edge 3) 4 [2029] 2029 innerTail305_13
    (if 3 ≤ innerCandidates305_12.length then search edge 3 [2029, 6247] innerCandidates305_12 8276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan305_13

def innerTail305_11 : List Nat := 6121 :: innerTail305_12
def innerCandidates305_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates305_11_checked : candidates edge 4 2029 6121 innerTail305_12 best.bound = innerCandidates305_11 := by
  unfold candidates
  have row : edge 6121 = SearchData.row6121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan305_11 : scan edge (search edge 3) 4 [2029] 2029 innerTail305_11 best = best := by
  rw [innerTail305_11, scan, if_neg (by decide), innerCandidates305_11_checked]
  change scan edge (search edge 3) 4 [2029] 2029 innerTail305_12
    (if 3 ≤ innerCandidates305_11.length then search edge 3 [2029, 6121] innerCandidates305_11 8150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan305_12

def innerTail305_10 : List Nat := 5701 :: innerTail305_11
def innerCandidates305_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates305_10_checked : candidates edge 4 2029 5701 innerTail305_11 best.bound = innerCandidates305_10 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan305_10 : scan edge (search edge 3) 4 [2029] 2029 innerTail305_10 best = best := by
  rw [innerTail305_10, scan, if_neg (by decide), innerCandidates305_10_checked]
  change scan edge (search edge 3) 4 [2029] 2029 innerTail305_11
    (if 3 ≤ innerCandidates305_10.length then search edge 3 [2029, 5701] innerCandidates305_10 7730 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan305_11

def innerTail305_9 : List Nat := 5689 :: innerTail305_10
def innerCandidates305_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates305_9_checked : candidates edge 4 2029 5689 innerTail305_10 best.bound = innerCandidates305_9 := by
  unfold candidates
  have row : edge 5689 = SearchData.row5689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan305_9 : scan edge (search edge 3) 4 [2029] 2029 innerTail305_9 best = best := by
  rw [innerTail305_9, scan, if_neg (by decide), innerCandidates305_9_checked]
  change scan edge (search edge 3) 4 [2029] 2029 innerTail305_10
    (if 3 ≤ innerCandidates305_9.length then search edge 3 [2029, 5689] innerCandidates305_9 7718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan305_10

def innerTail305_8 : List Nat := 5419 :: innerTail305_9
def innerCandidates305_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates305_8_checked : candidates edge 4 2029 5419 innerTail305_9 best.bound = innerCandidates305_8 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan305_8 : scan edge (search edge 3) 4 [2029] 2029 innerTail305_8 best = best := by
  rw [innerTail305_8, scan, if_neg (by decide), innerCandidates305_8_checked]
  change scan edge (search edge 3) 4 [2029] 2029 innerTail305_9
    (if 3 ≤ innerCandidates305_8.length then search edge 3 [2029, 5419] innerCandidates305_8 7448 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan305_9

def innerTail305_7 : List Nat := 5281 :: innerTail305_8
def innerCandidates305_7 : List Nat := [6121]
set_option maxRecDepth 4096 in
theorem innerCandidates305_7_checked : candidates edge 4 2029 5281 innerTail305_8 best.bound = innerCandidates305_7 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan305_7 : scan edge (search edge 3) 4 [2029] 2029 innerTail305_7 best = best := by
  rw [innerTail305_7, scan, if_neg (by decide), innerCandidates305_7_checked]
  change scan edge (search edge 3) 4 [2029] 2029 innerTail305_8
    (if 3 ≤ innerCandidates305_7.length then search edge 3 [2029, 5281] innerCandidates305_7 7310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan305_8

def innerTail305_6 : List Nat := 5227 :: innerTail305_7
def innerCandidates305_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates305_6_checked : candidates edge 4 2029 5227 innerTail305_7 best.bound = innerCandidates305_6 := by
  unfold candidates
  have row : edge 5227 = SearchData.row5227.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan305_6 : scan edge (search edge 3) 4 [2029] 2029 innerTail305_6 best = best := by
  rw [innerTail305_6, scan, if_neg (by decide), innerCandidates305_6_checked]
  change scan edge (search edge 3) 4 [2029] 2029 innerTail305_7
    (if 3 ≤ innerCandidates305_6.length then search edge 3 [2029, 5227] innerCandidates305_6 7256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan305_7

def innerTail305_5 : List Nat := 5101 :: innerTail305_6
def innerCandidates305_5 : List Nat := [8293]
set_option maxRecDepth 4096 in
theorem innerCandidates305_5_checked : candidates edge 4 2029 5101 innerTail305_6 best.bound = innerCandidates305_5 := by
  unfold candidates
  have row : edge 5101 = SearchData.row5101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan305_5 : scan edge (search edge 3) 4 [2029] 2029 innerTail305_5 best = best := by
  rw [innerTail305_5, scan, if_neg (by decide), innerCandidates305_5_checked]
  change scan edge (search edge 3) 4 [2029] 2029 innerTail305_6
    (if 3 ≤ innerCandidates305_5.length then search edge 3 [2029, 5101] innerCandidates305_5 7130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan305_6

def innerTail305_4 : List Nat := 4507 :: innerTail305_5
def innerCandidates305_4 : List Nat := [5419]
set_option maxRecDepth 4096 in
theorem innerCandidates305_4_checked : candidates edge 4 2029 4507 innerTail305_5 best.bound = innerCandidates305_4 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan305_4 : scan edge (search edge 3) 4 [2029] 2029 innerTail305_4 best = best := by
  rw [innerTail305_4, scan, if_neg (by decide), innerCandidates305_4_checked]
  change scan edge (search edge 3) 4 [2029] 2029 innerTail305_5
    (if 3 ≤ innerCandidates305_4.length then search edge 3 [2029, 4507] innerCandidates305_4 6536 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan305_5

def innerTail305_3 : List Nat := 3739 :: innerTail305_4
def innerCandidates305_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates305_3_checked : candidates edge 4 2029 3739 innerTail305_4 best.bound = innerCandidates305_3 := by
  unfold candidates
  have row : edge 3739 = SearchData.row3739.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan305_3 : scan edge (search edge 3) 4 [2029] 2029 innerTail305_3 best = best := by
  rw [innerTail305_3, scan, if_neg (by decide), innerCandidates305_3_checked]
  change scan edge (search edge 3) 4 [2029] 2029 innerTail305_4
    (if 3 ≤ innerCandidates305_3.length then search edge 3 [2029, 3739] innerCandidates305_3 5768 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan305_4

def innerTail305_2 : List Nat := 2791 :: innerTail305_3
def innerCandidates305_2 : List Nat := [5281]
set_option maxRecDepth 4096 in
theorem innerCandidates305_2_checked : candidates edge 4 2029 2791 innerTail305_3 best.bound = innerCandidates305_2 := by
  unfold candidates
  have row : edge 2791 = SearchData.row2791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan305_2 : scan edge (search edge 3) 4 [2029] 2029 innerTail305_2 best = best := by
  rw [innerTail305_2, scan, if_neg (by decide), innerCandidates305_2_checked]
  change scan edge (search edge 3) 4 [2029] 2029 innerTail305_3
    (if 3 ≤ innerCandidates305_2.length then search edge 3 [2029, 2791] innerCandidates305_2 4820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan305_3

def innerTail305_1 : List Nat := 2341 :: innerTail305_2
def innerCandidates305_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates305_1_checked : candidates edge 4 2029 2341 innerTail305_2 best.bound = innerCandidates305_1 := by
  unfold candidates
  have row : edge 2341 = PrimePairCache.smallRow2341.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan305_1 : scan edge (search edge 3) 4 [2029] 2029 innerTail305_1 best = best := by
  rw [innerTail305_1, scan, if_neg (by decide), innerCandidates305_1_checked]
  change scan edge (search edge 3) 4 [2029] 2029 innerTail305_2
    (if 3 ≤ innerCandidates305_1.length then search edge 3 [2029, 2341] innerCandidates305_1 4370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan305_2

def innerTail305_0 : List Nat := 2281 :: innerTail305_1
def innerCandidates305_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates305_0_checked : candidates edge 4 2029 2281 innerTail305_1 best.bound = innerCandidates305_0 := by
  unfold candidates
  have row : edge 2281 = PrimePairCache.smallRow2281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan305_0 : scan edge (search edge 3) 4 [2029] 2029 innerTail305_0 best = best := by
  rw [innerTail305_0, scan, if_neg (by decide), innerCandidates305_0_checked]
  change scan edge (search edge 3) 4 [2029] 2029 innerTail305_1
    (if 3 ≤ innerCandidates305_0.length then search edge 3 [2029, 2281] innerCandidates305_0 4310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan305_1

theorem rootBranch305 : search edge 4 [2029] rootCandidates305 2029 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2029] 2029 innerTail305_0 best = best
  exact innerScan305_0

def rootCandidates304 : List Nat := [3119, 3761, 4259, 5153, 5483, 6203, 6701, 6911, 7253, 7457, 7703]
set_option maxRecDepth 4096 in
theorem rootCandidates304_checked : candidates edge 5 0 2027 tail305 best.bound = rootCandidates304 := by
  unfold candidates
  have row : edge 2027 = PrimePairCache.smallRow2027.contains := by funext q; rfl
  rw [row]
  decide

def innerTail304_11 : List Nat := []
theorem innerScan304_11 : scan edge (search edge 3) 4 [2027] 2027 innerTail304_11 best = best := rfl

def innerTail304_10 : List Nat := 7703 :: innerTail304_11
def innerCandidates304_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates304_10_checked : candidates edge 4 2027 7703 innerTail304_11 best.bound = innerCandidates304_10 := by
  unfold candidates
  have row : edge 7703 = SearchData.row7703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan304_10 : scan edge (search edge 3) 4 [2027] 2027 innerTail304_10 best = best := by
  rw [innerTail304_10, scan, if_neg (by decide), innerCandidates304_10_checked]
  change scan edge (search edge 3) 4 [2027] 2027 innerTail304_11
    (if 3 ≤ innerCandidates304_10.length then search edge 3 [2027, 7703] innerCandidates304_10 9730 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan304_11

def innerTail304_9 : List Nat := 7457 :: innerTail304_10
def innerCandidates304_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates304_9_checked : candidates edge 4 2027 7457 innerTail304_10 best.bound = innerCandidates304_9 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan304_9 : scan edge (search edge 3) 4 [2027] 2027 innerTail304_9 best = best := by
  rw [innerTail304_9, scan, if_neg (by decide), innerCandidates304_9_checked]
  change scan edge (search edge 3) 4 [2027] 2027 innerTail304_10
    (if 3 ≤ innerCandidates304_9.length then search edge 3 [2027, 7457] innerCandidates304_9 9484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan304_10

def innerTail304_8 : List Nat := 7253 :: innerTail304_9
def innerCandidates304_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates304_8_checked : candidates edge 4 2027 7253 innerTail304_9 best.bound = innerCandidates304_8 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan304_8 : scan edge (search edge 3) 4 [2027] 2027 innerTail304_8 best = best := by
  rw [innerTail304_8, scan, if_neg (by decide), innerCandidates304_8_checked]
  change scan edge (search edge 3) 4 [2027] 2027 innerTail304_9
    (if 3 ≤ innerCandidates304_8.length then search edge 3 [2027, 7253] innerCandidates304_8 9280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan304_9

def innerTail304_7 : List Nat := 6911 :: innerTail304_8
def innerCandidates304_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates304_7_checked : candidates edge 4 2027 6911 innerTail304_8 best.bound = innerCandidates304_7 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan304_7 : scan edge (search edge 3) 4 [2027] 2027 innerTail304_7 best = best := by
  rw [innerTail304_7, scan, if_neg (by decide), innerCandidates304_7_checked]
  change scan edge (search edge 3) 4 [2027] 2027 innerTail304_8
    (if 3 ≤ innerCandidates304_7.length then search edge 3 [2027, 6911] innerCandidates304_7 8938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan304_8

def innerTail304_6 : List Nat := 6701 :: innerTail304_7
def innerCandidates304_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates304_6_checked : candidates edge 4 2027 6701 innerTail304_7 best.bound = innerCandidates304_6 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan304_6 : scan edge (search edge 3) 4 [2027] 2027 innerTail304_6 best = best := by
  rw [innerTail304_6, scan, if_neg (by decide), innerCandidates304_6_checked]
  change scan edge (search edge 3) 4 [2027] 2027 innerTail304_7
    (if 3 ≤ innerCandidates304_6.length then search edge 3 [2027, 6701] innerCandidates304_6 8728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan304_7

def innerTail304_5 : List Nat := 6203 :: innerTail304_6
def innerCandidates304_5 : List Nat := [7253]
set_option maxRecDepth 4096 in
theorem innerCandidates304_5_checked : candidates edge 4 2027 6203 innerTail304_6 best.bound = innerCandidates304_5 := by
  unfold candidates
  have row : edge 6203 = SearchData.row6203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan304_5 : scan edge (search edge 3) 4 [2027] 2027 innerTail304_5 best = best := by
  rw [innerTail304_5, scan, if_neg (by decide), innerCandidates304_5_checked]
  change scan edge (search edge 3) 4 [2027] 2027 innerTail304_6
    (if 3 ≤ innerCandidates304_5.length then search edge 3 [2027, 6203] innerCandidates304_5 8230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan304_6

def innerTail304_4 : List Nat := 5483 :: innerTail304_5
def innerCandidates304_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates304_4_checked : candidates edge 4 2027 5483 innerTail304_5 best.bound = innerCandidates304_4 := by
  unfold candidates
  have row : edge 5483 = SearchData.row5483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan304_4 : scan edge (search edge 3) 4 [2027] 2027 innerTail304_4 best = best := by
  rw [innerTail304_4, scan, if_neg (by decide), innerCandidates304_4_checked]
  change scan edge (search edge 3) 4 [2027] 2027 innerTail304_5
    (if 3 ≤ innerCandidates304_4.length then search edge 3 [2027, 5483] innerCandidates304_4 7510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan304_5

def innerTail304_3 : List Nat := 5153 :: innerTail304_4
def innerCandidates304_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates304_3_checked : candidates edge 4 2027 5153 innerTail304_4 best.bound = innerCandidates304_3 := by
  unfold candidates
  have row : edge 5153 = SearchData.row5153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan304_3 : scan edge (search edge 3) 4 [2027] 2027 innerTail304_3 best = best := by
  rw [innerTail304_3, scan, if_neg (by decide), innerCandidates304_3_checked]
  change scan edge (search edge 3) 4 [2027] 2027 innerTail304_4
    (if 3 ≤ innerCandidates304_3.length then search edge 3 [2027, 5153] innerCandidates304_3 7180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan304_4

def innerTail304_2 : List Nat := 4259 :: innerTail304_3
def innerCandidates304_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates304_2_checked : candidates edge 4 2027 4259 innerTail304_3 best.bound = innerCandidates304_2 := by
  unfold candidates
  have row : edge 4259 = SearchData.row4259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan304_2 : scan edge (search edge 3) 4 [2027] 2027 innerTail304_2 best = best := by
  rw [innerTail304_2, scan, if_neg (by decide), innerCandidates304_2_checked]
  change scan edge (search edge 3) 4 [2027] 2027 innerTail304_3
    (if 3 ≤ innerCandidates304_2.length then search edge 3 [2027, 4259] innerCandidates304_2 6286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan304_3

def innerTail304_1 : List Nat := 3761 :: innerTail304_2
def innerCandidates304_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates304_1_checked : candidates edge 4 2027 3761 innerTail304_2 best.bound = innerCandidates304_1 := by
  unfold candidates
  have row : edge 3761 = SearchData.row3761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan304_1 : scan edge (search edge 3) 4 [2027] 2027 innerTail304_1 best = best := by
  rw [innerTail304_1, scan, if_neg (by decide), innerCandidates304_1_checked]
  change scan edge (search edge 3) 4 [2027] 2027 innerTail304_2
    (if 3 ≤ innerCandidates304_1.length then search edge 3 [2027, 3761] innerCandidates304_1 5788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan304_2

def innerTail304_0 : List Nat := 3119 :: innerTail304_1
def innerCandidates304_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates304_0_checked : candidates edge 4 2027 3119 innerTail304_1 best.bound = innerCandidates304_0 := by
  unfold candidates
  have row : edge 3119 = SearchData.row3119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan304_0 : scan edge (search edge 3) 4 [2027] 2027 innerTail304_0 best = best := by
  rw [innerTail304_0, scan, if_neg (by decide), innerCandidates304_0_checked]
  change scan edge (search edge 3) 4 [2027] 2027 innerTail304_1
    (if 3 ≤ innerCandidates304_0.length then search edge 3 [2027, 3119] innerCandidates304_0 5146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan304_1

theorem rootBranch304 : search edge 4 [2027] rootCandidates304 2027 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2027] 2027 innerTail304_0 best = best
  exact innerScan304_0

def rootCandidates303 : List Nat := [2221, 2617, 2689, 3313, 3919, 4339, 4423, 4621, 5653, 6661, 6967, 7237, 7603, 8311, 8317]
set_option maxRecDepth 4096 in
theorem rootCandidates303_checked : candidates edge 5 0 2017 tail304 best.bound = rootCandidates303 := by
  unfold candidates
  have row : edge 2017 = PrimePairCache.smallRow2017.contains := by funext q; rfl
  rw [row]
  decide

def innerTail303_15 : List Nat := []
theorem innerScan303_15 : scan edge (search edge 3) 4 [2017] 2017 innerTail303_15 best = best := rfl

def innerTail303_14 : List Nat := 8317 :: innerTail303_15
def innerCandidates303_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates303_14_checked : candidates edge 4 2017 8317 innerTail303_15 best.bound = innerCandidates303_14 := by
  unfold candidates
  have row : edge 8317 = SearchData.row8317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan303_14 : scan edge (search edge 3) 4 [2017] 2017 innerTail303_14 best = best := by
  rw [innerTail303_14, scan, if_neg (by decide), innerCandidates303_14_checked]
  change scan edge (search edge 3) 4 [2017] 2017 innerTail303_15
    (if 3 ≤ innerCandidates303_14.length then search edge 3 [2017, 8317] innerCandidates303_14 10334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan303_15

def innerTail303_13 : List Nat := 8311 :: innerTail303_14
def innerCandidates303_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates303_13_checked : candidates edge 4 2017 8311 innerTail303_14 best.bound = innerCandidates303_13 := by
  unfold candidates
  have row : edge 8311 = SearchData.row8311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan303_13 : scan edge (search edge 3) 4 [2017] 2017 innerTail303_13 best = best := by
  rw [innerTail303_13, scan, if_neg (by decide), innerCandidates303_13_checked]
  change scan edge (search edge 3) 4 [2017] 2017 innerTail303_14
    (if 3 ≤ innerCandidates303_13.length then search edge 3 [2017, 8311] innerCandidates303_13 10328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan303_14

def innerTail303_12 : List Nat := 7603 :: innerTail303_13
def innerCandidates303_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates303_12_checked : candidates edge 4 2017 7603 innerTail303_13 best.bound = innerCandidates303_12 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan303_12 : scan edge (search edge 3) 4 [2017] 2017 innerTail303_12 best = best := by
  rw [innerTail303_12, scan, if_neg (by decide), innerCandidates303_12_checked]
  change scan edge (search edge 3) 4 [2017] 2017 innerTail303_13
    (if 3 ≤ innerCandidates303_12.length then search edge 3 [2017, 7603] innerCandidates303_12 9620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan303_13

def innerTail303_11 : List Nat := 7237 :: innerTail303_12
def innerCandidates303_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates303_11_checked : candidates edge 4 2017 7237 innerTail303_12 best.bound = innerCandidates303_11 := by
  unfold candidates
  have row : edge 7237 = SearchData.row7237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan303_11 : scan edge (search edge 3) 4 [2017] 2017 innerTail303_11 best = best := by
  rw [innerTail303_11, scan, if_neg (by decide), innerCandidates303_11_checked]
  change scan edge (search edge 3) 4 [2017] 2017 innerTail303_12
    (if 3 ≤ innerCandidates303_11.length then search edge 3 [2017, 7237] innerCandidates303_11 9254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan303_12

def innerTail303_10 : List Nat := 6967 :: innerTail303_11
def innerCandidates303_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates303_10_checked : candidates edge 4 2017 6967 innerTail303_11 best.bound = innerCandidates303_10 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan303_10 : scan edge (search edge 3) 4 [2017] 2017 innerTail303_10 best = best := by
  rw [innerTail303_10, scan, if_neg (by decide), innerCandidates303_10_checked]
  change scan edge (search edge 3) 4 [2017] 2017 innerTail303_11
    (if 3 ≤ innerCandidates303_10.length then search edge 3 [2017, 6967] innerCandidates303_10 8984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan303_11

def innerTail303_9 : List Nat := 6661 :: innerTail303_10
def innerCandidates303_9 : List Nat := [8311]
set_option maxRecDepth 4096 in
theorem innerCandidates303_9_checked : candidates edge 4 2017 6661 innerTail303_10 best.bound = innerCandidates303_9 := by
  unfold candidates
  have row : edge 6661 = SearchData.row6661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan303_9 : scan edge (search edge 3) 4 [2017] 2017 innerTail303_9 best = best := by
  rw [innerTail303_9, scan, if_neg (by decide), innerCandidates303_9_checked]
  change scan edge (search edge 3) 4 [2017] 2017 innerTail303_10
    (if 3 ≤ innerCandidates303_9.length then search edge 3 [2017, 6661] innerCandidates303_9 8678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan303_10

def innerTail303_8 : List Nat := 5653 :: innerTail303_9
def innerCandidates303_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates303_8_checked : candidates edge 4 2017 5653 innerTail303_9 best.bound = innerCandidates303_8 := by
  unfold candidates
  have row : edge 5653 = SearchData.row5653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan303_8 : scan edge (search edge 3) 4 [2017] 2017 innerTail303_8 best = best := by
  rw [innerTail303_8, scan, if_neg (by decide), innerCandidates303_8_checked]
  change scan edge (search edge 3) 4 [2017] 2017 innerTail303_9
    (if 3 ≤ innerCandidates303_8.length then search edge 3 [2017, 5653] innerCandidates303_8 7670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan303_9

def innerTail303_7 : List Nat := 4621 :: innerTail303_8
def innerCandidates303_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates303_7_checked : candidates edge 4 2017 4621 innerTail303_8 best.bound = innerCandidates303_7 := by
  unfold candidates
  have row : edge 4621 = SearchData.row4621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan303_7 : scan edge (search edge 3) 4 [2017] 2017 innerTail303_7 best = best := by
  rw [innerTail303_7, scan, if_neg (by decide), innerCandidates303_7_checked]
  change scan edge (search edge 3) 4 [2017] 2017 innerTail303_8
    (if 3 ≤ innerCandidates303_7.length then search edge 3 [2017, 4621] innerCandidates303_7 6638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan303_8

def innerTail303_6 : List Nat := 4423 :: innerTail303_7
def innerCandidates303_6 : List Nat := [8317]
set_option maxRecDepth 4096 in
theorem innerCandidates303_6_checked : candidates edge 4 2017 4423 innerTail303_7 best.bound = innerCandidates303_6 := by
  unfold candidates
  have row : edge 4423 = SearchData.row4423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan303_6 : scan edge (search edge 3) 4 [2017] 2017 innerTail303_6 best = best := by
  rw [innerTail303_6, scan, if_neg (by decide), innerCandidates303_6_checked]
  change scan edge (search edge 3) 4 [2017] 2017 innerTail303_7
    (if 3 ≤ innerCandidates303_6.length then search edge 3 [2017, 4423] innerCandidates303_6 6440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan303_7

def innerTail303_5 : List Nat := 4339 :: innerTail303_6
def innerCandidates303_5 : List Nat := [7603, 8317]
set_option maxRecDepth 4096 in
theorem innerCandidates303_5_checked : candidates edge 4 2017 4339 innerTail303_6 best.bound = innerCandidates303_5 := by
  unfold candidates
  have row : edge 4339 = SearchData.row4339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan303_5 : scan edge (search edge 3) 4 [2017] 2017 innerTail303_5 best = best := by
  rw [innerTail303_5, scan, if_neg (by decide), innerCandidates303_5_checked]
  change scan edge (search edge 3) 4 [2017] 2017 innerTail303_6
    (if 3 ≤ innerCandidates303_5.length then search edge 3 [2017, 4339] innerCandidates303_5 6356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan303_6

def innerTail303_4 : List Nat := 3919 :: innerTail303_5
def innerCandidates303_4 : List Nat := [8317]
set_option maxRecDepth 4096 in
theorem innerCandidates303_4_checked : candidates edge 4 2017 3919 innerTail303_5 best.bound = innerCandidates303_4 := by
  unfold candidates
  have row : edge 3919 = SearchData.row3919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan303_4 : scan edge (search edge 3) 4 [2017] 2017 innerTail303_4 best = best := by
  rw [innerTail303_4, scan, if_neg (by decide), innerCandidates303_4_checked]
  change scan edge (search edge 3) 4 [2017] 2017 innerTail303_5
    (if 3 ≤ innerCandidates303_4.length then search edge 3 [2017, 3919] innerCandidates303_4 5936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan303_5

def innerTail303_3 : List Nat := 3313 :: innerTail303_4
def innerCandidates303_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates303_3_checked : candidates edge 4 2017 3313 innerTail303_4 best.bound = innerCandidates303_3 := by
  unfold candidates
  have row : edge 3313 = SearchData.row3313.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan303_3 : scan edge (search edge 3) 4 [2017] 2017 innerTail303_3 best = best := by
  rw [innerTail303_3, scan, if_neg (by decide), innerCandidates303_3_checked]
  change scan edge (search edge 3) 4 [2017] 2017 innerTail303_4
    (if 3 ≤ innerCandidates303_3.length then search edge 3 [2017, 3313] innerCandidates303_3 5330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan303_4

def innerTail303_2 : List Nat := 2689 :: innerTail303_3
def innerCandidates303_2 : List Nat := [4423, 8317]
set_option maxRecDepth 4096 in
theorem innerCandidates303_2_checked : candidates edge 4 2017 2689 innerTail303_3 best.bound = innerCandidates303_2 := by
  unfold candidates
  have row : edge 2689 = SearchData.row2689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan303_2 : scan edge (search edge 3) 4 [2017] 2017 innerTail303_2 best = best := by
  rw [innerTail303_2, scan, if_neg (by decide), innerCandidates303_2_checked]
  change scan edge (search edge 3) 4 [2017] 2017 innerTail303_3
    (if 3 ≤ innerCandidates303_2.length then search edge 3 [2017, 2689] innerCandidates303_2 4706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan303_3

def innerTail303_1 : List Nat := 2617 :: innerTail303_2
def innerCandidates303_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates303_1_checked : candidates edge 4 2017 2617 innerTail303_2 best.bound = innerCandidates303_1 := by
  unfold candidates
  have row : edge 2617 = PrimePairCache.smallRow2617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan303_1 : scan edge (search edge 3) 4 [2017] 2017 innerTail303_1 best = best := by
  rw [innerTail303_1, scan, if_neg (by decide), innerCandidates303_1_checked]
  change scan edge (search edge 3) 4 [2017] 2017 innerTail303_2
    (if 3 ≤ innerCandidates303_1.length then search edge 3 [2017, 2617] innerCandidates303_1 4634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan303_2

def innerTail303_0 : List Nat := 2221 :: innerTail303_1
def innerCandidates303_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates303_0_checked : candidates edge 4 2017 2221 innerTail303_1 best.bound = innerCandidates303_0 := by
  unfold candidates
  have row : edge 2221 = PrimePairCache.smallRow2221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan303_0 : scan edge (search edge 3) 4 [2017] 2017 innerTail303_0 best = best := by
  rw [innerTail303_0, scan, if_neg (by decide), innerCandidates303_0_checked]
  change scan edge (search edge 3) 4 [2017] 2017 innerTail303_1
    (if 3 ≤ innerCandidates303_0.length then search edge 3 [2017, 2221] innerCandidates303_0 4238 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan303_1

theorem rootBranch303 : search edge 4 [2017] rootCandidates303 2017 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2017] 2017 innerTail303_0 best = best
  exact innerScan303_0

end ProjectEulerSolutions.PrimePairUpper.Search
