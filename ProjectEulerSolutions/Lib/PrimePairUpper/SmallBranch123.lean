import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates310 : List Nat := [2699, 3323, 4127, 4349, 4691, 4943, 5153, 5441, 5657, 5801, 5807, 6689, 6791, 7109, 7229, 7793]
set_option maxRecDepth 4096 in
theorem rootCandidates310_checked : candidates edge 5 0 2081 tail311 best.bound = rootCandidates310 := by
  unfold candidates
  have row : edge 2081 = PrimePairCache.smallRow2081.contains := by funext q; rfl
  rw [row]
  decide

def innerTail310_16 : List Nat := []
theorem innerScan310_16 : scan edge (search edge 3) 4 [2081] 2081 innerTail310_16 best = best := rfl

def innerTail310_15 : List Nat := 7793 :: innerTail310_16
def innerCandidates310_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates310_15_checked : candidates edge 4 2081 7793 innerTail310_16 best.bound = innerCandidates310_15 := by
  unfold candidates
  have row : edge 7793 = SearchData.row7793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan310_15 : scan edge (search edge 3) 4 [2081] 2081 innerTail310_15 best = best := by
  rw [innerTail310_15, scan, if_neg (by decide), innerCandidates310_15_checked]
  change scan edge (search edge 3) 4 [2081] 2081 innerTail310_16
    (if 3 ≤ innerCandidates310_15.length then search edge 3 [2081, 7793] innerCandidates310_15 9874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan310_16

def innerTail310_14 : List Nat := 7229 :: innerTail310_15
def innerCandidates310_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates310_14_checked : candidates edge 4 2081 7229 innerTail310_15 best.bound = innerCandidates310_14 := by
  unfold candidates
  have row : edge 7229 = SearchData.row7229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan310_14 : scan edge (search edge 3) 4 [2081] 2081 innerTail310_14 best = best := by
  rw [innerTail310_14, scan, if_neg (by decide), innerCandidates310_14_checked]
  change scan edge (search edge 3) 4 [2081] 2081 innerTail310_15
    (if 3 ≤ innerCandidates310_14.length then search edge 3 [2081, 7229] innerCandidates310_14 9310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan310_15

def innerTail310_13 : List Nat := 7109 :: innerTail310_14
def innerCandidates310_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates310_13_checked : candidates edge 4 2081 7109 innerTail310_14 best.bound = innerCandidates310_13 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan310_13 : scan edge (search edge 3) 4 [2081] 2081 innerTail310_13 best = best := by
  rw [innerTail310_13, scan, if_neg (by decide), innerCandidates310_13_checked]
  change scan edge (search edge 3) 4 [2081] 2081 innerTail310_14
    (if 3 ≤ innerCandidates310_13.length then search edge 3 [2081, 7109] innerCandidates310_13 9190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan310_14

def innerTail310_12 : List Nat := 6791 :: innerTail310_13
def innerCandidates310_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates310_12_checked : candidates edge 4 2081 6791 innerTail310_13 best.bound = innerCandidates310_12 := by
  unfold candidates
  have row : edge 6791 = SearchData.row6791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan310_12 : scan edge (search edge 3) 4 [2081] 2081 innerTail310_12 best = best := by
  rw [innerTail310_12, scan, if_neg (by decide), innerCandidates310_12_checked]
  change scan edge (search edge 3) 4 [2081] 2081 innerTail310_13
    (if 3 ≤ innerCandidates310_12.length then search edge 3 [2081, 6791] innerCandidates310_12 8872 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan310_13

def innerTail310_11 : List Nat := 6689 :: innerTail310_12
def innerCandidates310_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates310_11_checked : candidates edge 4 2081 6689 innerTail310_12 best.bound = innerCandidates310_11 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan310_11 : scan edge (search edge 3) 4 [2081] 2081 innerTail310_11 best = best := by
  rw [innerTail310_11, scan, if_neg (by decide), innerCandidates310_11_checked]
  change scan edge (search edge 3) 4 [2081] 2081 innerTail310_12
    (if 3 ≤ innerCandidates310_11.length then search edge 3 [2081, 6689] innerCandidates310_11 8770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan310_12

def innerTail310_10 : List Nat := 5807 :: innerTail310_11
def innerCandidates310_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates310_10_checked : candidates edge 4 2081 5807 innerTail310_11 best.bound = innerCandidates310_10 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan310_10 : scan edge (search edge 3) 4 [2081] 2081 innerTail310_10 best = best := by
  rw [innerTail310_10, scan, if_neg (by decide), innerCandidates310_10_checked]
  change scan edge (search edge 3) 4 [2081] 2081 innerTail310_11
    (if 3 ≤ innerCandidates310_10.length then search edge 3 [2081, 5807] innerCandidates310_10 7888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan310_11

def innerTail310_9 : List Nat := 5801 :: innerTail310_10
def innerCandidates310_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates310_9_checked : candidates edge 4 2081 5801 innerTail310_10 best.bound = innerCandidates310_9 := by
  unfold candidates
  have row : edge 5801 = SearchData.row5801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan310_9 : scan edge (search edge 3) 4 [2081] 2081 innerTail310_9 best = best := by
  rw [innerTail310_9, scan, if_neg (by decide), innerCandidates310_9_checked]
  change scan edge (search edge 3) 4 [2081] 2081 innerTail310_10
    (if 3 ≤ innerCandidates310_9.length then search edge 3 [2081, 5801] innerCandidates310_9 7882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan310_10

def innerTail310_8 : List Nat := 5657 :: innerTail310_9
def innerCandidates310_8 : List Nat := [6791]
set_option maxRecDepth 4096 in
theorem innerCandidates310_8_checked : candidates edge 4 2081 5657 innerTail310_9 best.bound = innerCandidates310_8 := by
  unfold candidates
  have row : edge 5657 = SearchData.row5657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan310_8 : scan edge (search edge 3) 4 [2081] 2081 innerTail310_8 best = best := by
  rw [innerTail310_8, scan, if_neg (by decide), innerCandidates310_8_checked]
  change scan edge (search edge 3) 4 [2081] 2081 innerTail310_9
    (if 3 ≤ innerCandidates310_8.length then search edge 3 [2081, 5657] innerCandidates310_8 7738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan310_9

def innerTail310_7 : List Nat := 5441 :: innerTail310_8
def innerCandidates310_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates310_7_checked : candidates edge 4 2081 5441 innerTail310_8 best.bound = innerCandidates310_7 := by
  unfold candidates
  have row : edge 5441 = SearchData.row5441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan310_7 : scan edge (search edge 3) 4 [2081] 2081 innerTail310_7 best = best := by
  rw [innerTail310_7, scan, if_neg (by decide), innerCandidates310_7_checked]
  change scan edge (search edge 3) 4 [2081] 2081 innerTail310_8
    (if 3 ≤ innerCandidates310_7.length then search edge 3 [2081, 5441] innerCandidates310_7 7522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan310_8

def innerTail310_6 : List Nat := 5153 :: innerTail310_7
def innerCandidates310_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates310_6_checked : candidates edge 4 2081 5153 innerTail310_7 best.bound = innerCandidates310_6 := by
  unfold candidates
  have row : edge 5153 = SearchData.row5153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan310_6 : scan edge (search edge 3) 4 [2081] 2081 innerTail310_6 best = best := by
  rw [innerTail310_6, scan, if_neg (by decide), innerCandidates310_6_checked]
  change scan edge (search edge 3) 4 [2081] 2081 innerTail310_7
    (if 3 ≤ innerCandidates310_6.length then search edge 3 [2081, 5153] innerCandidates310_6 7234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan310_7

def innerTail310_5 : List Nat := 4943 :: innerTail310_6
def innerCandidates310_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates310_5_checked : candidates edge 4 2081 4943 innerTail310_6 best.bound = innerCandidates310_5 := by
  unfold candidates
  have row : edge 4943 = SearchData.row4943.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan310_5 : scan edge (search edge 3) 4 [2081] 2081 innerTail310_5 best = best := by
  rw [innerTail310_5, scan, if_neg (by decide), innerCandidates310_5_checked]
  change scan edge (search edge 3) 4 [2081] 2081 innerTail310_6
    (if 3 ≤ innerCandidates310_5.length then search edge 3 [2081, 4943] innerCandidates310_5 7024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan310_6

def innerTail310_4 : List Nat := 4691 :: innerTail310_5
def innerCandidates310_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates310_4_checked : candidates edge 4 2081 4691 innerTail310_5 best.bound = innerCandidates310_4 := by
  unfold candidates
  have row : edge 4691 = SearchData.row4691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan310_4 : scan edge (search edge 3) 4 [2081] 2081 innerTail310_4 best = best := by
  rw [innerTail310_4, scan, if_neg (by decide), innerCandidates310_4_checked]
  change scan edge (search edge 3) 4 [2081] 2081 innerTail310_5
    (if 3 ≤ innerCandidates310_4.length then search edge 3 [2081, 4691] innerCandidates310_4 6772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan310_5

def innerTail310_3 : List Nat := 4349 :: innerTail310_4
def innerCandidates310_3 : List Nat := [7793]
set_option maxRecDepth 4096 in
theorem innerCandidates310_3_checked : candidates edge 4 2081 4349 innerTail310_4 best.bound = innerCandidates310_3 := by
  unfold candidates
  have row : edge 4349 = SearchData.row4349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan310_3 : scan edge (search edge 3) 4 [2081] 2081 innerTail310_3 best = best := by
  rw [innerTail310_3, scan, if_neg (by decide), innerCandidates310_3_checked]
  change scan edge (search edge 3) 4 [2081] 2081 innerTail310_4
    (if 3 ≤ innerCandidates310_3.length then search edge 3 [2081, 4349] innerCandidates310_3 6430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan310_4

def innerTail310_2 : List Nat := 4127 :: innerTail310_3
def innerCandidates310_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates310_2_checked : candidates edge 4 2081 4127 innerTail310_3 best.bound = innerCandidates310_2 := by
  unfold candidates
  have row : edge 4127 = SearchData.row4127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan310_2 : scan edge (search edge 3) 4 [2081] 2081 innerTail310_2 best = best := by
  rw [innerTail310_2, scan, if_neg (by decide), innerCandidates310_2_checked]
  change scan edge (search edge 3) 4 [2081] 2081 innerTail310_3
    (if 3 ≤ innerCandidates310_2.length then search edge 3 [2081, 4127] innerCandidates310_2 6208 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan310_3

def innerTail310_1 : List Nat := 3323 :: innerTail310_2
def innerCandidates310_1 : List Nat := [4349, 7229]
set_option maxRecDepth 4096 in
theorem innerCandidates310_1_checked : candidates edge 4 2081 3323 innerTail310_2 best.bound = innerCandidates310_1 := by
  unfold candidates
  have row : edge 3323 = SearchData.row3323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan310_1 : scan edge (search edge 3) 4 [2081] 2081 innerTail310_1 best = best := by
  rw [innerTail310_1, scan, if_neg (by decide), innerCandidates310_1_checked]
  change scan edge (search edge 3) 4 [2081] 2081 innerTail310_2
    (if 3 ≤ innerCandidates310_1.length then search edge 3 [2081, 3323] innerCandidates310_1 5404 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan310_2

def innerTail310_0 : List Nat := 2699 :: innerTail310_1
def innerCandidates310_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates310_0_checked : candidates edge 4 2081 2699 innerTail310_1 best.bound = innerCandidates310_0 := by
  unfold candidates
  have row : edge 2699 = SearchData.row2699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan310_0 : scan edge (search edge 3) 4 [2081] 2081 innerTail310_0 best = best := by
  rw [innerTail310_0, scan, if_neg (by decide), innerCandidates310_0_checked]
  change scan edge (search edge 3) 4 [2081] 2081 innerTail310_1
    (if 3 ≤ innerCandidates310_0.length then search edge 3 [2081, 2699] innerCandidates310_0 4780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan310_1

theorem rootBranch310 : search edge 4 [2081] rootCandidates310 2081 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2081] 2081 innerTail310_0 best = best
  exact innerScan310_0

def rootCandidates309 : List Nat := [2297, 2339, 4397, 4637, 4691, 5021, 6827, 6959, 6983, 7121, 7253, 7283, 7757, 7877, 8219, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates309_checked : candidates edge 5 0 2069 tail310 best.bound = rootCandidates309 := by
  unfold candidates
  have row : edge 2069 = PrimePairCache.smallRow2069.contains := by funext q; rfl
  rw [row]
  decide

def innerTail309_16 : List Nat := []
theorem innerScan309_16 : scan edge (search edge 3) 4 [2069] 2069 innerTail309_16 best = best := rfl

def innerTail309_15 : List Nat := 8369 :: innerTail309_16
def innerCandidates309_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates309_15_checked : candidates edge 4 2069 8369 innerTail309_16 best.bound = innerCandidates309_15 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan309_15 : scan edge (search edge 3) 4 [2069] 2069 innerTail309_15 best = best := by
  rw [innerTail309_15, scan, if_neg (by decide), innerCandidates309_15_checked]
  change scan edge (search edge 3) 4 [2069] 2069 innerTail309_16
    (if 3 ≤ innerCandidates309_15.length then search edge 3 [2069, 8369] innerCandidates309_15 10438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan309_16

def innerTail309_14 : List Nat := 8219 :: innerTail309_15
def innerCandidates309_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates309_14_checked : candidates edge 4 2069 8219 innerTail309_15 best.bound = innerCandidates309_14 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan309_14 : scan edge (search edge 3) 4 [2069] 2069 innerTail309_14 best = best := by
  rw [innerTail309_14, scan, if_neg (by decide), innerCandidates309_14_checked]
  change scan edge (search edge 3) 4 [2069] 2069 innerTail309_15
    (if 3 ≤ innerCandidates309_14.length then search edge 3 [2069, 8219] innerCandidates309_14 10288 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan309_15

def innerTail309_13 : List Nat := 7877 :: innerTail309_14
def innerCandidates309_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates309_13_checked : candidates edge 4 2069 7877 innerTail309_14 best.bound = innerCandidates309_13 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan309_13 : scan edge (search edge 3) 4 [2069] 2069 innerTail309_13 best = best := by
  rw [innerTail309_13, scan, if_neg (by decide), innerCandidates309_13_checked]
  change scan edge (search edge 3) 4 [2069] 2069 innerTail309_14
    (if 3 ≤ innerCandidates309_13.length then search edge 3 [2069, 7877] innerCandidates309_13 9946 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan309_14

def innerTail309_12 : List Nat := 7757 :: innerTail309_13
def innerCandidates309_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates309_12_checked : candidates edge 4 2069 7757 innerTail309_13 best.bound = innerCandidates309_12 := by
  unfold candidates
  have row : edge 7757 = SearchData.row7757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan309_12 : scan edge (search edge 3) 4 [2069] 2069 innerTail309_12 best = best := by
  rw [innerTail309_12, scan, if_neg (by decide), innerCandidates309_12_checked]
  change scan edge (search edge 3) 4 [2069] 2069 innerTail309_13
    (if 3 ≤ innerCandidates309_12.length then search edge 3 [2069, 7757] innerCandidates309_12 9826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan309_13

def innerTail309_11 : List Nat := 7283 :: innerTail309_12
def innerCandidates309_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates309_11_checked : candidates edge 4 2069 7283 innerTail309_12 best.bound = innerCandidates309_11 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan309_11 : scan edge (search edge 3) 4 [2069] 2069 innerTail309_11 best = best := by
  rw [innerTail309_11, scan, if_neg (by decide), innerCandidates309_11_checked]
  change scan edge (search edge 3) 4 [2069] 2069 innerTail309_12
    (if 3 ≤ innerCandidates309_11.length then search edge 3 [2069, 7283] innerCandidates309_11 9352 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan309_12

def innerTail309_10 : List Nat := 7253 :: innerTail309_11
def innerCandidates309_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates309_10_checked : candidates edge 4 2069 7253 innerTail309_11 best.bound = innerCandidates309_10 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan309_10 : scan edge (search edge 3) 4 [2069] 2069 innerTail309_10 best = best := by
  rw [innerTail309_10, scan, if_neg (by decide), innerCandidates309_10_checked]
  change scan edge (search edge 3) 4 [2069] 2069 innerTail309_11
    (if 3 ≤ innerCandidates309_10.length then search edge 3 [2069, 7253] innerCandidates309_10 9322 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan309_11

def innerTail309_9 : List Nat := 7121 :: innerTail309_10
def innerCandidates309_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates309_9_checked : candidates edge 4 2069 7121 innerTail309_10 best.bound = innerCandidates309_9 := by
  unfold candidates
  have row : edge 7121 = SearchData.row7121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan309_9 : scan edge (search edge 3) 4 [2069] 2069 innerTail309_9 best = best := by
  rw [innerTail309_9, scan, if_neg (by decide), innerCandidates309_9_checked]
  change scan edge (search edge 3) 4 [2069] 2069 innerTail309_10
    (if 3 ≤ innerCandidates309_9.length then search edge 3 [2069, 7121] innerCandidates309_9 9190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan309_10

def innerTail309_8 : List Nat := 6983 :: innerTail309_9
def innerCandidates309_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates309_8_checked : candidates edge 4 2069 6983 innerTail309_9 best.bound = innerCandidates309_8 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan309_8 : scan edge (search edge 3) 4 [2069] 2069 innerTail309_8 best = best := by
  rw [innerTail309_8, scan, if_neg (by decide), innerCandidates309_8_checked]
  change scan edge (search edge 3) 4 [2069] 2069 innerTail309_9
    (if 3 ≤ innerCandidates309_8.length then search edge 3 [2069, 6983] innerCandidates309_8 9052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan309_9

def innerTail309_7 : List Nat := 6959 :: innerTail309_8
def innerCandidates309_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates309_7_checked : candidates edge 4 2069 6959 innerTail309_8 best.bound = innerCandidates309_7 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan309_7 : scan edge (search edge 3) 4 [2069] 2069 innerTail309_7 best = best := by
  rw [innerTail309_7, scan, if_neg (by decide), innerCandidates309_7_checked]
  change scan edge (search edge 3) 4 [2069] 2069 innerTail309_8
    (if 3 ≤ innerCandidates309_7.length then search edge 3 [2069, 6959] innerCandidates309_7 9028 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan309_8

def innerTail309_6 : List Nat := 6827 :: innerTail309_7
def innerCandidates309_6 : List Nat := [8369]
set_option maxRecDepth 4096 in
theorem innerCandidates309_6_checked : candidates edge 4 2069 6827 innerTail309_7 best.bound = innerCandidates309_6 := by
  unfold candidates
  have row : edge 6827 = SearchData.row6827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan309_6 : scan edge (search edge 3) 4 [2069] 2069 innerTail309_6 best = best := by
  rw [innerTail309_6, scan, if_neg (by decide), innerCandidates309_6_checked]
  change scan edge (search edge 3) 4 [2069] 2069 innerTail309_7
    (if 3 ≤ innerCandidates309_6.length then search edge 3 [2069, 6827] innerCandidates309_6 8896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan309_7

def innerTail309_5 : List Nat := 5021 :: innerTail309_6
def innerCandidates309_5 : List Nat := [8219]
set_option maxRecDepth 4096 in
theorem innerCandidates309_5_checked : candidates edge 4 2069 5021 innerTail309_6 best.bound = innerCandidates309_5 := by
  unfold candidates
  have row : edge 5021 = SearchData.row5021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan309_5 : scan edge (search edge 3) 4 [2069] 2069 innerTail309_5 best = best := by
  rw [innerTail309_5, scan, if_neg (by decide), innerCandidates309_5_checked]
  change scan edge (search edge 3) 4 [2069] 2069 innerTail309_6
    (if 3 ≤ innerCandidates309_5.length then search edge 3 [2069, 5021] innerCandidates309_5 7090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan309_6

def innerTail309_4 : List Nat := 4691 :: innerTail309_5
def innerCandidates309_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates309_4_checked : candidates edge 4 2069 4691 innerTail309_5 best.bound = innerCandidates309_4 := by
  unfold candidates
  have row : edge 4691 = SearchData.row4691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan309_4 : scan edge (search edge 3) 4 [2069] 2069 innerTail309_4 best = best := by
  rw [innerTail309_4, scan, if_neg (by decide), innerCandidates309_4_checked]
  change scan edge (search edge 3) 4 [2069] 2069 innerTail309_5
    (if 3 ≤ innerCandidates309_4.length then search edge 3 [2069, 4691] innerCandidates309_4 6760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan309_5

def innerTail309_3 : List Nat := 4637 :: innerTail309_4
def innerCandidates309_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates309_3_checked : candidates edge 4 2069 4637 innerTail309_4 best.bound = innerCandidates309_3 := by
  unfold candidates
  have row : edge 4637 = SearchData.row4637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan309_3 : scan edge (search edge 3) 4 [2069] 2069 innerTail309_3 best = best := by
  rw [innerTail309_3, scan, if_neg (by decide), innerCandidates309_3_checked]
  change scan edge (search edge 3) 4 [2069] 2069 innerTail309_4
    (if 3 ≤ innerCandidates309_3.length then search edge 3 [2069, 4637] innerCandidates309_3 6706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan309_4

def innerTail309_2 : List Nat := 4397 :: innerTail309_3
def innerCandidates309_2 : List Nat := [4691]
set_option maxRecDepth 4096 in
theorem innerCandidates309_2_checked : candidates edge 4 2069 4397 innerTail309_3 best.bound = innerCandidates309_2 := by
  unfold candidates
  have row : edge 4397 = SearchData.row4397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan309_2 : scan edge (search edge 3) 4 [2069] 2069 innerTail309_2 best = best := by
  rw [innerTail309_2, scan, if_neg (by decide), innerCandidates309_2_checked]
  change scan edge (search edge 3) 4 [2069] 2069 innerTail309_3
    (if 3 ≤ innerCandidates309_2.length then search edge 3 [2069, 4397] innerCandidates309_2 6466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan309_3

def innerTail309_1 : List Nat := 2339 :: innerTail309_2
def innerCandidates309_1 : List Nat := [8369]
set_option maxRecDepth 4096 in
theorem innerCandidates309_1_checked : candidates edge 4 2069 2339 innerTail309_2 best.bound = innerCandidates309_1 := by
  unfold candidates
  have row : edge 2339 = PrimePairCache.smallRow2339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan309_1 : scan edge (search edge 3) 4 [2069] 2069 innerTail309_1 best = best := by
  rw [innerTail309_1, scan, if_neg (by decide), innerCandidates309_1_checked]
  change scan edge (search edge 3) 4 [2069] 2069 innerTail309_2
    (if 3 ≤ innerCandidates309_1.length then search edge 3 [2069, 2339] innerCandidates309_1 4408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan309_2

def innerTail309_0 : List Nat := 2297 :: innerTail309_1
def innerCandidates309_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates309_0_checked : candidates edge 4 2069 2297 innerTail309_1 best.bound = innerCandidates309_0 := by
  unfold candidates
  have row : edge 2297 = PrimePairCache.smallRow2297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan309_0 : scan edge (search edge 3) 4 [2069] 2069 innerTail309_0 best = best := by
  rw [innerTail309_0, scan, if_neg (by decide), innerCandidates309_0_checked]
  change scan edge (search edge 3) 4 [2069] 2069 innerTail309_1
    (if 3 ≤ innerCandidates309_0.length then search edge 3 [2069, 2297] innerCandidates309_0 4366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan309_1

theorem rootBranch309 : search edge 4 [2069] rootCandidates309 2069 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2069] 2069 innerTail309_0 best = best
  exact innerScan309_0

def rootCandidates308 : List Nat := [2099, 2243, 2333, 2411, 3089, 3617, 3803, 4217, 4349, 5231, 5477, 5711, 5903, 5981, 6449, 6599]
set_option maxRecDepth 4096 in
theorem rootCandidates308_checked : candidates edge 5 0 2063 tail309 best.bound = rootCandidates308 := by
  unfold candidates
  have row : edge 2063 = PrimePairCache.smallRow2063.contains := by funext q; rfl
  rw [row]
  decide

def innerTail308_16 : List Nat := []
theorem innerScan308_16 : scan edge (search edge 3) 4 [2063] 2063 innerTail308_16 best = best := rfl

def innerTail308_15 : List Nat := 6599 :: innerTail308_16
def innerCandidates308_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates308_15_checked : candidates edge 4 2063 6599 innerTail308_16 best.bound = innerCandidates308_15 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan308_15 : scan edge (search edge 3) 4 [2063] 2063 innerTail308_15 best = best := by
  rw [innerTail308_15, scan, if_neg (by decide), innerCandidates308_15_checked]
  change scan edge (search edge 3) 4 [2063] 2063 innerTail308_16
    (if 3 ≤ innerCandidates308_15.length then search edge 3 [2063, 6599] innerCandidates308_15 8662 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan308_16

def innerTail308_14 : List Nat := 6449 :: innerTail308_15
def innerCandidates308_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates308_14_checked : candidates edge 4 2063 6449 innerTail308_15 best.bound = innerCandidates308_14 := by
  unfold candidates
  have row : edge 6449 = SearchData.row6449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan308_14 : scan edge (search edge 3) 4 [2063] 2063 innerTail308_14 best = best := by
  rw [innerTail308_14, scan, if_neg (by decide), innerCandidates308_14_checked]
  change scan edge (search edge 3) 4 [2063] 2063 innerTail308_15
    (if 3 ≤ innerCandidates308_14.length then search edge 3 [2063, 6449] innerCandidates308_14 8512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan308_15

def innerTail308_13 : List Nat := 5981 :: innerTail308_14
def innerCandidates308_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates308_13_checked : candidates edge 4 2063 5981 innerTail308_14 best.bound = innerCandidates308_13 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan308_13 : scan edge (search edge 3) 4 [2063] 2063 innerTail308_13 best = best := by
  rw [innerTail308_13, scan, if_neg (by decide), innerCandidates308_13_checked]
  change scan edge (search edge 3) 4 [2063] 2063 innerTail308_14
    (if 3 ≤ innerCandidates308_13.length then search edge 3 [2063, 5981] innerCandidates308_13 8044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan308_14

def innerTail308_12 : List Nat := 5903 :: innerTail308_13
def innerCandidates308_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates308_12_checked : candidates edge 4 2063 5903 innerTail308_13 best.bound = innerCandidates308_12 := by
  unfold candidates
  have row : edge 5903 = SearchData.row5903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan308_12 : scan edge (search edge 3) 4 [2063] 2063 innerTail308_12 best = best := by
  rw [innerTail308_12, scan, if_neg (by decide), innerCandidates308_12_checked]
  change scan edge (search edge 3) 4 [2063] 2063 innerTail308_13
    (if 3 ≤ innerCandidates308_12.length then search edge 3 [2063, 5903] innerCandidates308_12 7966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan308_13

def innerTail308_11 : List Nat := 5711 :: innerTail308_12
def innerCandidates308_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates308_11_checked : candidates edge 4 2063 5711 innerTail308_12 best.bound = innerCandidates308_11 := by
  unfold candidates
  have row : edge 5711 = SearchData.row5711.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan308_11 : scan edge (search edge 3) 4 [2063] 2063 innerTail308_11 best = best := by
  rw [innerTail308_11, scan, if_neg (by decide), innerCandidates308_11_checked]
  change scan edge (search edge 3) 4 [2063] 2063 innerTail308_12
    (if 3 ≤ innerCandidates308_11.length then search edge 3 [2063, 5711] innerCandidates308_11 7774 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan308_12

def innerTail308_10 : List Nat := 5477 :: innerTail308_11
def innerCandidates308_10 : List Nat := [5711, 6599]
set_option maxRecDepth 4096 in
theorem innerCandidates308_10_checked : candidates edge 4 2063 5477 innerTail308_11 best.bound = innerCandidates308_10 := by
  unfold candidates
  have row : edge 5477 = SearchData.row5477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan308_10 : scan edge (search edge 3) 4 [2063] 2063 innerTail308_10 best = best := by
  rw [innerTail308_10, scan, if_neg (by decide), innerCandidates308_10_checked]
  change scan edge (search edge 3) 4 [2063] 2063 innerTail308_11
    (if 3 ≤ innerCandidates308_10.length then search edge 3 [2063, 5477] innerCandidates308_10 7540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan308_11

def innerTail308_9 : List Nat := 5231 :: innerTail308_10
def innerCandidates308_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates308_9_checked : candidates edge 4 2063 5231 innerTail308_10 best.bound = innerCandidates308_9 := by
  unfold candidates
  have row : edge 5231 = SearchData.row5231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan308_9 : scan edge (search edge 3) 4 [2063] 2063 innerTail308_9 best = best := by
  rw [innerTail308_9, scan, if_neg (by decide), innerCandidates308_9_checked]
  change scan edge (search edge 3) 4 [2063] 2063 innerTail308_10
    (if 3 ≤ innerCandidates308_9.length then search edge 3 [2063, 5231] innerCandidates308_9 7294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan308_10

def innerTail308_8 : List Nat := 4349 :: innerTail308_9
def innerCandidates308_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates308_8_checked : candidates edge 4 2063 4349 innerTail308_9 best.bound = innerCandidates308_8 := by
  unfold candidates
  have row : edge 4349 = SearchData.row4349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan308_8 : scan edge (search edge 3) 4 [2063] 2063 innerTail308_8 best = best := by
  rw [innerTail308_8, scan, if_neg (by decide), innerCandidates308_8_checked]
  change scan edge (search edge 3) 4 [2063] 2063 innerTail308_9
    (if 3 ≤ innerCandidates308_8.length then search edge 3 [2063, 4349] innerCandidates308_8 6412 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan308_9

def innerTail308_7 : List Nat := 4217 :: innerTail308_8
def innerCandidates308_7 : List Nat := [5981]
set_option maxRecDepth 4096 in
theorem innerCandidates308_7_checked : candidates edge 4 2063 4217 innerTail308_8 best.bound = innerCandidates308_7 := by
  unfold candidates
  have row : edge 4217 = SearchData.row4217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan308_7 : scan edge (search edge 3) 4 [2063] 2063 innerTail308_7 best = best := by
  rw [innerTail308_7, scan, if_neg (by decide), innerCandidates308_7_checked]
  change scan edge (search edge 3) 4 [2063] 2063 innerTail308_8
    (if 3 ≤ innerCandidates308_7.length then search edge 3 [2063, 4217] innerCandidates308_7 6280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan308_8

def innerTail308_6 : List Nat := 3803 :: innerTail308_7
def innerCandidates308_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates308_6_checked : candidates edge 4 2063 3803 innerTail308_7 best.bound = innerCandidates308_6 := by
  unfold candidates
  have row : edge 3803 = SearchData.row3803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan308_6 : scan edge (search edge 3) 4 [2063] 2063 innerTail308_6 best = best := by
  rw [innerTail308_6, scan, if_neg (by decide), innerCandidates308_6_checked]
  change scan edge (search edge 3) 4 [2063] 2063 innerTail308_7
    (if 3 ≤ innerCandidates308_6.length then search edge 3 [2063, 3803] innerCandidates308_6 5866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan308_7

def innerTail308_5 : List Nat := 3617 :: innerTail308_6
def innerCandidates308_5 : List Nat := [3803]
set_option maxRecDepth 4096 in
theorem innerCandidates308_5_checked : candidates edge 4 2063 3617 innerTail308_6 best.bound = innerCandidates308_5 := by
  unfold candidates
  have row : edge 3617 = SearchData.row3617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan308_5 : scan edge (search edge 3) 4 [2063] 2063 innerTail308_5 best = best := by
  rw [innerTail308_5, scan, if_neg (by decide), innerCandidates308_5_checked]
  change scan edge (search edge 3) 4 [2063] 2063 innerTail308_6
    (if 3 ≤ innerCandidates308_5.length then search edge 3 [2063, 3617] innerCandidates308_5 5680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan308_6

def innerTail308_4 : List Nat := 3089 :: innerTail308_5
def innerCandidates308_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates308_4_checked : candidates edge 4 2063 3089 innerTail308_5 best.bound = innerCandidates308_4 := by
  unfold candidates
  have row : edge 3089 = SearchData.row3089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan308_4 : scan edge (search edge 3) 4 [2063] 2063 innerTail308_4 best = best := by
  rw [innerTail308_4, scan, if_neg (by decide), innerCandidates308_4_checked]
  change scan edge (search edge 3) 4 [2063] 2063 innerTail308_5
    (if 3 ≤ innerCandidates308_4.length then search edge 3 [2063, 3089] innerCandidates308_4 5152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan308_5

def innerTail308_3 : List Nat := 2411 :: innerTail308_4
def innerCandidates308_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates308_3_checked : candidates edge 4 2063 2411 innerTail308_4 best.bound = innerCandidates308_3 := by
  unfold candidates
  have row : edge 2411 = PrimePairCache.smallRow2411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan308_3 : scan edge (search edge 3) 4 [2063] 2063 innerTail308_3 best = best := by
  rw [innerTail308_3, scan, if_neg (by decide), innerCandidates308_3_checked]
  change scan edge (search edge 3) 4 [2063] 2063 innerTail308_4
    (if 3 ≤ innerCandidates308_3.length then search edge 3 [2063, 2411] innerCandidates308_3 4474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan308_4

def innerTail308_2 : List Nat := 2333 :: innerTail308_3
def innerCandidates308_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates308_2_checked : candidates edge 4 2063 2333 innerTail308_3 best.bound = innerCandidates308_2 := by
  unfold candidates
  have row : edge 2333 = PrimePairCache.smallRow2333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan308_2 : scan edge (search edge 3) 4 [2063] 2063 innerTail308_2 best = best := by
  rw [innerTail308_2, scan, if_neg (by decide), innerCandidates308_2_checked]
  change scan edge (search edge 3) 4 [2063] 2063 innerTail308_3
    (if 3 ≤ innerCandidates308_2.length then search edge 3 [2063, 2333] innerCandidates308_2 4396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan308_3

def innerTail308_1 : List Nat := 2243 :: innerTail308_2
def innerCandidates308_1 : List Nat := [2411, 5477, 6599]
set_option maxRecDepth 4096 in
theorem innerCandidates308_1_checked : candidates edge 4 2063 2243 innerTail308_2 best.bound = innerCandidates308_1 := by
  unfold candidates
  have row : edge 2243 = PrimePairCache.smallRow2243.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild308_1 : search edge 3 [2063, 2243] innerCandidates308_1 4306 best = best := by decide

theorem innerScan308_1 : scan edge (search edge 3) 4 [2063] 2063 innerTail308_1 best = best := by
  rw [innerTail308_1, scan, if_neg (by decide), innerCandidates308_1_checked]
  change scan edge (search edge 3) 4 [2063] 2063 innerTail308_2
    (if 3 ≤ innerCandidates308_1.length then search edge 3 [2063, 2243] innerCandidates308_1 4306 best else best) = best
  rw [if_pos (by decide), innerChild308_1]
  exact innerScan308_2

def innerTail308_0 : List Nat := 2099 :: innerTail308_1
def innerCandidates308_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates308_0_checked : candidates edge 4 2063 2099 innerTail308_1 best.bound = innerCandidates308_0 := by
  unfold candidates
  have row : edge 2099 = PrimePairCache.smallRow2099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan308_0 : scan edge (search edge 3) 4 [2063] 2063 innerTail308_0 best = best := by
  rw [innerTail308_0, scan, if_neg (by decide), innerCandidates308_0_checked]
  change scan edge (search edge 3) 4 [2063] 2063 innerTail308_1
    (if 3 ≤ innerCandidates308_0.length then search edge 3 [2063, 2099] innerCandidates308_0 4162 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan308_1

theorem rootBranch308 : search edge 4 [2063] rootCandidates308 2063 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2063] 2063 innerTail308_0 best = best
  exact innerScan308_0

def rootCandidates307 : List Nat := [2797, 3049, 3361, 4729, 4783, 5653, 6343, 6361, 6421, 6619, 8317]
set_option maxRecDepth 4096 in
theorem rootCandidates307_checked : candidates edge 5 0 2053 tail308 best.bound = rootCandidates307 := by
  unfold candidates
  have row : edge 2053 = PrimePairCache.smallRow2053.contains := by funext q; rfl
  rw [row]
  decide

def innerTail307_11 : List Nat := []
theorem innerScan307_11 : scan edge (search edge 3) 4 [2053] 2053 innerTail307_11 best = best := rfl

def innerTail307_10 : List Nat := 8317 :: innerTail307_11
def innerCandidates307_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates307_10_checked : candidates edge 4 2053 8317 innerTail307_11 best.bound = innerCandidates307_10 := by
  unfold candidates
  have row : edge 8317 = SearchData.row8317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan307_10 : scan edge (search edge 3) 4 [2053] 2053 innerTail307_10 best = best := by
  rw [innerTail307_10, scan, if_neg (by decide), innerCandidates307_10_checked]
  change scan edge (search edge 3) 4 [2053] 2053 innerTail307_11
    (if 3 ≤ innerCandidates307_10.length then search edge 3 [2053, 8317] innerCandidates307_10 10370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan307_11

def innerTail307_9 : List Nat := 6619 :: innerTail307_10
def innerCandidates307_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates307_9_checked : candidates edge 4 2053 6619 innerTail307_10 best.bound = innerCandidates307_9 := by
  unfold candidates
  have row : edge 6619 = SearchData.row6619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan307_9 : scan edge (search edge 3) 4 [2053] 2053 innerTail307_9 best = best := by
  rw [innerTail307_9, scan, if_neg (by decide), innerCandidates307_9_checked]
  change scan edge (search edge 3) 4 [2053] 2053 innerTail307_10
    (if 3 ≤ innerCandidates307_9.length then search edge 3 [2053, 6619] innerCandidates307_9 8672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan307_10

def innerTail307_8 : List Nat := 6421 :: innerTail307_9
def innerCandidates307_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates307_8_checked : candidates edge 4 2053 6421 innerTail307_9 best.bound = innerCandidates307_8 := by
  unfold candidates
  have row : edge 6421 = SearchData.row6421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan307_8 : scan edge (search edge 3) 4 [2053] 2053 innerTail307_8 best = best := by
  rw [innerTail307_8, scan, if_neg (by decide), innerCandidates307_8_checked]
  change scan edge (search edge 3) 4 [2053] 2053 innerTail307_9
    (if 3 ≤ innerCandidates307_8.length then search edge 3 [2053, 6421] innerCandidates307_8 8474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan307_9

def innerTail307_7 : List Nat := 6361 :: innerTail307_8
def innerCandidates307_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates307_7_checked : candidates edge 4 2053 6361 innerTail307_8 best.bound = innerCandidates307_7 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan307_7 : scan edge (search edge 3) 4 [2053] 2053 innerTail307_7 best = best := by
  rw [innerTail307_7, scan, if_neg (by decide), innerCandidates307_7_checked]
  change scan edge (search edge 3) 4 [2053] 2053 innerTail307_8
    (if 3 ≤ innerCandidates307_7.length then search edge 3 [2053, 6361] innerCandidates307_7 8414 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan307_8

def innerTail307_6 : List Nat := 6343 :: innerTail307_7
def innerCandidates307_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates307_6_checked : candidates edge 4 2053 6343 innerTail307_7 best.bound = innerCandidates307_6 := by
  unfold candidates
  have row : edge 6343 = SearchData.row6343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan307_6 : scan edge (search edge 3) 4 [2053] 2053 innerTail307_6 best = best := by
  rw [innerTail307_6, scan, if_neg (by decide), innerCandidates307_6_checked]
  change scan edge (search edge 3) 4 [2053] 2053 innerTail307_7
    (if 3 ≤ innerCandidates307_6.length then search edge 3 [2053, 6343] innerCandidates307_6 8396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan307_7

def innerTail307_5 : List Nat := 5653 :: innerTail307_6
def innerCandidates307_5 : List Nat := [6619]
set_option maxRecDepth 4096 in
theorem innerCandidates307_5_checked : candidates edge 4 2053 5653 innerTail307_6 best.bound = innerCandidates307_5 := by
  unfold candidates
  have row : edge 5653 = SearchData.row5653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan307_5 : scan edge (search edge 3) 4 [2053] 2053 innerTail307_5 best = best := by
  rw [innerTail307_5, scan, if_neg (by decide), innerCandidates307_5_checked]
  change scan edge (search edge 3) 4 [2053] 2053 innerTail307_6
    (if 3 ≤ innerCandidates307_5.length then search edge 3 [2053, 5653] innerCandidates307_5 7706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan307_6

def innerTail307_4 : List Nat := 4783 :: innerTail307_5
def innerCandidates307_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates307_4_checked : candidates edge 4 2053 4783 innerTail307_5 best.bound = innerCandidates307_4 := by
  unfold candidates
  have row : edge 4783 = SearchData.row4783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan307_4 : scan edge (search edge 3) 4 [2053] 2053 innerTail307_4 best = best := by
  rw [innerTail307_4, scan, if_neg (by decide), innerCandidates307_4_checked]
  change scan edge (search edge 3) 4 [2053] 2053 innerTail307_5
    (if 3 ≤ innerCandidates307_4.length then search edge 3 [2053, 4783] innerCandidates307_4 6836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan307_5

def innerTail307_3 : List Nat := 4729 :: innerTail307_4
def innerCandidates307_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates307_3_checked : candidates edge 4 2053 4729 innerTail307_4 best.bound = innerCandidates307_3 := by
  unfold candidates
  have row : edge 4729 = SearchData.row4729.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan307_3 : scan edge (search edge 3) 4 [2053] 2053 innerTail307_3 best = best := by
  rw [innerTail307_3, scan, if_neg (by decide), innerCandidates307_3_checked]
  change scan edge (search edge 3) 4 [2053] 2053 innerTail307_4
    (if 3 ≤ innerCandidates307_3.length then search edge 3 [2053, 4729] innerCandidates307_3 6782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan307_4

def innerTail307_2 : List Nat := 3361 :: innerTail307_3
def innerCandidates307_2 : List Nat := [6343]
set_option maxRecDepth 4096 in
theorem innerCandidates307_2_checked : candidates edge 4 2053 3361 innerTail307_3 best.bound = innerCandidates307_2 := by
  unfold candidates
  have row : edge 3361 = SearchData.row3361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan307_2 : scan edge (search edge 3) 4 [2053] 2053 innerTail307_2 best = best := by
  rw [innerTail307_2, scan, if_neg (by decide), innerCandidates307_2_checked]
  change scan edge (search edge 3) 4 [2053] 2053 innerTail307_3
    (if 3 ≤ innerCandidates307_2.length then search edge 3 [2053, 3361] innerCandidates307_2 5414 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan307_3

def innerTail307_1 : List Nat := 3049 :: innerTail307_2
def innerCandidates307_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates307_1_checked : candidates edge 4 2053 3049 innerTail307_2 best.bound = innerCandidates307_1 := by
  unfold candidates
  have row : edge 3049 = SearchData.row3049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan307_1 : scan edge (search edge 3) 4 [2053] 2053 innerTail307_1 best = best := by
  rw [innerTail307_1, scan, if_neg (by decide), innerCandidates307_1_checked]
  change scan edge (search edge 3) 4 [2053] 2053 innerTail307_2
    (if 3 ≤ innerCandidates307_1.length then search edge 3 [2053, 3049] innerCandidates307_1 5102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan307_2

def innerTail307_0 : List Nat := 2797 :: innerTail307_1
def innerCandidates307_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates307_0_checked : candidates edge 4 2053 2797 innerTail307_1 best.bound = innerCandidates307_0 := by
  unfold candidates
  have row : edge 2797 = SearchData.row2797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan307_0 : scan edge (search edge 3) 4 [2053] 2053 innerTail307_0 best = best := by
  rw [innerTail307_0, scan, if_neg (by decide), innerCandidates307_0_checked]
  change scan edge (search edge 3) 4 [2053] 2053 innerTail307_1
    (if 3 ≤ innerCandidates307_0.length then search edge 3 [2053, 2797] innerCandidates307_0 4850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan307_1

theorem rootBranch307 : search edge 4 [2053] rootCandidates307 2053 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2053] 2053 innerTail307_0 best = best
  exact innerScan307_0

end ProjectEulerSolutions.PrimePairUpper.Search
