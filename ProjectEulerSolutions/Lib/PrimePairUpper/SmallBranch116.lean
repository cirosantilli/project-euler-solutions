import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates334 : List Nat := [2281, 3067, 3373, 3613, 3739, 4057, 4621, 4663, 4909, 5107, 5347, 5821, 5923, 7027, 7237, 7489, 7561, 7879]
set_option maxRecDepth 4096 in
theorem rootCandidates334_checked : candidates edge 5 0 2269 tail335 best.bound = rootCandidates334 := by
  unfold candidates
  have row : edge 2269 = PrimePairCache.smallRow2269.contains := by funext q; rfl
  rw [row]
  decide

def innerTail334_18 : List Nat := []
theorem innerScan334_18 : scan edge (search edge 3) 4 [2269] 2269 innerTail334_18 best = best := rfl

def innerTail334_17 : List Nat := 7879 :: innerTail334_18
def innerCandidates334_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates334_17_checked : candidates edge 4 2269 7879 innerTail334_18 best.bound = innerCandidates334_17 := by
  unfold candidates
  have row : edge 7879 = SearchData.row7879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan334_17 : scan edge (search edge 3) 4 [2269] 2269 innerTail334_17 best = best := by
  rw [innerTail334_17, scan, if_neg (by decide), innerCandidates334_17_checked]
  change scan edge (search edge 3) 4 [2269] 2269 innerTail334_18
    (if 3 ≤ innerCandidates334_17.length then search edge 3 [2269, 7879] innerCandidates334_17 10148 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan334_18

def innerTail334_16 : List Nat := 7561 :: innerTail334_17
def innerCandidates334_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates334_16_checked : candidates edge 4 2269 7561 innerTail334_17 best.bound = innerCandidates334_16 := by
  unfold candidates
  have row : edge 7561 = SearchData.row7561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan334_16 : scan edge (search edge 3) 4 [2269] 2269 innerTail334_16 best = best := by
  rw [innerTail334_16, scan, if_neg (by decide), innerCandidates334_16_checked]
  change scan edge (search edge 3) 4 [2269] 2269 innerTail334_17
    (if 3 ≤ innerCandidates334_16.length then search edge 3 [2269, 7561] innerCandidates334_16 9830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan334_17

def innerTail334_15 : List Nat := 7489 :: innerTail334_16
def innerCandidates334_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates334_15_checked : candidates edge 4 2269 7489 innerTail334_16 best.bound = innerCandidates334_15 := by
  unfold candidates
  have row : edge 7489 = SearchData.row7489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan334_15 : scan edge (search edge 3) 4 [2269] 2269 innerTail334_15 best = best := by
  rw [innerTail334_15, scan, if_neg (by decide), innerCandidates334_15_checked]
  change scan edge (search edge 3) 4 [2269] 2269 innerTail334_16
    (if 3 ≤ innerCandidates334_15.length then search edge 3 [2269, 7489] innerCandidates334_15 9758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan334_16

def innerTail334_14 : List Nat := 7237 :: innerTail334_15
def innerCandidates334_14 : List Nat := [7561]
set_option maxRecDepth 4096 in
theorem innerCandidates334_14_checked : candidates edge 4 2269 7237 innerTail334_15 best.bound = innerCandidates334_14 := by
  unfold candidates
  have row : edge 7237 = SearchData.row7237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan334_14 : scan edge (search edge 3) 4 [2269] 2269 innerTail334_14 best = best := by
  rw [innerTail334_14, scan, if_neg (by decide), innerCandidates334_14_checked]
  change scan edge (search edge 3) 4 [2269] 2269 innerTail334_15
    (if 3 ≤ innerCandidates334_14.length then search edge 3 [2269, 7237] innerCandidates334_14 9506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan334_15

def innerTail334_13 : List Nat := 7027 :: innerTail334_14
def innerCandidates334_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates334_13_checked : candidates edge 4 2269 7027 innerTail334_14 best.bound = innerCandidates334_13 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan334_13 : scan edge (search edge 3) 4 [2269] 2269 innerTail334_13 best = best := by
  rw [innerTail334_13, scan, if_neg (by decide), innerCandidates334_13_checked]
  change scan edge (search edge 3) 4 [2269] 2269 innerTail334_14
    (if 3 ≤ innerCandidates334_13.length then search edge 3 [2269, 7027] innerCandidates334_13 9296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan334_14

def innerTail334_12 : List Nat := 5923 :: innerTail334_13
def innerCandidates334_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates334_12_checked : candidates edge 4 2269 5923 innerTail334_13 best.bound = innerCandidates334_12 := by
  unfold candidates
  have row : edge 5923 = SearchData.row5923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan334_12 : scan edge (search edge 3) 4 [2269] 2269 innerTail334_12 best = best := by
  rw [innerTail334_12, scan, if_neg (by decide), innerCandidates334_12_checked]
  change scan edge (search edge 3) 4 [2269] 2269 innerTail334_13
    (if 3 ≤ innerCandidates334_12.length then search edge 3 [2269, 5923] innerCandidates334_12 8192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan334_13

def innerTail334_11 : List Nat := 5821 :: innerTail334_12
def innerCandidates334_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates334_11_checked : candidates edge 4 2269 5821 innerTail334_12 best.bound = innerCandidates334_11 := by
  unfold candidates
  have row : edge 5821 = SearchData.row5821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan334_11 : scan edge (search edge 3) 4 [2269] 2269 innerTail334_11 best = best := by
  rw [innerTail334_11, scan, if_neg (by decide), innerCandidates334_11_checked]
  change scan edge (search edge 3) 4 [2269] 2269 innerTail334_12
    (if 3 ≤ innerCandidates334_11.length then search edge 3 [2269, 5821] innerCandidates334_11 8090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan334_12

def innerTail334_10 : List Nat := 5347 :: innerTail334_11
def innerCandidates334_10 : List Nat := [7027]
set_option maxRecDepth 4096 in
theorem innerCandidates334_10_checked : candidates edge 4 2269 5347 innerTail334_11 best.bound = innerCandidates334_10 := by
  unfold candidates
  have row : edge 5347 = SearchData.row5347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan334_10 : scan edge (search edge 3) 4 [2269] 2269 innerTail334_10 best = best := by
  rw [innerTail334_10, scan, if_neg (by decide), innerCandidates334_10_checked]
  change scan edge (search edge 3) 4 [2269] 2269 innerTail334_11
    (if 3 ≤ innerCandidates334_10.length then search edge 3 [2269, 5347] innerCandidates334_10 7616 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan334_11

def innerTail334_9 : List Nat := 5107 :: innerTail334_10
def innerCandidates334_9 : List Nat := [7879]
set_option maxRecDepth 4096 in
theorem innerCandidates334_9_checked : candidates edge 4 2269 5107 innerTail334_10 best.bound = innerCandidates334_9 := by
  unfold candidates
  have row : edge 5107 = SearchData.row5107.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan334_9 : scan edge (search edge 3) 4 [2269] 2269 innerTail334_9 best = best := by
  rw [innerTail334_9, scan, if_neg (by decide), innerCandidates334_9_checked]
  change scan edge (search edge 3) 4 [2269] 2269 innerTail334_10
    (if 3 ≤ innerCandidates334_9.length then search edge 3 [2269, 5107] innerCandidates334_9 7376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan334_10

def innerTail334_8 : List Nat := 4909 :: innerTail334_9
def innerCandidates334_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates334_8_checked : candidates edge 4 2269 4909 innerTail334_9 best.bound = innerCandidates334_8 := by
  unfold candidates
  have row : edge 4909 = SearchData.row4909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan334_8 : scan edge (search edge 3) 4 [2269] 2269 innerTail334_8 best = best := by
  rw [innerTail334_8, scan, if_neg (by decide), innerCandidates334_8_checked]
  change scan edge (search edge 3) 4 [2269] 2269 innerTail334_9
    (if 3 ≤ innerCandidates334_8.length then search edge 3 [2269, 4909] innerCandidates334_8 7178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan334_9

def innerTail334_7 : List Nat := 4663 :: innerTail334_8
def innerCandidates334_7 : List Nat := [7879]
set_option maxRecDepth 4096 in
theorem innerCandidates334_7_checked : candidates edge 4 2269 4663 innerTail334_8 best.bound = innerCandidates334_7 := by
  unfold candidates
  have row : edge 4663 = SearchData.row4663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan334_7 : scan edge (search edge 3) 4 [2269] 2269 innerTail334_7 best = best := by
  rw [innerTail334_7, scan, if_neg (by decide), innerCandidates334_7_checked]
  change scan edge (search edge 3) 4 [2269] 2269 innerTail334_8
    (if 3 ≤ innerCandidates334_7.length then search edge 3 [2269, 4663] innerCandidates334_7 6932 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan334_8

def innerTail334_6 : List Nat := 4621 :: innerTail334_7
def innerCandidates334_6 : List Nat := [5347, 7489]
set_option maxRecDepth 4096 in
theorem innerCandidates334_6_checked : candidates edge 4 2269 4621 innerTail334_7 best.bound = innerCandidates334_6 := by
  unfold candidates
  have row : edge 4621 = SearchData.row4621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan334_6 : scan edge (search edge 3) 4 [2269] 2269 innerTail334_6 best = best := by
  rw [innerTail334_6, scan, if_neg (by decide), innerCandidates334_6_checked]
  change scan edge (search edge 3) 4 [2269] 2269 innerTail334_7
    (if 3 ≤ innerCandidates334_6.length then search edge 3 [2269, 4621] innerCandidates334_6 6890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan334_7

def innerTail334_5 : List Nat := 4057 :: innerTail334_6
def innerCandidates334_5 : List Nat := [4663, 7879]
set_option maxRecDepth 4096 in
theorem innerCandidates334_5_checked : candidates edge 4 2269 4057 innerTail334_6 best.bound = innerCandidates334_5 := by
  unfold candidates
  have row : edge 4057 = SearchData.row4057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan334_5 : scan edge (search edge 3) 4 [2269] 2269 innerTail334_5 best = best := by
  rw [innerTail334_5, scan, if_neg (by decide), innerCandidates334_5_checked]
  change scan edge (search edge 3) 4 [2269] 2269 innerTail334_6
    (if 3 ≤ innerCandidates334_5.length then search edge 3 [2269, 4057] innerCandidates334_5 6326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan334_6

def innerTail334_4 : List Nat := 3739 :: innerTail334_5
def innerCandidates334_4 : List Nat := [5107]
set_option maxRecDepth 4096 in
theorem innerCandidates334_4_checked : candidates edge 4 2269 3739 innerTail334_5 best.bound = innerCandidates334_4 := by
  unfold candidates
  have row : edge 3739 = SearchData.row3739.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan334_4 : scan edge (search edge 3) 4 [2269] 2269 innerTail334_4 best = best := by
  rw [innerTail334_4, scan, if_neg (by decide), innerCandidates334_4_checked]
  change scan edge (search edge 3) 4 [2269] 2269 innerTail334_5
    (if 3 ≤ innerCandidates334_4.length then search edge 3 [2269, 3739] innerCandidates334_4 6008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan334_5

def innerTail334_3 : List Nat := 3613 :: innerTail334_4
def innerCandidates334_3 : List Nat := [5821]
set_option maxRecDepth 4096 in
theorem innerCandidates334_3_checked : candidates edge 4 2269 3613 innerTail334_4 best.bound = innerCandidates334_3 := by
  unfold candidates
  have row : edge 3613 = SearchData.row3613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan334_3 : scan edge (search edge 3) 4 [2269] 2269 innerTail334_3 best = best := by
  rw [innerTail334_3, scan, if_neg (by decide), innerCandidates334_3_checked]
  change scan edge (search edge 3) 4 [2269] 2269 innerTail334_4
    (if 3 ≤ innerCandidates334_3.length then search edge 3 [2269, 3613] innerCandidates334_3 5882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan334_4

def innerTail334_2 : List Nat := 3373 :: innerTail334_3
def innerCandidates334_2 : List Nat := [4621]
set_option maxRecDepth 4096 in
theorem innerCandidates334_2_checked : candidates edge 4 2269 3373 innerTail334_3 best.bound = innerCandidates334_2 := by
  unfold candidates
  have row : edge 3373 = SearchData.row3373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan334_2 : scan edge (search edge 3) 4 [2269] 2269 innerTail334_2 best = best := by
  rw [innerTail334_2, scan, if_neg (by decide), innerCandidates334_2_checked]
  change scan edge (search edge 3) 4 [2269] 2269 innerTail334_3
    (if 3 ≤ innerCandidates334_2.length then search edge 3 [2269, 3373] innerCandidates334_2 5642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan334_3

def innerTail334_1 : List Nat := 3067 :: innerTail334_2
def innerCandidates334_1 : List Nat := [3373, 4909, 7489]
set_option maxRecDepth 4096 in
theorem innerCandidates334_1_checked : candidates edge 4 2269 3067 innerTail334_2 best.bound = innerCandidates334_1 := by
  unfold candidates
  have row : edge 3067 = SearchData.row3067.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild334_1 : search edge 3 [2269, 3067] innerCandidates334_1 5336 best = best := by decide

theorem innerScan334_1 : scan edge (search edge 3) 4 [2269] 2269 innerTail334_1 best = best := by
  rw [innerTail334_1, scan, if_neg (by decide), innerCandidates334_1_checked]
  change scan edge (search edge 3) 4 [2269] 2269 innerTail334_2
    (if 3 ≤ innerCandidates334_1.length then search edge 3 [2269, 3067] innerCandidates334_1 5336 best else best) = best
  rw [if_pos (by decide), innerChild334_1]
  exact innerScan334_2

def innerTail334_0 : List Nat := 2281 :: innerTail334_1
def innerCandidates334_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates334_0_checked : candidates edge 4 2269 2281 innerTail334_1 best.bound = innerCandidates334_0 := by
  unfold candidates
  have row : edge 2281 = PrimePairCache.smallRow2281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan334_0 : scan edge (search edge 3) 4 [2269] 2269 innerTail334_0 best = best := by
  rw [innerTail334_0, scan, if_neg (by decide), innerCandidates334_0_checked]
  change scan edge (search edge 3) 4 [2269] 2269 innerTail334_1
    (if 3 ≤ innerCandidates334_0.length then search edge 3 [2269, 2281] innerCandidates334_0 4550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan334_1

theorem rootBranch334 : search edge 4 [2269] rootCandidates334 2269 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2269] 2269 innerTail334_0 best = best
  exact innerScan334_0

def rootCandidates333 : List Nat := [2729, 2939, 3323, 3527, 4637, 4721, 5669, 5843, 5867, 5879, 5897, 6053, 6389, 6791, 6833, 6917, 7673, 8147, 8219]
set_option maxRecDepth 4096 in
theorem rootCandidates333_checked : candidates edge 5 0 2267 tail334 best.bound = rootCandidates333 := by
  unfold candidates
  have row : edge 2267 = PrimePairCache.smallRow2267.contains := by funext q; rfl
  rw [row]
  decide

def innerTail333_19 : List Nat := []
theorem innerScan333_19 : scan edge (search edge 3) 4 [2267] 2267 innerTail333_19 best = best := rfl

def innerTail333_18 : List Nat := 8219 :: innerTail333_19
def innerCandidates333_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates333_18_checked : candidates edge 4 2267 8219 innerTail333_19 best.bound = innerCandidates333_18 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan333_18 : scan edge (search edge 3) 4 [2267] 2267 innerTail333_18 best = best := by
  rw [innerTail333_18, scan, if_neg (by decide), innerCandidates333_18_checked]
  change scan edge (search edge 3) 4 [2267] 2267 innerTail333_19
    (if 3 ≤ innerCandidates333_18.length then search edge 3 [2267, 8219] innerCandidates333_18 10486 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan333_19

def innerTail333_17 : List Nat := 8147 :: innerTail333_18
def innerCandidates333_17 : List Nat := [8219]
set_option maxRecDepth 4096 in
theorem innerCandidates333_17_checked : candidates edge 4 2267 8147 innerTail333_18 best.bound = innerCandidates333_17 := by
  unfold candidates
  have row : edge 8147 = SearchData.row8147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan333_17 : scan edge (search edge 3) 4 [2267] 2267 innerTail333_17 best = best := by
  rw [innerTail333_17, scan, if_neg (by decide), innerCandidates333_17_checked]
  change scan edge (search edge 3) 4 [2267] 2267 innerTail333_18
    (if 3 ≤ innerCandidates333_17.length then search edge 3 [2267, 8147] innerCandidates333_17 10414 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan333_18

def innerTail333_16 : List Nat := 7673 :: innerTail333_17
def innerCandidates333_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates333_16_checked : candidates edge 4 2267 7673 innerTail333_17 best.bound = innerCandidates333_16 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan333_16 : scan edge (search edge 3) 4 [2267] 2267 innerTail333_16 best = best := by
  rw [innerTail333_16, scan, if_neg (by decide), innerCandidates333_16_checked]
  change scan edge (search edge 3) 4 [2267] 2267 innerTail333_17
    (if 3 ≤ innerCandidates333_16.length then search edge 3 [2267, 7673] innerCandidates333_16 9940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan333_17

def innerTail333_15 : List Nat := 6917 :: innerTail333_16
def innerCandidates333_15 : List Nat := [8147]
set_option maxRecDepth 4096 in
theorem innerCandidates333_15_checked : candidates edge 4 2267 6917 innerTail333_16 best.bound = innerCandidates333_15 := by
  unfold candidates
  have row : edge 6917 = SearchData.row6917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan333_15 : scan edge (search edge 3) 4 [2267] 2267 innerTail333_15 best = best := by
  rw [innerTail333_15, scan, if_neg (by decide), innerCandidates333_15_checked]
  change scan edge (search edge 3) 4 [2267] 2267 innerTail333_16
    (if 3 ≤ innerCandidates333_15.length then search edge 3 [2267, 6917] innerCandidates333_15 9184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan333_16

def innerTail333_14 : List Nat := 6833 :: innerTail333_15
def innerCandidates333_14 : List Nat := [7673]
set_option maxRecDepth 4096 in
theorem innerCandidates333_14_checked : candidates edge 4 2267 6833 innerTail333_15 best.bound = innerCandidates333_14 := by
  unfold candidates
  have row : edge 6833 = SearchData.row6833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan333_14 : scan edge (search edge 3) 4 [2267] 2267 innerTail333_14 best = best := by
  rw [innerTail333_14, scan, if_neg (by decide), innerCandidates333_14_checked]
  change scan edge (search edge 3) 4 [2267] 2267 innerTail333_15
    (if 3 ≤ innerCandidates333_14.length then search edge 3 [2267, 6833] innerCandidates333_14 9100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan333_15

def innerTail333_13 : List Nat := 6791 :: innerTail333_14
def innerCandidates333_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates333_13_checked : candidates edge 4 2267 6791 innerTail333_14 best.bound = innerCandidates333_13 := by
  unfold candidates
  have row : edge 6791 = SearchData.row6791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan333_13 : scan edge (search edge 3) 4 [2267] 2267 innerTail333_13 best = best := by
  rw [innerTail333_13, scan, if_neg (by decide), innerCandidates333_13_checked]
  change scan edge (search edge 3) 4 [2267] 2267 innerTail333_14
    (if 3 ≤ innerCandidates333_13.length then search edge 3 [2267, 6791] innerCandidates333_13 9058 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan333_14

def innerTail333_12 : List Nat := 6389 :: innerTail333_13
def innerCandidates333_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates333_12_checked : candidates edge 4 2267 6389 innerTail333_13 best.bound = innerCandidates333_12 := by
  unfold candidates
  have row : edge 6389 = SearchData.row6389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan333_12 : scan edge (search edge 3) 4 [2267] 2267 innerTail333_12 best = best := by
  rw [innerTail333_12, scan, if_neg (by decide), innerCandidates333_12_checked]
  change scan edge (search edge 3) 4 [2267] 2267 innerTail333_13
    (if 3 ≤ innerCandidates333_12.length then search edge 3 [2267, 6389] innerCandidates333_12 8656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan333_13

def innerTail333_11 : List Nat := 6053 :: innerTail333_12
def innerCandidates333_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates333_11_checked : candidates edge 4 2267 6053 innerTail333_12 best.bound = innerCandidates333_11 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan333_11 : scan edge (search edge 3) 4 [2267] 2267 innerTail333_11 best = best := by
  rw [innerTail333_11, scan, if_neg (by decide), innerCandidates333_11_checked]
  change scan edge (search edge 3) 4 [2267] 2267 innerTail333_12
    (if 3 ≤ innerCandidates333_11.length then search edge 3 [2267, 6053] innerCandidates333_11 8320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan333_12

def innerTail333_10 : List Nat := 5897 :: innerTail333_11
def innerCandidates333_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates333_10_checked : candidates edge 4 2267 5897 innerTail333_11 best.bound = innerCandidates333_10 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan333_10 : scan edge (search edge 3) 4 [2267] 2267 innerTail333_10 best = best := by
  rw [innerTail333_10, scan, if_neg (by decide), innerCandidates333_10_checked]
  change scan edge (search edge 3) 4 [2267] 2267 innerTail333_11
    (if 3 ≤ innerCandidates333_10.length then search edge 3 [2267, 5897] innerCandidates333_10 8164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan333_11

def innerTail333_9 : List Nat := 5879 :: innerTail333_10
def innerCandidates333_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates333_9_checked : candidates edge 4 2267 5879 innerTail333_10 best.bound = innerCandidates333_9 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan333_9 : scan edge (search edge 3) 4 [2267] 2267 innerTail333_9 best = best := by
  rw [innerTail333_9, scan, if_neg (by decide), innerCandidates333_9_checked]
  change scan edge (search edge 3) 4 [2267] 2267 innerTail333_10
    (if 3 ≤ innerCandidates333_9.length then search edge 3 [2267, 5879] innerCandidates333_9 8146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan333_10

def innerTail333_8 : List Nat := 5867 :: innerTail333_9
def innerCandidates333_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates333_8_checked : candidates edge 4 2267 5867 innerTail333_9 best.bound = innerCandidates333_8 := by
  unfold candidates
  have row : edge 5867 = SearchData.row5867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan333_8 : scan edge (search edge 3) 4 [2267] 2267 innerTail333_8 best = best := by
  rw [innerTail333_8, scan, if_neg (by decide), innerCandidates333_8_checked]
  change scan edge (search edge 3) 4 [2267] 2267 innerTail333_9
    (if 3 ≤ innerCandidates333_8.length then search edge 3 [2267, 5867] innerCandidates333_8 8134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan333_9

def innerTail333_7 : List Nat := 5843 :: innerTail333_8
def innerCandidates333_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates333_7_checked : candidates edge 4 2267 5843 innerTail333_8 best.bound = innerCandidates333_7 := by
  unfold candidates
  have row : edge 5843 = SearchData.row5843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan333_7 : scan edge (search edge 3) 4 [2267] 2267 innerTail333_7 best = best := by
  rw [innerTail333_7, scan, if_neg (by decide), innerCandidates333_7_checked]
  change scan edge (search edge 3) 4 [2267] 2267 innerTail333_8
    (if 3 ≤ innerCandidates333_7.length then search edge 3 [2267, 5843] innerCandidates333_7 8110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan333_8

def innerTail333_6 : List Nat := 5669 :: innerTail333_7
def innerCandidates333_6 : List Nat := [5879]
set_option maxRecDepth 4096 in
theorem innerCandidates333_6_checked : candidates edge 4 2267 5669 innerTail333_7 best.bound = innerCandidates333_6 := by
  unfold candidates
  have row : edge 5669 = SearchData.row5669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan333_6 : scan edge (search edge 3) 4 [2267] 2267 innerTail333_6 best = best := by
  rw [innerTail333_6, scan, if_neg (by decide), innerCandidates333_6_checked]
  change scan edge (search edge 3) 4 [2267] 2267 innerTail333_7
    (if 3 ≤ innerCandidates333_6.length then search edge 3 [2267, 5669] innerCandidates333_6 7936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan333_7

def innerTail333_5 : List Nat := 4721 :: innerTail333_6
def innerCandidates333_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates333_5_checked : candidates edge 4 2267 4721 innerTail333_6 best.bound = innerCandidates333_5 := by
  unfold candidates
  have row : edge 4721 = SearchData.row4721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan333_5 : scan edge (search edge 3) 4 [2267] 2267 innerTail333_5 best = best := by
  rw [innerTail333_5, scan, if_neg (by decide), innerCandidates333_5_checked]
  change scan edge (search edge 3) 4 [2267] 2267 innerTail333_6
    (if 3 ≤ innerCandidates333_5.length then search edge 3 [2267, 4721] innerCandidates333_5 6988 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan333_6

def innerTail333_4 : List Nat := 4637 :: innerTail333_5
def innerCandidates333_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates333_4_checked : candidates edge 4 2267 4637 innerTail333_5 best.bound = innerCandidates333_4 := by
  unfold candidates
  have row : edge 4637 = SearchData.row4637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan333_4 : scan edge (search edge 3) 4 [2267] 2267 innerTail333_4 best = best := by
  rw [innerTail333_4, scan, if_neg (by decide), innerCandidates333_4_checked]
  change scan edge (search edge 3) 4 [2267] 2267 innerTail333_5
    (if 3 ≤ innerCandidates333_4.length then search edge 3 [2267, 4637] innerCandidates333_4 6904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan333_5

def innerTail333_3 : List Nat := 3527 :: innerTail333_4
def innerCandidates333_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates333_3_checked : candidates edge 4 2267 3527 innerTail333_4 best.bound = innerCandidates333_3 := by
  unfold candidates
  have row : edge 3527 = SearchData.row3527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan333_3 : scan edge (search edge 3) 4 [2267] 2267 innerTail333_3 best = best := by
  rw [innerTail333_3, scan, if_neg (by decide), innerCandidates333_3_checked]
  change scan edge (search edge 3) 4 [2267] 2267 innerTail333_4
    (if 3 ≤ innerCandidates333_3.length then search edge 3 [2267, 3527] innerCandidates333_3 5794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan333_4

def innerTail333_2 : List Nat := 3323 :: innerTail333_3
def innerCandidates333_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates333_2_checked : candidates edge 4 2267 3323 innerTail333_3 best.bound = innerCandidates333_2 := by
  unfold candidates
  have row : edge 3323 = SearchData.row3323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan333_2 : scan edge (search edge 3) 4 [2267] 2267 innerTail333_2 best = best := by
  rw [innerTail333_2, scan, if_neg (by decide), innerCandidates333_2_checked]
  change scan edge (search edge 3) 4 [2267] 2267 innerTail333_3
    (if 3 ≤ innerCandidates333_2.length then search edge 3 [2267, 3323] innerCandidates333_2 5590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan333_3

def innerTail333_1 : List Nat := 2939 :: innerTail333_2
def innerCandidates333_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates333_1_checked : candidates edge 4 2267 2939 innerTail333_2 best.bound = innerCandidates333_1 := by
  unfold candidates
  have row : edge 2939 = SearchData.row2939.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan333_1 : scan edge (search edge 3) 4 [2267] 2267 innerTail333_1 best = best := by
  rw [innerTail333_1, scan, if_neg (by decide), innerCandidates333_1_checked]
  change scan edge (search edge 3) 4 [2267] 2267 innerTail333_2
    (if 3 ≤ innerCandidates333_1.length then search edge 3 [2267, 2939] innerCandidates333_1 5206 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan333_2

def innerTail333_0 : List Nat := 2729 :: innerTail333_1
def innerCandidates333_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates333_0_checked : candidates edge 4 2267 2729 innerTail333_1 best.bound = innerCandidates333_0 := by
  unfold candidates
  have row : edge 2729 = SearchData.row2729.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan333_0 : scan edge (search edge 3) 4 [2267] 2267 innerTail333_0 best = best := by
  rw [innerTail333_0, scan, if_neg (by decide), innerCandidates333_0_checked]
  change scan edge (search edge 3) 4 [2267] 2267 innerTail333_1
    (if 3 ≤ innerCandidates333_0.length then search edge 3 [2267, 2729] innerCandidates333_0 4996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan333_1

theorem rootBranch333 : search edge 4 [2267] rootCandidates333 2267 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2267] 2267 innerTail333_0 best = best
  exact innerScan333_0

def rootCandidates332 : List Nat := [2593, 3217, 3307, 4021, 4111, 4201, 5503, 5881, 6451, 6841, 7669, 7927, 8191, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates332_checked : candidates edge 5 0 2251 tail333 best.bound = rootCandidates332 := by
  unfold candidates
  have row : edge 2251 = PrimePairCache.smallRow2251.contains := by funext q; rfl
  rw [row]
  decide

def innerTail332_14 : List Nat := []
theorem innerScan332_14 : scan edge (search edge 3) 4 [2251] 2251 innerTail332_14 best = best := rfl

def innerTail332_13 : List Nat := 8233 :: innerTail332_14
def innerCandidates332_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates332_13_checked : candidates edge 4 2251 8233 innerTail332_14 best.bound = innerCandidates332_13 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan332_13 : scan edge (search edge 3) 4 [2251] 2251 innerTail332_13 best = best := by
  rw [innerTail332_13, scan, if_neg (by decide), innerCandidates332_13_checked]
  change scan edge (search edge 3) 4 [2251] 2251 innerTail332_14
    (if 3 ≤ innerCandidates332_13.length then search edge 3 [2251, 8233] innerCandidates332_13 10484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan332_14

def innerTail332_12 : List Nat := 8191 :: innerTail332_13
def innerCandidates332_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates332_12_checked : candidates edge 4 2251 8191 innerTail332_13 best.bound = innerCandidates332_12 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan332_12 : scan edge (search edge 3) 4 [2251] 2251 innerTail332_12 best = best := by
  rw [innerTail332_12, scan, if_neg (by decide), innerCandidates332_12_checked]
  change scan edge (search edge 3) 4 [2251] 2251 innerTail332_13
    (if 3 ≤ innerCandidates332_12.length then search edge 3 [2251, 8191] innerCandidates332_12 10442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan332_13

def innerTail332_11 : List Nat := 7927 :: innerTail332_12
def innerCandidates332_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates332_11_checked : candidates edge 4 2251 7927 innerTail332_12 best.bound = innerCandidates332_11 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan332_11 : scan edge (search edge 3) 4 [2251] 2251 innerTail332_11 best = best := by
  rw [innerTail332_11, scan, if_neg (by decide), innerCandidates332_11_checked]
  change scan edge (search edge 3) 4 [2251] 2251 innerTail332_12
    (if 3 ≤ innerCandidates332_11.length then search edge 3 [2251, 7927] innerCandidates332_11 10178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan332_12

def innerTail332_10 : List Nat := 7669 :: innerTail332_11
def innerCandidates332_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates332_10_checked : candidates edge 4 2251 7669 innerTail332_11 best.bound = innerCandidates332_10 := by
  unfold candidates
  have row : edge 7669 = SearchData.row7669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan332_10 : scan edge (search edge 3) 4 [2251] 2251 innerTail332_10 best = best := by
  rw [innerTail332_10, scan, if_neg (by decide), innerCandidates332_10_checked]
  change scan edge (search edge 3) 4 [2251] 2251 innerTail332_11
    (if 3 ≤ innerCandidates332_10.length then search edge 3 [2251, 7669] innerCandidates332_10 9920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan332_11

def innerTail332_9 : List Nat := 6841 :: innerTail332_10
def innerCandidates332_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates332_9_checked : candidates edge 4 2251 6841 innerTail332_10 best.bound = innerCandidates332_9 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan332_9 : scan edge (search edge 3) 4 [2251] 2251 innerTail332_9 best = best := by
  rw [innerTail332_9, scan, if_neg (by decide), innerCandidates332_9_checked]
  change scan edge (search edge 3) 4 [2251] 2251 innerTail332_10
    (if 3 ≤ innerCandidates332_9.length then search edge 3 [2251, 6841] innerCandidates332_9 9092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan332_10

def innerTail332_8 : List Nat := 6451 :: innerTail332_9
def innerCandidates332_8 : List Nat := [8191]
set_option maxRecDepth 4096 in
theorem innerCandidates332_8_checked : candidates edge 4 2251 6451 innerTail332_9 best.bound = innerCandidates332_8 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan332_8 : scan edge (search edge 3) 4 [2251] 2251 innerTail332_8 best = best := by
  rw [innerTail332_8, scan, if_neg (by decide), innerCandidates332_8_checked]
  change scan edge (search edge 3) 4 [2251] 2251 innerTail332_9
    (if 3 ≤ innerCandidates332_8.length then search edge 3 [2251, 6451] innerCandidates332_8 8702 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan332_9

def innerTail332_7 : List Nat := 5881 :: innerTail332_8
def innerCandidates332_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates332_7_checked : candidates edge 4 2251 5881 innerTail332_8 best.bound = innerCandidates332_7 := by
  unfold candidates
  have row : edge 5881 = SearchData.row5881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan332_7 : scan edge (search edge 3) 4 [2251] 2251 innerTail332_7 best = best := by
  rw [innerTail332_7, scan, if_neg (by decide), innerCandidates332_7_checked]
  change scan edge (search edge 3) 4 [2251] 2251 innerTail332_8
    (if 3 ≤ innerCandidates332_7.length then search edge 3 [2251, 5881] innerCandidates332_7 8132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan332_8

def innerTail332_6 : List Nat := 5503 :: innerTail332_7
def innerCandidates332_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates332_6_checked : candidates edge 4 2251 5503 innerTail332_7 best.bound = innerCandidates332_6 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan332_6 : scan edge (search edge 3) 4 [2251] 2251 innerTail332_6 best = best := by
  rw [innerTail332_6, scan, if_neg (by decide), innerCandidates332_6_checked]
  change scan edge (search edge 3) 4 [2251] 2251 innerTail332_7
    (if 3 ≤ innerCandidates332_6.length then search edge 3 [2251, 5503] innerCandidates332_6 7754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan332_7

def innerTail332_5 : List Nat := 4201 :: innerTail332_6
def innerCandidates332_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates332_5_checked : candidates edge 4 2251 4201 innerTail332_6 best.bound = innerCandidates332_5 := by
  unfold candidates
  have row : edge 4201 = SearchData.row4201.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan332_5 : scan edge (search edge 3) 4 [2251] 2251 innerTail332_5 best = best := by
  rw [innerTail332_5, scan, if_neg (by decide), innerCandidates332_5_checked]
  change scan edge (search edge 3) 4 [2251] 2251 innerTail332_6
    (if 3 ≤ innerCandidates332_5.length then search edge 3 [2251, 4201] innerCandidates332_5 6452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan332_6

def innerTail332_4 : List Nat := 4111 :: innerTail332_5
def innerCandidates332_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates332_4_checked : candidates edge 4 2251 4111 innerTail332_5 best.bound = innerCandidates332_4 := by
  unfold candidates
  have row : edge 4111 = SearchData.row4111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan332_4 : scan edge (search edge 3) 4 [2251] 2251 innerTail332_4 best = best := by
  rw [innerTail332_4, scan, if_neg (by decide), innerCandidates332_4_checked]
  change scan edge (search edge 3) 4 [2251] 2251 innerTail332_5
    (if 3 ≤ innerCandidates332_4.length then search edge 3 [2251, 4111] innerCandidates332_4 6362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan332_5

def innerTail332_3 : List Nat := 4021 :: innerTail332_4
def innerCandidates332_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates332_3_checked : candidates edge 4 2251 4021 innerTail332_4 best.bound = innerCandidates332_3 := by
  unfold candidates
  have row : edge 4021 = SearchData.row4021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan332_3 : scan edge (search edge 3) 4 [2251] 2251 innerTail332_3 best = best := by
  rw [innerTail332_3, scan, if_neg (by decide), innerCandidates332_3_checked]
  change scan edge (search edge 3) 4 [2251] 2251 innerTail332_4
    (if 3 ≤ innerCandidates332_3.length then search edge 3 [2251, 4021] innerCandidates332_3 6272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan332_4

def innerTail332_2 : List Nat := 3307 :: innerTail332_3
def innerCandidates332_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates332_2_checked : candidates edge 4 2251 3307 innerTail332_3 best.bound = innerCandidates332_2 := by
  unfold candidates
  have row : edge 3307 = SearchData.row3307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan332_2 : scan edge (search edge 3) 4 [2251] 2251 innerTail332_2 best = best := by
  rw [innerTail332_2, scan, if_neg (by decide), innerCandidates332_2_checked]
  change scan edge (search edge 3) 4 [2251] 2251 innerTail332_3
    (if 3 ≤ innerCandidates332_2.length then search edge 3 [2251, 3307] innerCandidates332_2 5558 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan332_3

def innerTail332_1 : List Nat := 3217 :: innerTail332_2
def innerCandidates332_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates332_1_checked : candidates edge 4 2251 3217 innerTail332_2 best.bound = innerCandidates332_1 := by
  unfold candidates
  have row : edge 3217 = SearchData.row3217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan332_1 : scan edge (search edge 3) 4 [2251] 2251 innerTail332_1 best = best := by
  rw [innerTail332_1, scan, if_neg (by decide), innerCandidates332_1_checked]
  change scan edge (search edge 3) 4 [2251] 2251 innerTail332_2
    (if 3 ≤ innerCandidates332_1.length then search edge 3 [2251, 3217] innerCandidates332_1 5468 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan332_2

def innerTail332_0 : List Nat := 2593 :: innerTail332_1
def innerCandidates332_0 : List Nat := [3307, 7927]
set_option maxRecDepth 4096 in
theorem innerCandidates332_0_checked : candidates edge 4 2251 2593 innerTail332_1 best.bound = innerCandidates332_0 := by
  unfold candidates
  have row : edge 2593 = PrimePairCache.smallRow2593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan332_0 : scan edge (search edge 3) 4 [2251] 2251 innerTail332_0 best = best := by
  rw [innerTail332_0, scan, if_neg (by decide), innerCandidates332_0_checked]
  change scan edge (search edge 3) 4 [2251] 2251 innerTail332_1
    (if 3 ≤ innerCandidates332_0.length then search edge 3 [2251, 2593] innerCandidates332_0 4844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan332_1

theorem rootBranch332 : search edge 4 [2251] rootCandidates332 2251 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2251] 2251 innerTail332_0 best = best
  exact innerScan332_0

end ProjectEulerSolutions.PrimePairUpper.Search
