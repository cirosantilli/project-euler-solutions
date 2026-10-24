import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates208 : List Nat := [1549, 2293, 2671, 3529, 3547, 4813, 5821, 6781, 7603]
set_option maxRecDepth 4096 in
theorem rootCandidates208_checked : candidates edge 5 0 1297 tail209 best.bound = rootCandidates208 := by
  unfold candidates
  have row : edge 1297 = PrimePairCache.smallRow1297.contains := by funext q; rfl
  rw [row]
  decide

def innerTail208_9 : List Nat := []
theorem innerScan208_9 : scan edge (search edge 3) 4 [1297] 1297 innerTail208_9 best = best := rfl

def innerTail208_8 : List Nat := 7603 :: innerTail208_9
def innerCandidates208_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates208_8_checked : candidates edge 4 1297 7603 innerTail208_9 best.bound = innerCandidates208_8 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan208_8 : scan edge (search edge 3) 4 [1297] 1297 innerTail208_8 best = best := by
  rw [innerTail208_8, scan, if_neg (by decide), innerCandidates208_8_checked]
  change scan edge (search edge 3) 4 [1297] 1297 innerTail208_9
    (if 3 ≤ innerCandidates208_8.length then search edge 3 [1297, 7603] innerCandidates208_8 8900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan208_9

def innerTail208_7 : List Nat := 6781 :: innerTail208_8
def innerCandidates208_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates208_7_checked : candidates edge 4 1297 6781 innerTail208_8 best.bound = innerCandidates208_7 := by
  unfold candidates
  have row : edge 6781 = SearchData.row6781.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan208_7 : scan edge (search edge 3) 4 [1297] 1297 innerTail208_7 best = best := by
  rw [innerTail208_7, scan, if_neg (by decide), innerCandidates208_7_checked]
  change scan edge (search edge 3) 4 [1297] 1297 innerTail208_8
    (if 3 ≤ innerCandidates208_7.length then search edge 3 [1297, 6781] innerCandidates208_7 8078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan208_8

def innerTail208_6 : List Nat := 5821 :: innerTail208_7
def innerCandidates208_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates208_6_checked : candidates edge 4 1297 5821 innerTail208_7 best.bound = innerCandidates208_6 := by
  unfold candidates
  have row : edge 5821 = SearchData.row5821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan208_6 : scan edge (search edge 3) 4 [1297] 1297 innerTail208_6 best = best := by
  rw [innerTail208_6, scan, if_neg (by decide), innerCandidates208_6_checked]
  change scan edge (search edge 3) 4 [1297] 1297 innerTail208_7
    (if 3 ≤ innerCandidates208_6.length then search edge 3 [1297, 5821] innerCandidates208_6 7118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan208_7

def innerTail208_5 : List Nat := 4813 :: innerTail208_6
def innerCandidates208_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates208_5_checked : candidates edge 4 1297 4813 innerTail208_6 best.bound = innerCandidates208_5 := by
  unfold candidates
  have row : edge 4813 = SearchData.row4813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan208_5 : scan edge (search edge 3) 4 [1297] 1297 innerTail208_5 best = best := by
  rw [innerTail208_5, scan, if_neg (by decide), innerCandidates208_5_checked]
  change scan edge (search edge 3) 4 [1297] 1297 innerTail208_6
    (if 3 ≤ innerCandidates208_5.length then search edge 3 [1297, 4813] innerCandidates208_5 6110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan208_6

def innerTail208_4 : List Nat := 3547 :: innerTail208_5
def innerCandidates208_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates208_4_checked : candidates edge 4 1297 3547 innerTail208_5 best.bound = innerCandidates208_4 := by
  unfold candidates
  have row : edge 3547 = SearchData.row3547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan208_4 : scan edge (search edge 3) 4 [1297] 1297 innerTail208_4 best = best := by
  rw [innerTail208_4, scan, if_neg (by decide), innerCandidates208_4_checked]
  change scan edge (search edge 3) 4 [1297] 1297 innerTail208_5
    (if 3 ≤ innerCandidates208_4.length then search edge 3 [1297, 3547] innerCandidates208_4 4844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan208_5

def innerTail208_3 : List Nat := 3529 :: innerTail208_4
def innerCandidates208_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates208_3_checked : candidates edge 4 1297 3529 innerTail208_4 best.bound = innerCandidates208_3 := by
  unfold candidates
  have row : edge 3529 = SearchData.row3529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan208_3 : scan edge (search edge 3) 4 [1297] 1297 innerTail208_3 best = best := by
  rw [innerTail208_3, scan, if_neg (by decide), innerCandidates208_3_checked]
  change scan edge (search edge 3) 4 [1297] 1297 innerTail208_4
    (if 3 ≤ innerCandidates208_3.length then search edge 3 [1297, 3529] innerCandidates208_3 4826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan208_4

def innerTail208_2 : List Nat := 2671 :: innerTail208_3
def innerCandidates208_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates208_2_checked : candidates edge 4 1297 2671 innerTail208_3 best.bound = innerCandidates208_2 := by
  unfold candidates
  have row : edge 2671 = SearchData.row2671.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan208_2 : scan edge (search edge 3) 4 [1297] 1297 innerTail208_2 best = best := by
  rw [innerTail208_2, scan, if_neg (by decide), innerCandidates208_2_checked]
  change scan edge (search edge 3) 4 [1297] 1297 innerTail208_3
    (if 3 ≤ innerCandidates208_2.length then search edge 3 [1297, 2671] innerCandidates208_2 3968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan208_3

def innerTail208_1 : List Nat := 2293 :: innerTail208_2
def innerCandidates208_1 : List Nat := [2671]
set_option maxRecDepth 4096 in
theorem innerCandidates208_1_checked : candidates edge 4 1297 2293 innerTail208_2 best.bound = innerCandidates208_1 := by
  unfold candidates
  have row : edge 2293 = PrimePairCache.smallRow2293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan208_1 : scan edge (search edge 3) 4 [1297] 1297 innerTail208_1 best = best := by
  rw [innerTail208_1, scan, if_neg (by decide), innerCandidates208_1_checked]
  change scan edge (search edge 3) 4 [1297] 1297 innerTail208_2
    (if 3 ≤ innerCandidates208_1.length then search edge 3 [1297, 2293] innerCandidates208_1 3590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan208_2

def innerTail208_0 : List Nat := 1549 :: innerTail208_1
def innerCandidates208_0 : List Nat := [3529]
set_option maxRecDepth 4096 in
theorem innerCandidates208_0_checked : candidates edge 4 1297 1549 innerTail208_1 best.bound = innerCandidates208_0 := by
  unfold candidates
  have row : edge 1549 = PrimePairCache.smallRow1549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan208_0 : scan edge (search edge 3) 4 [1297] 1297 innerTail208_0 best = best := by
  rw [innerTail208_0, scan, if_neg (by decide), innerCandidates208_0_checked]
  change scan edge (search edge 3) 4 [1297] 1297 innerTail208_1
    (if 3 ≤ innerCandidates208_0.length then search edge 3 [1297, 1549] innerCandidates208_0 2846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan208_1

theorem rootBranch208 : search edge 4 [1297] rootCandidates208 1297 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1297] 1297 innerTail208_0 best = best
  exact innerScan208_0

def rootCandidates207 : List Nat := [1777, 2281, 2293, 3457, 3793, 4273, 4513, 4903, 5059, 5557, 6733, 6907, 7741, 8209, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates207_checked : candidates edge 5 0 1291 tail208 best.bound = rootCandidates207 := by
  unfold candidates
  have row : edge 1291 = PrimePairCache.smallRow1291.contains := by funext q; rfl
  rw [row]
  decide

def innerTail207_15 : List Nat := []
theorem innerScan207_15 : scan edge (search edge 3) 4 [1291] 1291 innerTail207_15 best = best := rfl

def innerTail207_14 : List Nat := 8293 :: innerTail207_15
def innerCandidates207_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates207_14_checked : candidates edge 4 1291 8293 innerTail207_15 best.bound = innerCandidates207_14 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan207_14 : scan edge (search edge 3) 4 [1291] 1291 innerTail207_14 best = best := by
  rw [innerTail207_14, scan, if_neg (by decide), innerCandidates207_14_checked]
  change scan edge (search edge 3) 4 [1291] 1291 innerTail207_15
    (if 3 ≤ innerCandidates207_14.length then search edge 3 [1291, 8293] innerCandidates207_14 9584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan207_15

def innerTail207_13 : List Nat := 8209 :: innerTail207_14
def innerCandidates207_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates207_13_checked : candidates edge 4 1291 8209 innerTail207_14 best.bound = innerCandidates207_13 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan207_13 : scan edge (search edge 3) 4 [1291] 1291 innerTail207_13 best = best := by
  rw [innerTail207_13, scan, if_neg (by decide), innerCandidates207_13_checked]
  change scan edge (search edge 3) 4 [1291] 1291 innerTail207_14
    (if 3 ≤ innerCandidates207_13.length then search edge 3 [1291, 8209] innerCandidates207_13 9500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan207_14

def innerTail207_12 : List Nat := 7741 :: innerTail207_13
def innerCandidates207_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates207_12_checked : candidates edge 4 1291 7741 innerTail207_13 best.bound = innerCandidates207_12 := by
  unfold candidates
  have row : edge 7741 = SearchData.row7741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan207_12 : scan edge (search edge 3) 4 [1291] 1291 innerTail207_12 best = best := by
  rw [innerTail207_12, scan, if_neg (by decide), innerCandidates207_12_checked]
  change scan edge (search edge 3) 4 [1291] 1291 innerTail207_13
    (if 3 ≤ innerCandidates207_12.length then search edge 3 [1291, 7741] innerCandidates207_12 9032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan207_13

def innerTail207_11 : List Nat := 6907 :: innerTail207_12
def innerCandidates207_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates207_11_checked : candidates edge 4 1291 6907 innerTail207_12 best.bound = innerCandidates207_11 := by
  unfold candidates
  have row : edge 6907 = SearchData.row6907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan207_11 : scan edge (search edge 3) 4 [1291] 1291 innerTail207_11 best = best := by
  rw [innerTail207_11, scan, if_neg (by decide), innerCandidates207_11_checked]
  change scan edge (search edge 3) 4 [1291] 1291 innerTail207_12
    (if 3 ≤ innerCandidates207_11.length then search edge 3 [1291, 6907] innerCandidates207_11 8198 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan207_12

def innerTail207_10 : List Nat := 6733 :: innerTail207_11
def innerCandidates207_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates207_10_checked : candidates edge 4 1291 6733 innerTail207_11 best.bound = innerCandidates207_10 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan207_10 : scan edge (search edge 3) 4 [1291] 1291 innerTail207_10 best = best := by
  rw [innerTail207_10, scan, if_neg (by decide), innerCandidates207_10_checked]
  change scan edge (search edge 3) 4 [1291] 1291 innerTail207_11
    (if 3 ≤ innerCandidates207_10.length then search edge 3 [1291, 6733] innerCandidates207_10 8024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan207_11

def innerTail207_9 : List Nat := 5557 :: innerTail207_10
def innerCandidates207_9 : List Nat := [8293]
set_option maxRecDepth 4096 in
theorem innerCandidates207_9_checked : candidates edge 4 1291 5557 innerTail207_10 best.bound = innerCandidates207_9 := by
  unfold candidates
  have row : edge 5557 = SearchData.row5557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan207_9 : scan edge (search edge 3) 4 [1291] 1291 innerTail207_9 best = best := by
  rw [innerTail207_9, scan, if_neg (by decide), innerCandidates207_9_checked]
  change scan edge (search edge 3) 4 [1291] 1291 innerTail207_10
    (if 3 ≤ innerCandidates207_9.length then search edge 3 [1291, 5557] innerCandidates207_9 6848 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan207_10

def innerTail207_8 : List Nat := 5059 :: innerTail207_9
def innerCandidates207_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates207_8_checked : candidates edge 4 1291 5059 innerTail207_9 best.bound = innerCandidates207_8 := by
  unfold candidates
  have row : edge 5059 = SearchData.row5059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan207_8 : scan edge (search edge 3) 4 [1291] 1291 innerTail207_8 best = best := by
  rw [innerTail207_8, scan, if_neg (by decide), innerCandidates207_8_checked]
  change scan edge (search edge 3) 4 [1291] 1291 innerTail207_9
    (if 3 ≤ innerCandidates207_8.length then search edge 3 [1291, 5059] innerCandidates207_8 6350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan207_9

def innerTail207_7 : List Nat := 4903 :: innerTail207_8
def innerCandidates207_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates207_7_checked : candidates edge 4 1291 4903 innerTail207_8 best.bound = innerCandidates207_7 := by
  unfold candidates
  have row : edge 4903 = SearchData.row4903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan207_7 : scan edge (search edge 3) 4 [1291] 1291 innerTail207_7 best = best := by
  rw [innerTail207_7, scan, if_neg (by decide), innerCandidates207_7_checked]
  change scan edge (search edge 3) 4 [1291] 1291 innerTail207_8
    (if 3 ≤ innerCandidates207_7.length then search edge 3 [1291, 4903] innerCandidates207_7 6194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan207_8

def innerTail207_6 : List Nat := 4513 :: innerTail207_7
def innerCandidates207_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates207_6_checked : candidates edge 4 1291 4513 innerTail207_7 best.bound = innerCandidates207_6 := by
  unfold candidates
  have row : edge 4513 = SearchData.row4513.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan207_6 : scan edge (search edge 3) 4 [1291] 1291 innerTail207_6 best = best := by
  rw [innerTail207_6, scan, if_neg (by decide), innerCandidates207_6_checked]
  change scan edge (search edge 3) 4 [1291] 1291 innerTail207_7
    (if 3 ≤ innerCandidates207_6.length then search edge 3 [1291, 4513] innerCandidates207_6 5804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan207_7

def innerTail207_5 : List Nat := 4273 :: innerTail207_6
def innerCandidates207_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates207_5_checked : candidates edge 4 1291 4273 innerTail207_6 best.bound = innerCandidates207_5 := by
  unfold candidates
  have row : edge 4273 = SearchData.row4273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan207_5 : scan edge (search edge 3) 4 [1291] 1291 innerTail207_5 best = best := by
  rw [innerTail207_5, scan, if_neg (by decide), innerCandidates207_5_checked]
  change scan edge (search edge 3) 4 [1291] 1291 innerTail207_6
    (if 3 ≤ innerCandidates207_5.length then search edge 3 [1291, 4273] innerCandidates207_5 5564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan207_6

def innerTail207_4 : List Nat := 3793 :: innerTail207_5
def innerCandidates207_4 : List Nat := [6733, 6907]
set_option maxRecDepth 4096 in
theorem innerCandidates207_4_checked : candidates edge 4 1291 3793 innerTail207_5 best.bound = innerCandidates207_4 := by
  unfold candidates
  have row : edge 3793 = SearchData.row3793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan207_4 : scan edge (search edge 3) 4 [1291] 1291 innerTail207_4 best = best := by
  rw [innerTail207_4, scan, if_neg (by decide), innerCandidates207_4_checked]
  change scan edge (search edge 3) 4 [1291] 1291 innerTail207_5
    (if 3 ≤ innerCandidates207_4.length then search edge 3 [1291, 3793] innerCandidates207_4 5084 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan207_5

def innerTail207_3 : List Nat := 3457 :: innerTail207_4
def innerCandidates207_3 : List Nat := [8209]
set_option maxRecDepth 4096 in
theorem innerCandidates207_3_checked : candidates edge 4 1291 3457 innerTail207_4 best.bound = innerCandidates207_3 := by
  unfold candidates
  have row : edge 3457 = SearchData.row3457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan207_3 : scan edge (search edge 3) 4 [1291] 1291 innerTail207_3 best = best := by
  rw [innerTail207_3, scan, if_neg (by decide), innerCandidates207_3_checked]
  change scan edge (search edge 3) 4 [1291] 1291 innerTail207_4
    (if 3 ≤ innerCandidates207_3.length then search edge 3 [1291, 3457] innerCandidates207_3 4748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan207_4

def innerTail207_2 : List Nat := 2293 :: innerTail207_3
def innerCandidates207_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates207_2_checked : candidates edge 4 1291 2293 innerTail207_3 best.bound = innerCandidates207_2 := by
  unfold candidates
  have row : edge 2293 = PrimePairCache.smallRow2293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan207_2 : scan edge (search edge 3) 4 [1291] 1291 innerTail207_2 best = best := by
  rw [innerTail207_2, scan, if_neg (by decide), innerCandidates207_2_checked]
  change scan edge (search edge 3) 4 [1291] 1291 innerTail207_3
    (if 3 ≤ innerCandidates207_2.length then search edge 3 [1291, 2293] innerCandidates207_2 3584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan207_3

def innerTail207_1 : List Nat := 2281 :: innerTail207_2
def innerCandidates207_1 : List Nat := [4273]
set_option maxRecDepth 4096 in
theorem innerCandidates207_1_checked : candidates edge 4 1291 2281 innerTail207_2 best.bound = innerCandidates207_1 := by
  unfold candidates
  have row : edge 2281 = PrimePairCache.smallRow2281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan207_1 : scan edge (search edge 3) 4 [1291] 1291 innerTail207_1 best = best := by
  rw [innerTail207_1, scan, if_neg (by decide), innerCandidates207_1_checked]
  change scan edge (search edge 3) 4 [1291] 1291 innerTail207_2
    (if 3 ≤ innerCandidates207_1.length then search edge 3 [1291, 2281] innerCandidates207_1 3572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan207_2

def innerTail207_0 : List Nat := 1777 :: innerTail207_1
def innerCandidates207_0 : List Nat := [7741]
set_option maxRecDepth 4096 in
theorem innerCandidates207_0_checked : candidates edge 4 1291 1777 innerTail207_1 best.bound = innerCandidates207_0 := by
  unfold candidates
  have row : edge 1777 = PrimePairCache.smallRow1777.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan207_0 : scan edge (search edge 3) 4 [1291] 1291 innerTail207_0 best = best := by
  rw [innerTail207_0, scan, if_neg (by decide), innerCandidates207_0_checked]
  change scan edge (search edge 3) 4 [1291] 1291 innerTail207_1
    (if 3 ≤ innerCandidates207_0.length then search edge 3 [1291, 1777] innerCandidates207_0 3068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan207_1

theorem rootBranch207 : search edge 4 [1291] rootCandidates207 1291 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1291] 1291 innerTail207_0 best = best
  exact innerScan207_0

def rootCandidates206 : List Nat := [1409, 1721, 2027, 2039, 2237, 2801, 3767, 3833, 4139, 4733, 6311, 6449, 7541, 7853, 7919]
set_option maxRecDepth 4096 in
theorem rootCandidates206_checked : candidates edge 5 0 1289 tail207 best.bound = rootCandidates206 := by
  unfold candidates
  have row : edge 1289 = PrimePairCache.smallRow1289.contains := by funext q; rfl
  rw [row]
  decide

def innerTail206_15 : List Nat := []
theorem innerScan206_15 : scan edge (search edge 3) 4 [1289] 1289 innerTail206_15 best = best := rfl

def innerTail206_14 : List Nat := 7919 :: innerTail206_15
def innerCandidates206_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates206_14_checked : candidates edge 4 1289 7919 innerTail206_15 best.bound = innerCandidates206_14 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan206_14 : scan edge (search edge 3) 4 [1289] 1289 innerTail206_14 best = best := by
  rw [innerTail206_14, scan, if_neg (by decide), innerCandidates206_14_checked]
  change scan edge (search edge 3) 4 [1289] 1289 innerTail206_15
    (if 3 ≤ innerCandidates206_14.length then search edge 3 [1289, 7919] innerCandidates206_14 9208 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan206_15

def innerTail206_13 : List Nat := 7853 :: innerTail206_14
def innerCandidates206_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates206_13_checked : candidates edge 4 1289 7853 innerTail206_14 best.bound = innerCandidates206_13 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan206_13 : scan edge (search edge 3) 4 [1289] 1289 innerTail206_13 best = best := by
  rw [innerTail206_13, scan, if_neg (by decide), innerCandidates206_13_checked]
  change scan edge (search edge 3) 4 [1289] 1289 innerTail206_14
    (if 3 ≤ innerCandidates206_13.length then search edge 3 [1289, 7853] innerCandidates206_13 9142 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan206_14

def innerTail206_12 : List Nat := 7541 :: innerTail206_13
def innerCandidates206_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates206_12_checked : candidates edge 4 1289 7541 innerTail206_13 best.bound = innerCandidates206_12 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan206_12 : scan edge (search edge 3) 4 [1289] 1289 innerTail206_12 best = best := by
  rw [innerTail206_12, scan, if_neg (by decide), innerCandidates206_12_checked]
  change scan edge (search edge 3) 4 [1289] 1289 innerTail206_13
    (if 3 ≤ innerCandidates206_12.length then search edge 3 [1289, 7541] innerCandidates206_12 8830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan206_13

def innerTail206_11 : List Nat := 6449 :: innerTail206_12
def innerCandidates206_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates206_11_checked : candidates edge 4 1289 6449 innerTail206_12 best.bound = innerCandidates206_11 := by
  unfold candidates
  have row : edge 6449 = SearchData.row6449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan206_11 : scan edge (search edge 3) 4 [1289] 1289 innerTail206_11 best = best := by
  rw [innerTail206_11, scan, if_neg (by decide), innerCandidates206_11_checked]
  change scan edge (search edge 3) 4 [1289] 1289 innerTail206_12
    (if 3 ≤ innerCandidates206_11.length then search edge 3 [1289, 6449] innerCandidates206_11 7738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan206_12

def innerTail206_10 : List Nat := 6311 :: innerTail206_11
def innerCandidates206_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates206_10_checked : candidates edge 4 1289 6311 innerTail206_11 best.bound = innerCandidates206_10 := by
  unfold candidates
  have row : edge 6311 = SearchData.row6311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan206_10 : scan edge (search edge 3) 4 [1289] 1289 innerTail206_10 best = best := by
  rw [innerTail206_10, scan, if_neg (by decide), innerCandidates206_10_checked]
  change scan edge (search edge 3) 4 [1289] 1289 innerTail206_11
    (if 3 ≤ innerCandidates206_10.length then search edge 3 [1289, 6311] innerCandidates206_10 7600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan206_11

def innerTail206_9 : List Nat := 4733 :: innerTail206_10
def innerCandidates206_9 : List Nat := [7919]
set_option maxRecDepth 4096 in
theorem innerCandidates206_9_checked : candidates edge 4 1289 4733 innerTail206_10 best.bound = innerCandidates206_9 := by
  unfold candidates
  have row : edge 4733 = SearchData.row4733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan206_9 : scan edge (search edge 3) 4 [1289] 1289 innerTail206_9 best = best := by
  rw [innerTail206_9, scan, if_neg (by decide), innerCandidates206_9_checked]
  change scan edge (search edge 3) 4 [1289] 1289 innerTail206_10
    (if 3 ≤ innerCandidates206_9.length then search edge 3 [1289, 4733] innerCandidates206_9 6022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan206_10

def innerTail206_8 : List Nat := 4139 :: innerTail206_9
def innerCandidates206_8 : List Nat := [7541]
set_option maxRecDepth 4096 in
theorem innerCandidates206_8_checked : candidates edge 4 1289 4139 innerTail206_9 best.bound = innerCandidates206_8 := by
  unfold candidates
  have row : edge 4139 = SearchData.row4139.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan206_8 : scan edge (search edge 3) 4 [1289] 1289 innerTail206_8 best = best := by
  rw [innerTail206_8, scan, if_neg (by decide), innerCandidates206_8_checked]
  change scan edge (search edge 3) 4 [1289] 1289 innerTail206_9
    (if 3 ≤ innerCandidates206_8.length then search edge 3 [1289, 4139] innerCandidates206_8 5428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan206_9

def innerTail206_7 : List Nat := 3833 :: innerTail206_8
def innerCandidates206_7 : List Nat := [4733, 6449]
set_option maxRecDepth 4096 in
theorem innerCandidates206_7_checked : candidates edge 4 1289 3833 innerTail206_8 best.bound = innerCandidates206_7 := by
  unfold candidates
  have row : edge 3833 = SearchData.row3833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan206_7 : scan edge (search edge 3) 4 [1289] 1289 innerTail206_7 best = best := by
  rw [innerTail206_7, scan, if_neg (by decide), innerCandidates206_7_checked]
  change scan edge (search edge 3) 4 [1289] 1289 innerTail206_8
    (if 3 ≤ innerCandidates206_7.length then search edge 3 [1289, 3833] innerCandidates206_7 5122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan206_8

def innerTail206_6 : List Nat := 3767 :: innerTail206_7
def innerCandidates206_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates206_6_checked : candidates edge 4 1289 3767 innerTail206_7 best.bound = innerCandidates206_6 := by
  unfold candidates
  have row : edge 3767 = SearchData.row3767.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan206_6 : scan edge (search edge 3) 4 [1289] 1289 innerTail206_6 best = best := by
  rw [innerTail206_6, scan, if_neg (by decide), innerCandidates206_6_checked]
  change scan edge (search edge 3) 4 [1289] 1289 innerTail206_7
    (if 3 ≤ innerCandidates206_6.length then search edge 3 [1289, 3767] innerCandidates206_6 5056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan206_7

def innerTail206_5 : List Nat := 2801 :: innerTail206_6
def innerCandidates206_5 : List Nat := [4733, 6311, 7919]
set_option maxRecDepth 4096 in
theorem innerCandidates206_5_checked : candidates edge 4 1289 2801 innerTail206_6 best.bound = innerCandidates206_5 := by
  unfold candidates
  have row : edge 2801 = SearchData.row2801.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild206_5 : search edge 3 [1289, 2801] innerCandidates206_5 4090 best = best := by decide

theorem innerScan206_5 : scan edge (search edge 3) 4 [1289] 1289 innerTail206_5 best = best := by
  rw [innerTail206_5, scan, if_neg (by decide), innerCandidates206_5_checked]
  change scan edge (search edge 3) 4 [1289] 1289 innerTail206_6
    (if 3 ≤ innerCandidates206_5.length then search edge 3 [1289, 2801] innerCandidates206_5 4090 best else best) = best
  rw [if_pos (by decide), innerChild206_5]
  exact innerScan206_6

def innerTail206_4 : List Nat := 2237 :: innerTail206_5
def innerCandidates206_4 : List Nat := [4139]
set_option maxRecDepth 4096 in
theorem innerCandidates206_4_checked : candidates edge 4 1289 2237 innerTail206_5 best.bound = innerCandidates206_4 := by
  unfold candidates
  have row : edge 2237 = PrimePairCache.smallRow2237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan206_4 : scan edge (search edge 3) 4 [1289] 1289 innerTail206_4 best = best := by
  rw [innerTail206_4, scan, if_neg (by decide), innerCandidates206_4_checked]
  change scan edge (search edge 3) 4 [1289] 1289 innerTail206_5
    (if 3 ≤ innerCandidates206_4.length then search edge 3 [1289, 2237] innerCandidates206_4 3526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan206_5

def innerTail206_3 : List Nat := 2039 :: innerTail206_4
def innerCandidates206_3 : List Nat := [3767]
set_option maxRecDepth 4096 in
theorem innerCandidates206_3_checked : candidates edge 4 1289 2039 innerTail206_4 best.bound = innerCandidates206_3 := by
  unfold candidates
  have row : edge 2039 = PrimePairCache.smallRow2039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan206_3 : scan edge (search edge 3) 4 [1289] 1289 innerTail206_3 best = best := by
  rw [innerTail206_3, scan, if_neg (by decide), innerCandidates206_3_checked]
  change scan edge (search edge 3) 4 [1289] 1289 innerTail206_4
    (if 3 ≤ innerCandidates206_3.length then search edge 3 [1289, 2039] innerCandidates206_3 3328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan206_4

def innerTail206_2 : List Nat := 2027 :: innerTail206_3
def innerCandidates206_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates206_2_checked : candidates edge 4 1289 2027 innerTail206_3 best.bound = innerCandidates206_2 := by
  unfold candidates
  have row : edge 2027 = PrimePairCache.smallRow2027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan206_2 : scan edge (search edge 3) 4 [1289] 1289 innerTail206_2 best = best := by
  rw [innerTail206_2, scan, if_neg (by decide), innerCandidates206_2_checked]
  change scan edge (search edge 3) 4 [1289] 1289 innerTail206_3
    (if 3 ≤ innerCandidates206_2.length then search edge 3 [1289, 2027] innerCandidates206_2 3316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan206_3

def innerTail206_1 : List Nat := 1721 :: innerTail206_2
def innerCandidates206_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates206_1_checked : candidates edge 4 1289 1721 innerTail206_2 best.bound = innerCandidates206_1 := by
  unfold candidates
  have row : edge 1721 = PrimePairCache.smallRow1721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan206_1 : scan edge (search edge 3) 4 [1289] 1289 innerTail206_1 best = best := by
  rw [innerTail206_1, scan, if_neg (by decide), innerCandidates206_1_checked]
  change scan edge (search edge 3) 4 [1289] 1289 innerTail206_2
    (if 3 ≤ innerCandidates206_1.length then search edge 3 [1289, 1721] innerCandidates206_1 3010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan206_2

def innerTail206_0 : List Nat := 1409 :: innerTail206_1
def innerCandidates206_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates206_0_checked : candidates edge 4 1289 1409 innerTail206_1 best.bound = innerCandidates206_0 := by
  unfold candidates
  have row : edge 1409 = PrimePairCache.smallRow1409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan206_0 : scan edge (search edge 3) 4 [1289] 1289 innerTail206_0 best = best := by
  rw [innerTail206_0, scan, if_neg (by decide), innerCandidates206_0_checked]
  change scan edge (search edge 3) 4 [1289] 1289 innerTail206_1
    (if 3 ≤ innerCandidates206_0.length then search edge 3 [1289, 1409] innerCandidates206_0 2698 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan206_1

theorem rootBranch206 : search edge 4 [1289] rootCandidates206 1289 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1289] 1289 innerTail206_0 best = best
  exact innerScan206_0

def rootCandidates205 : List Nat := [1619, 1667, 1913, 2111, 2423, 2741, 2801, 3593, 3617, 3719, 4127, 4397, 5021, 5441, 5807, 6221, 6287, 6827, 6869, 7001, 7247, 7757, 7949, 8237]
set_option maxRecDepth 4096 in
theorem rootCandidates205_checked : candidates edge 5 0 1283 tail206 best.bound = rootCandidates205 := by
  unfold candidates
  have row : edge 1283 = PrimePairCache.smallRow1283.contains := by funext q; rfl
  rw [row]
  decide

def innerTail205_24 : List Nat := []
theorem innerScan205_24 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_24 best = best := rfl

def innerTail205_23 : List Nat := 8237 :: innerTail205_24
def innerCandidates205_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates205_23_checked : candidates edge 4 1283 8237 innerTail205_24 best.bound = innerCandidates205_23 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_23 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_23 best = best := by
  rw [innerTail205_23, scan, if_neg (by decide), innerCandidates205_23_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_24
    (if 3 ≤ innerCandidates205_23.length then search edge 3 [1283, 8237] innerCandidates205_23 9520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_24

def innerTail205_22 : List Nat := 7949 :: innerTail205_23
def innerCandidates205_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates205_22_checked : candidates edge 4 1283 7949 innerTail205_23 best.bound = innerCandidates205_22 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_22 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_22 best = best := by
  rw [innerTail205_22, scan, if_neg (by decide), innerCandidates205_22_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_23
    (if 3 ≤ innerCandidates205_22.length then search edge 3 [1283, 7949] innerCandidates205_22 9232 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_23

def innerTail205_21 : List Nat := 7757 :: innerTail205_22
def innerCandidates205_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates205_21_checked : candidates edge 4 1283 7757 innerTail205_22 best.bound = innerCandidates205_21 := by
  unfold candidates
  have row : edge 7757 = SearchData.row7757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_21 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_21 best = best := by
  rw [innerTail205_21, scan, if_neg (by decide), innerCandidates205_21_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_22
    (if 3 ≤ innerCandidates205_21.length then search edge 3 [1283, 7757] innerCandidates205_21 9040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_22

def innerTail205_20 : List Nat := 7247 :: innerTail205_21
def innerCandidates205_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates205_20_checked : candidates edge 4 1283 7247 innerTail205_21 best.bound = innerCandidates205_20 := by
  unfold candidates
  have row : edge 7247 = SearchData.row7247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_20 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_20 best = best := by
  rw [innerTail205_20, scan, if_neg (by decide), innerCandidates205_20_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_21
    (if 3 ≤ innerCandidates205_20.length then search edge 3 [1283, 7247] innerCandidates205_20 8530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_21

def innerTail205_19 : List Nat := 7001 :: innerTail205_20
def innerCandidates205_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates205_19_checked : candidates edge 4 1283 7001 innerTail205_20 best.bound = innerCandidates205_19 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_19 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_19 best = best := by
  rw [innerTail205_19, scan, if_neg (by decide), innerCandidates205_19_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_20
    (if 3 ≤ innerCandidates205_19.length then search edge 3 [1283, 7001] innerCandidates205_19 8284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_20

def innerTail205_18 : List Nat := 6869 :: innerTail205_19
def innerCandidates205_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates205_18_checked : candidates edge 4 1283 6869 innerTail205_19 best.bound = innerCandidates205_18 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_18 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_18 best = best := by
  rw [innerTail205_18, scan, if_neg (by decide), innerCandidates205_18_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_19
    (if 3 ≤ innerCandidates205_18.length then search edge 3 [1283, 6869] innerCandidates205_18 8152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_19

def innerTail205_17 : List Nat := 6827 :: innerTail205_18
def innerCandidates205_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates205_17_checked : candidates edge 4 1283 6827 innerTail205_18 best.bound = innerCandidates205_17 := by
  unfold candidates
  have row : edge 6827 = SearchData.row6827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_17 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_17 best = best := by
  rw [innerTail205_17, scan, if_neg (by decide), innerCandidates205_17_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_18
    (if 3 ≤ innerCandidates205_17.length then search edge 3 [1283, 6827] innerCandidates205_17 8110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_18

def innerTail205_16 : List Nat := 6287 :: innerTail205_17
def innerCandidates205_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates205_16_checked : candidates edge 4 1283 6287 innerTail205_17 best.bound = innerCandidates205_16 := by
  unfold candidates
  have row : edge 6287 = SearchData.row6287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_16 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_16 best = best := by
  rw [innerTail205_16, scan, if_neg (by decide), innerCandidates205_16_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_17
    (if 3 ≤ innerCandidates205_16.length then search edge 3 [1283, 6287] innerCandidates205_16 7570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_17

def innerTail205_15 : List Nat := 6221 :: innerTail205_16
def innerCandidates205_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates205_15_checked : candidates edge 4 1283 6221 innerTail205_16 best.bound = innerCandidates205_15 := by
  unfold candidates
  have row : edge 6221 = SearchData.row6221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_15 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_15 best = best := by
  rw [innerTail205_15, scan, if_neg (by decide), innerCandidates205_15_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_16
    (if 3 ≤ innerCandidates205_15.length then search edge 3 [1283, 6221] innerCandidates205_15 7504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_16

def innerTail205_14 : List Nat := 5807 :: innerTail205_15
def innerCandidates205_14 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates205_14_checked : candidates edge 4 1283 5807 innerTail205_15 best.bound = innerCandidates205_14 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_14 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_14 best = best := by
  rw [innerTail205_14, scan, if_neg (by decide), innerCandidates205_14_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_15
    (if 3 ≤ innerCandidates205_14.length then search edge 3 [1283, 5807] innerCandidates205_14 7090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_15

def innerTail205_13 : List Nat := 5441 :: innerTail205_14
def innerCandidates205_13 : List Nat := [6287]
set_option maxRecDepth 4096 in
theorem innerCandidates205_13_checked : candidates edge 4 1283 5441 innerTail205_14 best.bound = innerCandidates205_13 := by
  unfold candidates
  have row : edge 5441 = SearchData.row5441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_13 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_13 best = best := by
  rw [innerTail205_13, scan, if_neg (by decide), innerCandidates205_13_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_14
    (if 3 ≤ innerCandidates205_13.length then search edge 3 [1283, 5441] innerCandidates205_13 6724 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_14

def innerTail205_12 : List Nat := 5021 :: innerTail205_13
def innerCandidates205_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates205_12_checked : candidates edge 4 1283 5021 innerTail205_13 best.bound = innerCandidates205_12 := by
  unfold candidates
  have row : edge 5021 = SearchData.row5021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_12 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_12 best = best := by
  rw [innerTail205_12, scan, if_neg (by decide), innerCandidates205_12_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_13
    (if 3 ≤ innerCandidates205_12.length then search edge 3 [1283, 5021] innerCandidates205_12 6304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_13

def innerTail205_11 : List Nat := 4397 :: innerTail205_12
def innerCandidates205_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates205_11_checked : candidates edge 4 1283 4397 innerTail205_12 best.bound = innerCandidates205_11 := by
  unfold candidates
  have row : edge 4397 = SearchData.row4397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_11 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_11 best = best := by
  rw [innerTail205_11, scan, if_neg (by decide), innerCandidates205_11_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_12
    (if 3 ≤ innerCandidates205_11.length then search edge 3 [1283, 4397] innerCandidates205_11 5680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_12

def innerTail205_10 : List Nat := 4127 :: innerTail205_11
def innerCandidates205_10 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates205_10_checked : candidates edge 4 1283 4127 innerTail205_11 best.bound = innerCandidates205_10 := by
  unfold candidates
  have row : edge 4127 = SearchData.row4127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_10 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_10 best = best := by
  rw [innerTail205_10, scan, if_neg (by decide), innerCandidates205_10_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_11
    (if 3 ≤ innerCandidates205_10.length then search edge 3 [1283, 4127] innerCandidates205_10 5410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_11

def innerTail205_9 : List Nat := 3719 :: innerTail205_10
def innerCandidates205_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates205_9_checked : candidates edge 4 1283 3719 innerTail205_10 best.bound = innerCandidates205_9 := by
  unfold candidates
  have row : edge 3719 = SearchData.row3719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_9 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_9 best = best := by
  rw [innerTail205_9, scan, if_neg (by decide), innerCandidates205_9_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_10
    (if 3 ≤ innerCandidates205_9.length then search edge 3 [1283, 3719] innerCandidates205_9 5002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_10

def innerTail205_8 : List Nat := 3617 :: innerTail205_9
def innerCandidates205_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates205_8_checked : candidates edge 4 1283 3617 innerTail205_9 best.bound = innerCandidates205_8 := by
  unfold candidates
  have row : edge 3617 = SearchData.row3617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_8 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_8 best = best := by
  rw [innerTail205_8, scan, if_neg (by decide), innerCandidates205_8_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_9
    (if 3 ≤ innerCandidates205_8.length then search edge 3 [1283, 3617] innerCandidates205_8 4900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_9

def innerTail205_7 : List Nat := 3593 :: innerTail205_8
def innerCandidates205_7 : List Nat := [3719, 5021]
set_option maxRecDepth 4096 in
theorem innerCandidates205_7_checked : candidates edge 4 1283 3593 innerTail205_8 best.bound = innerCandidates205_7 := by
  unfold candidates
  have row : edge 3593 = SearchData.row3593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_7 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_7 best = best := by
  rw [innerTail205_7, scan, if_neg (by decide), innerCandidates205_7_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_8
    (if 3 ≤ innerCandidates205_7.length then search edge 3 [1283, 3593] innerCandidates205_7 4876 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_8

def innerTail205_6 : List Nat := 2801 :: innerTail205_7
def innerCandidates205_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates205_6_checked : candidates edge 4 1283 2801 innerTail205_7 best.bound = innerCandidates205_6 := by
  unfold candidates
  have row : edge 2801 = SearchData.row2801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_6 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_6 best = best := by
  rw [innerTail205_6, scan, if_neg (by decide), innerCandidates205_6_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_7
    (if 3 ≤ innerCandidates205_6.length then search edge 3 [1283, 2801] innerCandidates205_6 4084 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_7

def innerTail205_5 : List Nat := 2741 :: innerTail205_6
def innerCandidates205_5 : List Nat := [7757]
set_option maxRecDepth 4096 in
theorem innerCandidates205_5_checked : candidates edge 4 1283 2741 innerTail205_6 best.bound = innerCandidates205_5 := by
  unfold candidates
  have row : edge 2741 = SearchData.row2741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_5 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_5 best = best := by
  rw [innerTail205_5, scan, if_neg (by decide), innerCandidates205_5_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_6
    (if 3 ≤ innerCandidates205_5.length then search edge 3 [1283, 2741] innerCandidates205_5 4024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_6

def innerTail205_4 : List Nat := 2423 :: innerTail205_5
def innerCandidates205_4 : List Nat := [5021]
set_option maxRecDepth 4096 in
theorem innerCandidates205_4_checked : candidates edge 4 1283 2423 innerTail205_5 best.bound = innerCandidates205_4 := by
  unfold candidates
  have row : edge 2423 = PrimePairCache.smallRow2423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_4 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_4 best = best := by
  rw [innerTail205_4, scan, if_neg (by decide), innerCandidates205_4_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_5
    (if 3 ≤ innerCandidates205_4.length then search edge 3 [1283, 2423] innerCandidates205_4 3706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_5

def innerTail205_3 : List Nat := 2111 :: innerTail205_4
def innerCandidates205_3 : List Nat := [4127]
set_option maxRecDepth 4096 in
theorem innerCandidates205_3_checked : candidates edge 4 1283 2111 innerTail205_4 best.bound = innerCandidates205_3 := by
  unfold candidates
  have row : edge 2111 = PrimePairCache.smallRow2111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_3 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_3 best = best := by
  rw [innerTail205_3, scan, if_neg (by decide), innerCandidates205_3_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_4
    (if 3 ≤ innerCandidates205_3.length then search edge 3 [1283, 2111] innerCandidates205_3 3394 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_4

def innerTail205_2 : List Nat := 1913 :: innerTail205_3
def innerCandidates205_2 : List Nat := [2741]
set_option maxRecDepth 4096 in
theorem innerCandidates205_2_checked : candidates edge 4 1283 1913 innerTail205_3 best.bound = innerCandidates205_2 := by
  unfold candidates
  have row : edge 1913 = PrimePairCache.smallRow1913.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_2 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_2 best = best := by
  rw [innerTail205_2, scan, if_neg (by decide), innerCandidates205_2_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_3
    (if 3 ≤ innerCandidates205_2.length then search edge 3 [1283, 1913] innerCandidates205_2 3196 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_3

def innerTail205_1 : List Nat := 1667 :: innerTail205_2
def innerCandidates205_1 : List Nat := [6287]
set_option maxRecDepth 4096 in
theorem innerCandidates205_1_checked : candidates edge 4 1283 1667 innerTail205_2 best.bound = innerCandidates205_1 := by
  unfold candidates
  have row : edge 1667 = PrimePairCache.smallRow1667.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_1 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_1 best = best := by
  rw [innerTail205_1, scan, if_neg (by decide), innerCandidates205_1_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_2
    (if 3 ≤ innerCandidates205_1.length then search edge 3 [1283, 1667] innerCandidates205_1 2950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_2

def innerTail205_0 : List Nat := 1619 :: innerTail205_1
def innerCandidates205_0 : List Nat := [4127, 7949]
set_option maxRecDepth 4096 in
theorem innerCandidates205_0_checked : candidates edge 4 1283 1619 innerTail205_1 best.bound = innerCandidates205_0 := by
  unfold candidates
  have row : edge 1619 = PrimePairCache.smallRow1619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan205_0 : scan edge (search edge 3) 4 [1283] 1283 innerTail205_0 best = best := by
  rw [innerTail205_0, scan, if_neg (by decide), innerCandidates205_0_checked]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_1
    (if 3 ≤ innerCandidates205_0.length then search edge 3 [1283, 1619] innerCandidates205_0 2902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan205_1

theorem rootBranch205 : search edge 4 [1283] rootCandidates205 1283 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1283] 1283 innerTail205_0 best = best
  exact innerScan205_0

end ProjectEulerSolutions.PrimePairUpper.Search
