import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates382 : List Nat := [3079, 3169, 3631, 3739, 3907, 4423, 4723, 4759, 4933, 5167, 5827, 5857, 5869, 6949, 7603, 7927]
set_option maxRecDepth 4096 in
theorem rootCandidates382_checked : candidates edge 5 0 2659 tail383 best.bound = rootCandidates382 := by
  unfold candidates
  have row : edge 2659 = PrimePairCache.smallRow2659.contains := by funext q; rfl
  rw [row]
  decide

def innerTail382_16 : List Nat := []
theorem innerScan382_16 : scan edge (search edge 3) 4 [2659] 2659 innerTail382_16 best = best := rfl

def innerTail382_15 : List Nat := 7927 :: innerTail382_16
def innerCandidates382_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates382_15_checked : candidates edge 4 2659 7927 innerTail382_16 best.bound = innerCandidates382_15 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan382_15 : scan edge (search edge 3) 4 [2659] 2659 innerTail382_15 best = best := by
  rw [innerTail382_15, scan, if_neg (by decide), innerCandidates382_15_checked]
  change scan edge (search edge 3) 4 [2659] 2659 innerTail382_16
    (if 3 ≤ innerCandidates382_15.length then search edge 3 [2659, 7927] innerCandidates382_15 10586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan382_16

def innerTail382_14 : List Nat := 7603 :: innerTail382_15
def innerCandidates382_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates382_14_checked : candidates edge 4 2659 7603 innerTail382_15 best.bound = innerCandidates382_14 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan382_14 : scan edge (search edge 3) 4 [2659] 2659 innerTail382_14 best = best := by
  rw [innerTail382_14, scan, if_neg (by decide), innerCandidates382_14_checked]
  change scan edge (search edge 3) 4 [2659] 2659 innerTail382_15
    (if 3 ≤ innerCandidates382_14.length then search edge 3 [2659, 7603] innerCandidates382_14 10262 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan382_15

def innerTail382_13 : List Nat := 6949 :: innerTail382_14
def innerCandidates382_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates382_13_checked : candidates edge 4 2659 6949 innerTail382_14 best.bound = innerCandidates382_13 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan382_13 : scan edge (search edge 3) 4 [2659] 2659 innerTail382_13 best = best := by
  rw [innerTail382_13, scan, if_neg (by decide), innerCandidates382_13_checked]
  change scan edge (search edge 3) 4 [2659] 2659 innerTail382_14
    (if 3 ≤ innerCandidates382_13.length then search edge 3 [2659, 6949] innerCandidates382_13 9608 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan382_14

def innerTail382_12 : List Nat := 5869 :: innerTail382_13
def innerCandidates382_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates382_12_checked : candidates edge 4 2659 5869 innerTail382_13 best.bound = innerCandidates382_12 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan382_12 : scan edge (search edge 3) 4 [2659] 2659 innerTail382_12 best = best := by
  rw [innerTail382_12, scan, if_neg (by decide), innerCandidates382_12_checked]
  change scan edge (search edge 3) 4 [2659] 2659 innerTail382_13
    (if 3 ≤ innerCandidates382_12.length then search edge 3 [2659, 5869] innerCandidates382_12 8528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan382_13

def innerTail382_11 : List Nat := 5857 :: innerTail382_12
def innerCandidates382_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates382_11_checked : candidates edge 4 2659 5857 innerTail382_12 best.bound = innerCandidates382_11 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan382_11 : scan edge (search edge 3) 4 [2659] 2659 innerTail382_11 best = best := by
  rw [innerTail382_11, scan, if_neg (by decide), innerCandidates382_11_checked]
  change scan edge (search edge 3) 4 [2659] 2659 innerTail382_12
    (if 3 ≤ innerCandidates382_11.length then search edge 3 [2659, 5857] innerCandidates382_11 8516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan382_12

def innerTail382_10 : List Nat := 5827 :: innerTail382_11
def innerCandidates382_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates382_10_checked : candidates edge 4 2659 5827 innerTail382_11 best.bound = innerCandidates382_10 := by
  unfold candidates
  have row : edge 5827 = SearchData.row5827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan382_10 : scan edge (search edge 3) 4 [2659] 2659 innerTail382_10 best = best := by
  rw [innerTail382_10, scan, if_neg (by decide), innerCandidates382_10_checked]
  change scan edge (search edge 3) 4 [2659] 2659 innerTail382_11
    (if 3 ≤ innerCandidates382_10.length then search edge 3 [2659, 5827] innerCandidates382_10 8486 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan382_11

def innerTail382_9 : List Nat := 5167 :: innerTail382_10
def innerCandidates382_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates382_9_checked : candidates edge 4 2659 5167 innerTail382_10 best.bound = innerCandidates382_9 := by
  unfold candidates
  have row : edge 5167 = SearchData.row5167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan382_9 : scan edge (search edge 3) 4 [2659] 2659 innerTail382_9 best = best := by
  rw [innerTail382_9, scan, if_neg (by decide), innerCandidates382_9_checked]
  change scan edge (search edge 3) 4 [2659] 2659 innerTail382_10
    (if 3 ≤ innerCandidates382_9.length then search edge 3 [2659, 5167] innerCandidates382_9 7826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan382_10

def innerTail382_8 : List Nat := 4933 :: innerTail382_9
def innerCandidates382_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates382_8_checked : candidates edge 4 2659 4933 innerTail382_9 best.bound = innerCandidates382_8 := by
  unfold candidates
  have row : edge 4933 = SearchData.row4933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan382_8 : scan edge (search edge 3) 4 [2659] 2659 innerTail382_8 best = best := by
  rw [innerTail382_8, scan, if_neg (by decide), innerCandidates382_8_checked]
  change scan edge (search edge 3) 4 [2659] 2659 innerTail382_9
    (if 3 ≤ innerCandidates382_8.length then search edge 3 [2659, 4933] innerCandidates382_8 7592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan382_9

def innerTail382_7 : List Nat := 4759 :: innerTail382_8
def innerCandidates382_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates382_7_checked : candidates edge 4 2659 4759 innerTail382_8 best.bound = innerCandidates382_7 := by
  unfold candidates
  have row : edge 4759 = SearchData.row4759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan382_7 : scan edge (search edge 3) 4 [2659] 2659 innerTail382_7 best = best := by
  rw [innerTail382_7, scan, if_neg (by decide), innerCandidates382_7_checked]
  change scan edge (search edge 3) 4 [2659] 2659 innerTail382_8
    (if 3 ≤ innerCandidates382_7.length then search edge 3 [2659, 4759] innerCandidates382_7 7418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan382_8

def innerTail382_6 : List Nat := 4723 :: innerTail382_7
def innerCandidates382_6 : List Nat := [5869]
set_option maxRecDepth 4096 in
theorem innerCandidates382_6_checked : candidates edge 4 2659 4723 innerTail382_7 best.bound = innerCandidates382_6 := by
  unfold candidates
  have row : edge 4723 = SearchData.row4723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan382_6 : scan edge (search edge 3) 4 [2659] 2659 innerTail382_6 best = best := by
  rw [innerTail382_6, scan, if_neg (by decide), innerCandidates382_6_checked]
  change scan edge (search edge 3) 4 [2659] 2659 innerTail382_7
    (if 3 ≤ innerCandidates382_6.length then search edge 3 [2659, 4723] innerCandidates382_6 7382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan382_7

def innerTail382_5 : List Nat := 4423 :: innerTail382_6
def innerCandidates382_5 : List Nat := [5869]
set_option maxRecDepth 4096 in
theorem innerCandidates382_5_checked : candidates edge 4 2659 4423 innerTail382_6 best.bound = innerCandidates382_5 := by
  unfold candidates
  have row : edge 4423 = SearchData.row4423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan382_5 : scan edge (search edge 3) 4 [2659] 2659 innerTail382_5 best = best := by
  rw [innerTail382_5, scan, if_neg (by decide), innerCandidates382_5_checked]
  change scan edge (search edge 3) 4 [2659] 2659 innerTail382_6
    (if 3 ≤ innerCandidates382_5.length then search edge 3 [2659, 4423] innerCandidates382_5 7082 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan382_6

def innerTail382_4 : List Nat := 3907 :: innerTail382_5
def innerCandidates382_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates382_4_checked : candidates edge 4 2659 3907 innerTail382_5 best.bound = innerCandidates382_4 := by
  unfold candidates
  have row : edge 3907 = SearchData.row3907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan382_4 : scan edge (search edge 3) 4 [2659] 2659 innerTail382_4 best = best := by
  rw [innerTail382_4, scan, if_neg (by decide), innerCandidates382_4_checked]
  change scan edge (search edge 3) 4 [2659] 2659 innerTail382_5
    (if 3 ≤ innerCandidates382_4.length then search edge 3 [2659, 3907] innerCandidates382_4 6566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan382_5

def innerTail382_3 : List Nat := 3739 :: innerTail382_4
def innerCandidates382_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates382_3_checked : candidates edge 4 2659 3739 innerTail382_4 best.bound = innerCandidates382_3 := by
  unfold candidates
  have row : edge 3739 = SearchData.row3739.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan382_3 : scan edge (search edge 3) 4 [2659] 2659 innerTail382_3 best = best := by
  rw [innerTail382_3, scan, if_neg (by decide), innerCandidates382_3_checked]
  change scan edge (search edge 3) 4 [2659] 2659 innerTail382_4
    (if 3 ≤ innerCandidates382_3.length then search edge 3 [2659, 3739] innerCandidates382_3 6398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan382_4

def innerTail382_2 : List Nat := 3631 :: innerTail382_3
def innerCandidates382_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates382_2_checked : candidates edge 4 2659 3631 innerTail382_3 best.bound = innerCandidates382_2 := by
  unfold candidates
  have row : edge 3631 = SearchData.row3631.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan382_2 : scan edge (search edge 3) 4 [2659] 2659 innerTail382_2 best = best := by
  rw [innerTail382_2, scan, if_neg (by decide), innerCandidates382_2_checked]
  change scan edge (search edge 3) 4 [2659] 2659 innerTail382_3
    (if 3 ≤ innerCandidates382_2.length then search edge 3 [2659, 3631] innerCandidates382_2 6290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan382_3

def innerTail382_1 : List Nat := 3169 :: innerTail382_2
def innerCandidates382_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates382_1_checked : candidates edge 4 2659 3169 innerTail382_2 best.bound = innerCandidates382_1 := by
  unfold candidates
  have row : edge 3169 = SearchData.row3169.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan382_1 : scan edge (search edge 3) 4 [2659] 2659 innerTail382_1 best = best := by
  rw [innerTail382_1, scan, if_neg (by decide), innerCandidates382_1_checked]
  change scan edge (search edge 3) 4 [2659] 2659 innerTail382_2
    (if 3 ≤ innerCandidates382_1.length then search edge 3 [2659, 3169] innerCandidates382_1 5828 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan382_2

def innerTail382_0 : List Nat := 3079 :: innerTail382_1
def innerCandidates382_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates382_0_checked : candidates edge 4 2659 3079 innerTail382_1 best.bound = innerCandidates382_0 := by
  unfold candidates
  have row : edge 3079 = SearchData.row3079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan382_0 : scan edge (search edge 3) 4 [2659] 2659 innerTail382_0 best = best := by
  rw [innerTail382_0, scan, if_neg (by decide), innerCandidates382_0_checked]
  change scan edge (search edge 3) 4 [2659] 2659 innerTail382_1
    (if 3 ≤ innerCandidates382_0.length then search edge 3 [2659, 3079] innerCandidates382_0 5738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan382_1

theorem rootBranch382 : search edge 4 [2659] rootCandidates382 2659 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2659] 2659 innerTail382_0 best = best
  exact innerScan382_0

def rootCandidates381 : List Nat := [3719, 3923, 4391, 4517, 5099, 5279, 6449, 6833, 7247, 8243]
set_option maxRecDepth 4096 in
theorem rootCandidates381_checked : candidates edge 5 0 2657 tail382 best.bound = rootCandidates381 := by
  unfold candidates
  have row : edge 2657 = PrimePairCache.smallRow2657.contains := by funext q; rfl
  rw [row]
  decide

def innerTail381_10 : List Nat := []
theorem innerScan381_10 : scan edge (search edge 3) 4 [2657] 2657 innerTail381_10 best = best := rfl

def innerTail381_9 : List Nat := 8243 :: innerTail381_10
def innerCandidates381_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates381_9_checked : candidates edge 4 2657 8243 innerTail381_10 best.bound = innerCandidates381_9 := by
  unfold candidates
  have row : edge 8243 = SearchData.row8243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan381_9 : scan edge (search edge 3) 4 [2657] 2657 innerTail381_9 best = best := by
  rw [innerTail381_9, scan, if_neg (by decide), innerCandidates381_9_checked]
  change scan edge (search edge 3) 4 [2657] 2657 innerTail381_10
    (if 3 ≤ innerCandidates381_9.length then search edge 3 [2657, 8243] innerCandidates381_9 10900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan381_10

def innerTail381_8 : List Nat := 7247 :: innerTail381_9
def innerCandidates381_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates381_8_checked : candidates edge 4 2657 7247 innerTail381_9 best.bound = innerCandidates381_8 := by
  unfold candidates
  have row : edge 7247 = SearchData.row7247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan381_8 : scan edge (search edge 3) 4 [2657] 2657 innerTail381_8 best = best := by
  rw [innerTail381_8, scan, if_neg (by decide), innerCandidates381_8_checked]
  change scan edge (search edge 3) 4 [2657] 2657 innerTail381_9
    (if 3 ≤ innerCandidates381_8.length then search edge 3 [2657, 7247] innerCandidates381_8 9904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan381_9

def innerTail381_7 : List Nat := 6833 :: innerTail381_8
def innerCandidates381_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates381_7_checked : candidates edge 4 2657 6833 innerTail381_8 best.bound = innerCandidates381_7 := by
  unfold candidates
  have row : edge 6833 = SearchData.row6833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan381_7 : scan edge (search edge 3) 4 [2657] 2657 innerTail381_7 best = best := by
  rw [innerTail381_7, scan, if_neg (by decide), innerCandidates381_7_checked]
  change scan edge (search edge 3) 4 [2657] 2657 innerTail381_8
    (if 3 ≤ innerCandidates381_7.length then search edge 3 [2657, 6833] innerCandidates381_7 9490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan381_8

def innerTail381_6 : List Nat := 6449 :: innerTail381_7
def innerCandidates381_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates381_6_checked : candidates edge 4 2657 6449 innerTail381_7 best.bound = innerCandidates381_6 := by
  unfold candidates
  have row : edge 6449 = SearchData.row6449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan381_6 : scan edge (search edge 3) 4 [2657] 2657 innerTail381_6 best = best := by
  rw [innerTail381_6, scan, if_neg (by decide), innerCandidates381_6_checked]
  change scan edge (search edge 3) 4 [2657] 2657 innerTail381_7
    (if 3 ≤ innerCandidates381_6.length then search edge 3 [2657, 6449] innerCandidates381_6 9106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan381_7

def innerTail381_5 : List Nat := 5279 :: innerTail381_6
def innerCandidates381_5 : List Nat := [6449]
set_option maxRecDepth 4096 in
theorem innerCandidates381_5_checked : candidates edge 4 2657 5279 innerTail381_6 best.bound = innerCandidates381_5 := by
  unfold candidates
  have row : edge 5279 = SearchData.row5279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan381_5 : scan edge (search edge 3) 4 [2657] 2657 innerTail381_5 best = best := by
  rw [innerTail381_5, scan, if_neg (by decide), innerCandidates381_5_checked]
  change scan edge (search edge 3) 4 [2657] 2657 innerTail381_6
    (if 3 ≤ innerCandidates381_5.length then search edge 3 [2657, 5279] innerCandidates381_5 7936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan381_6

def innerTail381_4 : List Nat := 5099 :: innerTail381_5
def innerCandidates381_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates381_4_checked : candidates edge 4 2657 5099 innerTail381_5 best.bound = innerCandidates381_4 := by
  unfold candidates
  have row : edge 5099 = SearchData.row5099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan381_4 : scan edge (search edge 3) 4 [2657] 2657 innerTail381_4 best = best := by
  rw [innerTail381_4, scan, if_neg (by decide), innerCandidates381_4_checked]
  change scan edge (search edge 3) 4 [2657] 2657 innerTail381_5
    (if 3 ≤ innerCandidates381_4.length then search edge 3 [2657, 5099] innerCandidates381_4 7756 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan381_5

def innerTail381_3 : List Nat := 4517 :: innerTail381_4
def innerCandidates381_3 : List Nat := [5279]
set_option maxRecDepth 4096 in
theorem innerCandidates381_3_checked : candidates edge 4 2657 4517 innerTail381_4 best.bound = innerCandidates381_3 := by
  unfold candidates
  have row : edge 4517 = SearchData.row4517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan381_3 : scan edge (search edge 3) 4 [2657] 2657 innerTail381_3 best = best := by
  rw [innerTail381_3, scan, if_neg (by decide), innerCandidates381_3_checked]
  change scan edge (search edge 3) 4 [2657] 2657 innerTail381_4
    (if 3 ≤ innerCandidates381_3.length then search edge 3 [2657, 4517] innerCandidates381_3 7174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan381_4

def innerTail381_2 : List Nat := 4391 :: innerTail381_3
def innerCandidates381_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates381_2_checked : candidates edge 4 2657 4391 innerTail381_3 best.bound = innerCandidates381_2 := by
  unfold candidates
  have row : edge 4391 = SearchData.row4391.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan381_2 : scan edge (search edge 3) 4 [2657] 2657 innerTail381_2 best = best := by
  rw [innerTail381_2, scan, if_neg (by decide), innerCandidates381_2_checked]
  change scan edge (search edge 3) 4 [2657] 2657 innerTail381_3
    (if 3 ≤ innerCandidates381_2.length then search edge 3 [2657, 4391] innerCandidates381_2 7048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan381_3

def innerTail381_1 : List Nat := 3923 :: innerTail381_2
def innerCandidates381_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates381_1_checked : candidates edge 4 2657 3923 innerTail381_2 best.bound = innerCandidates381_1 := by
  unfold candidates
  have row : edge 3923 = SearchData.row3923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan381_1 : scan edge (search edge 3) 4 [2657] 2657 innerTail381_1 best = best := by
  rw [innerTail381_1, scan, if_neg (by decide), innerCandidates381_1_checked]
  change scan edge (search edge 3) 4 [2657] 2657 innerTail381_2
    (if 3 ≤ innerCandidates381_1.length then search edge 3 [2657, 3923] innerCandidates381_1 6580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan381_2

def innerTail381_0 : List Nat := 3719 :: innerTail381_1
def innerCandidates381_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates381_0_checked : candidates edge 4 2657 3719 innerTail381_1 best.bound = innerCandidates381_0 := by
  unfold candidates
  have row : edge 3719 = SearchData.row3719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan381_0 : scan edge (search edge 3) 4 [2657] 2657 innerTail381_0 best = best := by
  rw [innerTail381_0, scan, if_neg (by decide), innerCandidates381_0_checked]
  change scan edge (search edge 3) 4 [2657] 2657 innerTail381_1
    (if 3 ≤ innerCandidates381_0.length then search edge 3 [2657, 3719] innerCandidates381_0 6376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan381_1

theorem rootBranch381 : search edge 4 [2657] rootCandidates381 2657 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2657] 2657 innerTail381_0 best = best
  exact innerScan381_0

def rootCandidates380 : List Nat := [2851, 3793, 4057, 4663, 5431, 6091, 6271, 6553, 6577, 6673, 7951, 8161]
set_option maxRecDepth 4096 in
theorem rootCandidates380_checked : candidates edge 5 0 2647 tail381 best.bound = rootCandidates380 := by
  unfold candidates
  have row : edge 2647 = PrimePairCache.smallRow2647.contains := by funext q; rfl
  rw [row]
  decide

def innerTail380_12 : List Nat := []
theorem innerScan380_12 : scan edge (search edge 3) 4 [2647] 2647 innerTail380_12 best = best := rfl

def innerTail380_11 : List Nat := 8161 :: innerTail380_12
def innerCandidates380_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates380_11_checked : candidates edge 4 2647 8161 innerTail380_12 best.bound = innerCandidates380_11 := by
  unfold candidates
  have row : edge 8161 = SearchData.row8161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan380_11 : scan edge (search edge 3) 4 [2647] 2647 innerTail380_11 best = best := by
  rw [innerTail380_11, scan, if_neg (by decide), innerCandidates380_11_checked]
  change scan edge (search edge 3) 4 [2647] 2647 innerTail380_12
    (if 3 ≤ innerCandidates380_11.length then search edge 3 [2647, 8161] innerCandidates380_11 10808 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan380_12

def innerTail380_10 : List Nat := 7951 :: innerTail380_11
def innerCandidates380_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates380_10_checked : candidates edge 4 2647 7951 innerTail380_11 best.bound = innerCandidates380_10 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan380_10 : scan edge (search edge 3) 4 [2647] 2647 innerTail380_10 best = best := by
  rw [innerTail380_10, scan, if_neg (by decide), innerCandidates380_10_checked]
  change scan edge (search edge 3) 4 [2647] 2647 innerTail380_11
    (if 3 ≤ innerCandidates380_10.length then search edge 3 [2647, 7951] innerCandidates380_10 10598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan380_11

def innerTail380_9 : List Nat := 6673 :: innerTail380_10
def innerCandidates380_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates380_9_checked : candidates edge 4 2647 6673 innerTail380_10 best.bound = innerCandidates380_9 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan380_9 : scan edge (search edge 3) 4 [2647] 2647 innerTail380_9 best = best := by
  rw [innerTail380_9, scan, if_neg (by decide), innerCandidates380_9_checked]
  change scan edge (search edge 3) 4 [2647] 2647 innerTail380_10
    (if 3 ≤ innerCandidates380_9.length then search edge 3 [2647, 6673] innerCandidates380_9 9320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan380_10

def innerTail380_8 : List Nat := 6577 :: innerTail380_9
def innerCandidates380_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates380_8_checked : candidates edge 4 2647 6577 innerTail380_9 best.bound = innerCandidates380_8 := by
  unfold candidates
  have row : edge 6577 = SearchData.row6577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan380_8 : scan edge (search edge 3) 4 [2647] 2647 innerTail380_8 best = best := by
  rw [innerTail380_8, scan, if_neg (by decide), innerCandidates380_8_checked]
  change scan edge (search edge 3) 4 [2647] 2647 innerTail380_9
    (if 3 ≤ innerCandidates380_8.length then search edge 3 [2647, 6577] innerCandidates380_8 9224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan380_9

def innerTail380_7 : List Nat := 6553 :: innerTail380_8
def innerCandidates380_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates380_7_checked : candidates edge 4 2647 6553 innerTail380_8 best.bound = innerCandidates380_7 := by
  unfold candidates
  have row : edge 6553 = SearchData.row6553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan380_7 : scan edge (search edge 3) 4 [2647] 2647 innerTail380_7 best = best := by
  rw [innerTail380_7, scan, if_neg (by decide), innerCandidates380_7_checked]
  change scan edge (search edge 3) 4 [2647] 2647 innerTail380_8
    (if 3 ≤ innerCandidates380_7.length then search edge 3 [2647, 6553] innerCandidates380_7 9200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan380_8

def innerTail380_6 : List Nat := 6271 :: innerTail380_7
def innerCandidates380_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates380_6_checked : candidates edge 4 2647 6271 innerTail380_7 best.bound = innerCandidates380_6 := by
  unfold candidates
  have row : edge 6271 = SearchData.row6271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan380_6 : scan edge (search edge 3) 4 [2647] 2647 innerTail380_6 best = best := by
  rw [innerTail380_6, scan, if_neg (by decide), innerCandidates380_6_checked]
  change scan edge (search edge 3) 4 [2647] 2647 innerTail380_7
    (if 3 ≤ innerCandidates380_6.length then search edge 3 [2647, 6271] innerCandidates380_6 8918 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan380_7

def innerTail380_5 : List Nat := 6091 :: innerTail380_6
def innerCandidates380_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates380_5_checked : candidates edge 4 2647 6091 innerTail380_6 best.bound = innerCandidates380_5 := by
  unfold candidates
  have row : edge 6091 = SearchData.row6091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan380_5 : scan edge (search edge 3) 4 [2647] 2647 innerTail380_5 best = best := by
  rw [innerTail380_5, scan, if_neg (by decide), innerCandidates380_5_checked]
  change scan edge (search edge 3) 4 [2647] 2647 innerTail380_6
    (if 3 ≤ innerCandidates380_5.length then search edge 3 [2647, 6091] innerCandidates380_5 8738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan380_6

def innerTail380_4 : List Nat := 5431 :: innerTail380_5
def innerCandidates380_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates380_4_checked : candidates edge 4 2647 5431 innerTail380_5 best.bound = innerCandidates380_4 := by
  unfold candidates
  have row : edge 5431 = SearchData.row5431.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan380_4 : scan edge (search edge 3) 4 [2647] 2647 innerTail380_4 best = best := by
  rw [innerTail380_4, scan, if_neg (by decide), innerCandidates380_4_checked]
  change scan edge (search edge 3) 4 [2647] 2647 innerTail380_5
    (if 3 ≤ innerCandidates380_4.length then search edge 3 [2647, 5431] innerCandidates380_4 8078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan380_5

def innerTail380_3 : List Nat := 4663 :: innerTail380_4
def innerCandidates380_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates380_3_checked : candidates edge 4 2647 4663 innerTail380_4 best.bound = innerCandidates380_3 := by
  unfold candidates
  have row : edge 4663 = SearchData.row4663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan380_3 : scan edge (search edge 3) 4 [2647] 2647 innerTail380_3 best = best := by
  rw [innerTail380_3, scan, if_neg (by decide), innerCandidates380_3_checked]
  change scan edge (search edge 3) 4 [2647] 2647 innerTail380_4
    (if 3 ≤ innerCandidates380_3.length then search edge 3 [2647, 4663] innerCandidates380_3 7310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan380_4

def innerTail380_2 : List Nat := 4057 :: innerTail380_3
def innerCandidates380_2 : List Nat := [4663]
set_option maxRecDepth 4096 in
theorem innerCandidates380_2_checked : candidates edge 4 2647 4057 innerTail380_3 best.bound = innerCandidates380_2 := by
  unfold candidates
  have row : edge 4057 = SearchData.row4057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan380_2 : scan edge (search edge 3) 4 [2647] 2647 innerTail380_2 best = best := by
  rw [innerTail380_2, scan, if_neg (by decide), innerCandidates380_2_checked]
  change scan edge (search edge 3) 4 [2647] 2647 innerTail380_3
    (if 3 ≤ innerCandidates380_2.length then search edge 3 [2647, 4057] innerCandidates380_2 6704 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan380_3

def innerTail380_1 : List Nat := 3793 :: innerTail380_2
def innerCandidates380_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates380_1_checked : candidates edge 4 2647 3793 innerTail380_2 best.bound = innerCandidates380_1 := by
  unfold candidates
  have row : edge 3793 = SearchData.row3793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan380_1 : scan edge (search edge 3) 4 [2647] 2647 innerTail380_1 best = best := by
  rw [innerTail380_1, scan, if_neg (by decide), innerCandidates380_1_checked]
  change scan edge (search edge 3) 4 [2647] 2647 innerTail380_2
    (if 3 ≤ innerCandidates380_1.length then search edge 3 [2647, 3793] innerCandidates380_1 6440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan380_2

def innerTail380_0 : List Nat := 2851 :: innerTail380_1
def innerCandidates380_0 : List Nat := [6091]
set_option maxRecDepth 4096 in
theorem innerCandidates380_0_checked : candidates edge 4 2647 2851 innerTail380_1 best.bound = innerCandidates380_0 := by
  unfold candidates
  have row : edge 2851 = SearchData.row2851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan380_0 : scan edge (search edge 3) 4 [2647] 2647 innerTail380_0 best = best := by
  rw [innerTail380_0, scan, if_neg (by decide), innerCandidates380_0_checked]
  change scan edge (search edge 3) 4 [2647] 2647 innerTail380_1
    (if 3 ≤ innerCandidates380_0.length then search edge 3 [2647, 2851] innerCandidates380_0 5498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan380_1

theorem rootBranch380 : search edge 4 [2647] rootCandidates380 2647 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2647] 2647 innerTail380_0 best = best
  exact innerScan380_0

def rootCandidates379 : List Nat := [2699, 3929, 4337, 5051, 5147, 5693, 5807, 5867, 6581, 7079, 7193, 7517, 7547, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates379_checked : candidates edge 5 0 2633 tail380 best.bound = rootCandidates379 := by
  unfold candidates
  have row : edge 2633 = PrimePairCache.smallRow2633.contains := by funext q; rfl
  rw [row]
  decide

def innerTail379_14 : List Nat := []
theorem innerScan379_14 : scan edge (search edge 3) 4 [2633] 2633 innerTail379_14 best = best := rfl

def innerTail379_13 : List Nat := 8369 :: innerTail379_14
def innerCandidates379_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates379_13_checked : candidates edge 4 2633 8369 innerTail379_14 best.bound = innerCandidates379_13 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan379_13 : scan edge (search edge 3) 4 [2633] 2633 innerTail379_13 best = best := by
  rw [innerTail379_13, scan, if_neg (by decide), innerCandidates379_13_checked]
  change scan edge (search edge 3) 4 [2633] 2633 innerTail379_14
    (if 3 ≤ innerCandidates379_13.length then search edge 3 [2633, 8369] innerCandidates379_13 11002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan379_14

def innerTail379_12 : List Nat := 7547 :: innerTail379_13
def innerCandidates379_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates379_12_checked : candidates edge 4 2633 7547 innerTail379_13 best.bound = innerCandidates379_12 := by
  unfold candidates
  have row : edge 7547 = SearchData.row7547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan379_12 : scan edge (search edge 3) 4 [2633] 2633 innerTail379_12 best = best := by
  rw [innerTail379_12, scan, if_neg (by decide), innerCandidates379_12_checked]
  change scan edge (search edge 3) 4 [2633] 2633 innerTail379_13
    (if 3 ≤ innerCandidates379_12.length then search edge 3 [2633, 7547] innerCandidates379_12 10180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan379_13

def innerTail379_11 : List Nat := 7517 :: innerTail379_12
def innerCandidates379_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates379_11_checked : candidates edge 4 2633 7517 innerTail379_12 best.bound = innerCandidates379_11 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan379_11 : scan edge (search edge 3) 4 [2633] 2633 innerTail379_11 best = best := by
  rw [innerTail379_11, scan, if_neg (by decide), innerCandidates379_11_checked]
  change scan edge (search edge 3) 4 [2633] 2633 innerTail379_12
    (if 3 ≤ innerCandidates379_11.length then search edge 3 [2633, 7517] innerCandidates379_11 10150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan379_12

def innerTail379_10 : List Nat := 7193 :: innerTail379_11
def innerCandidates379_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates379_10_checked : candidates edge 4 2633 7193 innerTail379_11 best.bound = innerCandidates379_10 := by
  unfold candidates
  have row : edge 7193 = SearchData.row7193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan379_10 : scan edge (search edge 3) 4 [2633] 2633 innerTail379_10 best = best := by
  rw [innerTail379_10, scan, if_neg (by decide), innerCandidates379_10_checked]
  change scan edge (search edge 3) 4 [2633] 2633 innerTail379_11
    (if 3 ≤ innerCandidates379_10.length then search edge 3 [2633, 7193] innerCandidates379_10 9826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan379_11

def innerTail379_9 : List Nat := 7079 :: innerTail379_10
def innerCandidates379_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates379_9_checked : candidates edge 4 2633 7079 innerTail379_10 best.bound = innerCandidates379_9 := by
  unfold candidates
  have row : edge 7079 = SearchData.row7079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan379_9 : scan edge (search edge 3) 4 [2633] 2633 innerTail379_9 best = best := by
  rw [innerTail379_9, scan, if_neg (by decide), innerCandidates379_9_checked]
  change scan edge (search edge 3) 4 [2633] 2633 innerTail379_10
    (if 3 ≤ innerCandidates379_9.length then search edge 3 [2633, 7079] innerCandidates379_9 9712 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan379_10

def innerTail379_8 : List Nat := 6581 :: innerTail379_9
def innerCandidates379_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates379_8_checked : candidates edge 4 2633 6581 innerTail379_9 best.bound = innerCandidates379_8 := by
  unfold candidates
  have row : edge 6581 = SearchData.row6581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan379_8 : scan edge (search edge 3) 4 [2633] 2633 innerTail379_8 best = best := by
  rw [innerTail379_8, scan, if_neg (by decide), innerCandidates379_8_checked]
  change scan edge (search edge 3) 4 [2633] 2633 innerTail379_9
    (if 3 ≤ innerCandidates379_8.length then search edge 3 [2633, 6581] innerCandidates379_8 9214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan379_9

def innerTail379_7 : List Nat := 5867 :: innerTail379_8
def innerCandidates379_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates379_7_checked : candidates edge 4 2633 5867 innerTail379_8 best.bound = innerCandidates379_7 := by
  unfold candidates
  have row : edge 5867 = SearchData.row5867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan379_7 : scan edge (search edge 3) 4 [2633] 2633 innerTail379_7 best = best := by
  rw [innerTail379_7, scan, if_neg (by decide), innerCandidates379_7_checked]
  change scan edge (search edge 3) 4 [2633] 2633 innerTail379_8
    (if 3 ≤ innerCandidates379_7.length then search edge 3 [2633, 5867] innerCandidates379_7 8500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan379_8

def innerTail379_6 : List Nat := 5807 :: innerTail379_7
def innerCandidates379_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates379_6_checked : candidates edge 4 2633 5807 innerTail379_7 best.bound = innerCandidates379_6 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan379_6 : scan edge (search edge 3) 4 [2633] 2633 innerTail379_6 best = best := by
  rw [innerTail379_6, scan, if_neg (by decide), innerCandidates379_6_checked]
  change scan edge (search edge 3) 4 [2633] 2633 innerTail379_7
    (if 3 ≤ innerCandidates379_6.length then search edge 3 [2633, 5807] innerCandidates379_6 8440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan379_7

def innerTail379_5 : List Nat := 5693 :: innerTail379_6
def innerCandidates379_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates379_5_checked : candidates edge 4 2633 5693 innerTail379_6 best.bound = innerCandidates379_5 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan379_5 : scan edge (search edge 3) 4 [2633] 2633 innerTail379_5 best = best := by
  rw [innerTail379_5, scan, if_neg (by decide), innerCandidates379_5_checked]
  change scan edge (search edge 3) 4 [2633] 2633 innerTail379_6
    (if 3 ≤ innerCandidates379_5.length then search edge 3 [2633, 5693] innerCandidates379_5 8326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan379_6

def innerTail379_4 : List Nat := 5147 :: innerTail379_5
def innerCandidates379_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates379_4_checked : candidates edge 4 2633 5147 innerTail379_5 best.bound = innerCandidates379_4 := by
  unfold candidates
  have row : edge 5147 = SearchData.row5147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan379_4 : scan edge (search edge 3) 4 [2633] 2633 innerTail379_4 best = best := by
  rw [innerTail379_4, scan, if_neg (by decide), innerCandidates379_4_checked]
  change scan edge (search edge 3) 4 [2633] 2633 innerTail379_5
    (if 3 ≤ innerCandidates379_4.length then search edge 3 [2633, 5147] innerCandidates379_4 7780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan379_5

def innerTail379_3 : List Nat := 5051 :: innerTail379_4
def innerCandidates379_3 : List Nat := [5867]
set_option maxRecDepth 4096 in
theorem innerCandidates379_3_checked : candidates edge 4 2633 5051 innerTail379_4 best.bound = innerCandidates379_3 := by
  unfold candidates
  have row : edge 5051 = SearchData.row5051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan379_3 : scan edge (search edge 3) 4 [2633] 2633 innerTail379_3 best = best := by
  rw [innerTail379_3, scan, if_neg (by decide), innerCandidates379_3_checked]
  change scan edge (search edge 3) 4 [2633] 2633 innerTail379_4
    (if 3 ≤ innerCandidates379_3.length then search edge 3 [2633, 5051] innerCandidates379_3 7684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan379_4

def innerTail379_2 : List Nat := 4337 :: innerTail379_3
def innerCandidates379_2 : List Nat := [7547]
set_option maxRecDepth 4096 in
theorem innerCandidates379_2_checked : candidates edge 4 2633 4337 innerTail379_3 best.bound = innerCandidates379_2 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan379_2 : scan edge (search edge 3) 4 [2633] 2633 innerTail379_2 best = best := by
  rw [innerTail379_2, scan, if_neg (by decide), innerCandidates379_2_checked]
  change scan edge (search edge 3) 4 [2633] 2633 innerTail379_3
    (if 3 ≤ innerCandidates379_2.length then search edge 3 [2633, 4337] innerCandidates379_2 6970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan379_3

def innerTail379_1 : List Nat := 3929 :: innerTail379_2
def innerCandidates379_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates379_1_checked : candidates edge 4 2633 3929 innerTail379_2 best.bound = innerCandidates379_1 := by
  unfold candidates
  have row : edge 3929 = SearchData.row3929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan379_1 : scan edge (search edge 3) 4 [2633] 2633 innerTail379_1 best = best := by
  rw [innerTail379_1, scan, if_neg (by decide), innerCandidates379_1_checked]
  change scan edge (search edge 3) 4 [2633] 2633 innerTail379_2
    (if 3 ≤ innerCandidates379_1.length then search edge 3 [2633, 3929] innerCandidates379_1 6562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan379_2

def innerTail379_0 : List Nat := 2699 :: innerTail379_1
def innerCandidates379_0 : List Nat := [4337]
set_option maxRecDepth 4096 in
theorem innerCandidates379_0_checked : candidates edge 4 2633 2699 innerTail379_1 best.bound = innerCandidates379_0 := by
  unfold candidates
  have row : edge 2699 = SearchData.row2699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan379_0 : scan edge (search edge 3) 4 [2633] 2633 innerTail379_0 best = best := by
  rw [innerTail379_0, scan, if_neg (by decide), innerCandidates379_0_checked]
  change scan edge (search edge 3) 4 [2633] 2633 innerTail379_1
    (if 3 ≤ innerCandidates379_0.length then search edge 3 [2633, 2699] innerCandidates379_0 5332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan379_1

theorem rootBranch379 : search edge 4 [2633] rootCandidates379 2633 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2633] 2633 innerTail379_0 best = best
  exact innerScan379_0

end ProjectEulerSolutions.PrimePairUpper.Search
