import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates378 : List Nat := [2789, 3623, 4583, 4973, 5477, 6053, 6101, 6263, 6317, 6521, 6977, 7151, 7817, 8363]
set_option maxRecDepth 4096 in
theorem rootCandidates378_checked : candidates edge 5 0 2621 tail379 best.bound = rootCandidates378 := by
  unfold candidates
  have row : edge 2621 = PrimePairCache.smallRow2621.contains := by funext q; rfl
  rw [row]
  decide

def innerTail378_14 : List Nat := []
theorem innerScan378_14 : scan edge (search edge 3) 4 [2621] 2621 innerTail378_14 best = best := rfl

def innerTail378_13 : List Nat := 8363 :: innerTail378_14
def innerCandidates378_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates378_13_checked : candidates edge 4 2621 8363 innerTail378_14 best.bound = innerCandidates378_13 := by
  unfold candidates
  have row : edge 8363 = SearchData.row8363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan378_13 : scan edge (search edge 3) 4 [2621] 2621 innerTail378_13 best = best := by
  rw [innerTail378_13, scan, if_neg (by decide), innerCandidates378_13_checked]
  change scan edge (search edge 3) 4 [2621] 2621 innerTail378_14
    (if 3 ≤ innerCandidates378_13.length then search edge 3 [2621, 8363] innerCandidates378_13 10984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan378_14

def innerTail378_12 : List Nat := 7817 :: innerTail378_13
def innerCandidates378_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates378_12_checked : candidates edge 4 2621 7817 innerTail378_13 best.bound = innerCandidates378_12 := by
  unfold candidates
  have row : edge 7817 = SearchData.row7817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan378_12 : scan edge (search edge 3) 4 [2621] 2621 innerTail378_12 best = best := by
  rw [innerTail378_12, scan, if_neg (by decide), innerCandidates378_12_checked]
  change scan edge (search edge 3) 4 [2621] 2621 innerTail378_13
    (if 3 ≤ innerCandidates378_12.length then search edge 3 [2621, 7817] innerCandidates378_12 10438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan378_13

def innerTail378_11 : List Nat := 7151 :: innerTail378_12
def innerCandidates378_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates378_11_checked : candidates edge 4 2621 7151 innerTail378_12 best.bound = innerCandidates378_11 := by
  unfold candidates
  have row : edge 7151 = SearchData.row7151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan378_11 : scan edge (search edge 3) 4 [2621] 2621 innerTail378_11 best = best := by
  rw [innerTail378_11, scan, if_neg (by decide), innerCandidates378_11_checked]
  change scan edge (search edge 3) 4 [2621] 2621 innerTail378_12
    (if 3 ≤ innerCandidates378_11.length then search edge 3 [2621, 7151] innerCandidates378_11 9772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan378_12

def innerTail378_10 : List Nat := 6977 :: innerTail378_11
def innerCandidates378_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates378_10_checked : candidates edge 4 2621 6977 innerTail378_11 best.bound = innerCandidates378_10 := by
  unfold candidates
  have row : edge 6977 = SearchData.row6977.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan378_10 : scan edge (search edge 3) 4 [2621] 2621 innerTail378_10 best = best := by
  rw [innerTail378_10, scan, if_neg (by decide), innerCandidates378_10_checked]
  change scan edge (search edge 3) 4 [2621] 2621 innerTail378_11
    (if 3 ≤ innerCandidates378_10.length then search edge 3 [2621, 6977] innerCandidates378_10 9598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan378_11

def innerTail378_9 : List Nat := 6521 :: innerTail378_10
def innerCandidates378_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates378_9_checked : candidates edge 4 2621 6521 innerTail378_10 best.bound = innerCandidates378_9 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan378_9 : scan edge (search edge 3) 4 [2621] 2621 innerTail378_9 best = best := by
  rw [innerTail378_9, scan, if_neg (by decide), innerCandidates378_9_checked]
  change scan edge (search edge 3) 4 [2621] 2621 innerTail378_10
    (if 3 ≤ innerCandidates378_9.length then search edge 3 [2621, 6521] innerCandidates378_9 9142 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan378_10

def innerTail378_8 : List Nat := 6317 :: innerTail378_9
def innerCandidates378_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates378_8_checked : candidates edge 4 2621 6317 innerTail378_9 best.bound = innerCandidates378_8 := by
  unfold candidates
  have row : edge 6317 = SearchData.row6317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan378_8 : scan edge (search edge 3) 4 [2621] 2621 innerTail378_8 best = best := by
  rw [innerTail378_8, scan, if_neg (by decide), innerCandidates378_8_checked]
  change scan edge (search edge 3) 4 [2621] 2621 innerTail378_9
    (if 3 ≤ innerCandidates378_8.length then search edge 3 [2621, 6317] innerCandidates378_8 8938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan378_9

def innerTail378_7 : List Nat := 6263 :: innerTail378_8
def innerCandidates378_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates378_7_checked : candidates edge 4 2621 6263 innerTail378_8 best.bound = innerCandidates378_7 := by
  unfold candidates
  have row : edge 6263 = SearchData.row6263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan378_7 : scan edge (search edge 3) 4 [2621] 2621 innerTail378_7 best = best := by
  rw [innerTail378_7, scan, if_neg (by decide), innerCandidates378_7_checked]
  change scan edge (search edge 3) 4 [2621] 2621 innerTail378_8
    (if 3 ≤ innerCandidates378_7.length then search edge 3 [2621, 6263] innerCandidates378_7 8884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan378_8

def innerTail378_6 : List Nat := 6101 :: innerTail378_7
def innerCandidates378_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates378_6_checked : candidates edge 4 2621 6101 innerTail378_7 best.bound = innerCandidates378_6 := by
  unfold candidates
  have row : edge 6101 = SearchData.row6101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan378_6 : scan edge (search edge 3) 4 [2621] 2621 innerTail378_6 best = best := by
  rw [innerTail378_6, scan, if_neg (by decide), innerCandidates378_6_checked]
  change scan edge (search edge 3) 4 [2621] 2621 innerTail378_7
    (if 3 ≤ innerCandidates378_6.length then search edge 3 [2621, 6101] innerCandidates378_6 8722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan378_7

def innerTail378_5 : List Nat := 6053 :: innerTail378_6
def innerCandidates378_5 : List Nat := [6977]
set_option maxRecDepth 4096 in
theorem innerCandidates378_5_checked : candidates edge 4 2621 6053 innerTail378_6 best.bound = innerCandidates378_5 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan378_5 : scan edge (search edge 3) 4 [2621] 2621 innerTail378_5 best = best := by
  rw [innerTail378_5, scan, if_neg (by decide), innerCandidates378_5_checked]
  change scan edge (search edge 3) 4 [2621] 2621 innerTail378_6
    (if 3 ≤ innerCandidates378_5.length then search edge 3 [2621, 6053] innerCandidates378_5 8674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan378_6

def innerTail378_4 : List Nat := 5477 :: innerTail378_5
def innerCandidates378_4 : List Nat := [6053]
set_option maxRecDepth 4096 in
theorem innerCandidates378_4_checked : candidates edge 4 2621 5477 innerTail378_5 best.bound = innerCandidates378_4 := by
  unfold candidates
  have row : edge 5477 = SearchData.row5477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan378_4 : scan edge (search edge 3) 4 [2621] 2621 innerTail378_4 best = best := by
  rw [innerTail378_4, scan, if_neg (by decide), innerCandidates378_4_checked]
  change scan edge (search edge 3) 4 [2621] 2621 innerTail378_5
    (if 3 ≤ innerCandidates378_4.length then search edge 3 [2621, 5477] innerCandidates378_4 8098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan378_5

def innerTail378_3 : List Nat := 4973 :: innerTail378_4
def innerCandidates378_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates378_3_checked : candidates edge 4 2621 4973 innerTail378_4 best.bound = innerCandidates378_3 := by
  unfold candidates
  have row : edge 4973 = SearchData.row4973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan378_3 : scan edge (search edge 3) 4 [2621] 2621 innerTail378_3 best = best := by
  rw [innerTail378_3, scan, if_neg (by decide), innerCandidates378_3_checked]
  change scan edge (search edge 3) 4 [2621] 2621 innerTail378_4
    (if 3 ≤ innerCandidates378_3.length then search edge 3 [2621, 4973] innerCandidates378_3 7594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan378_4

def innerTail378_2 : List Nat := 4583 :: innerTail378_3
def innerCandidates378_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates378_2_checked : candidates edge 4 2621 4583 innerTail378_3 best.bound = innerCandidates378_2 := by
  unfold candidates
  have row : edge 4583 = SearchData.row4583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan378_2 : scan edge (search edge 3) 4 [2621] 2621 innerTail378_2 best = best := by
  rw [innerTail378_2, scan, if_neg (by decide), innerCandidates378_2_checked]
  change scan edge (search edge 3) 4 [2621] 2621 innerTail378_3
    (if 3 ≤ innerCandidates378_2.length then search edge 3 [2621, 4583] innerCandidates378_2 7204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan378_3

def innerTail378_1 : List Nat := 3623 :: innerTail378_2
def innerCandidates378_1 : List Nat := [6521]
set_option maxRecDepth 4096 in
theorem innerCandidates378_1_checked : candidates edge 4 2621 3623 innerTail378_2 best.bound = innerCandidates378_1 := by
  unfold candidates
  have row : edge 3623 = SearchData.row3623.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan378_1 : scan edge (search edge 3) 4 [2621] 2621 innerTail378_1 best = best := by
  rw [innerTail378_1, scan, if_neg (by decide), innerCandidates378_1_checked]
  change scan edge (search edge 3) 4 [2621] 2621 innerTail378_2
    (if 3 ≤ innerCandidates378_1.length then search edge 3 [2621, 3623] innerCandidates378_1 6244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan378_2

def innerTail378_0 : List Nat := 2789 :: innerTail378_1
def innerCandidates378_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates378_0_checked : candidates edge 4 2621 2789 innerTail378_1 best.bound = innerCandidates378_0 := by
  unfold candidates
  have row : edge 2789 = SearchData.row2789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan378_0 : scan edge (search edge 3) 4 [2621] 2621 innerTail378_0 best = best := by
  rw [innerTail378_0, scan, if_neg (by decide), innerCandidates378_0_checked]
  change scan edge (search edge 3) 4 [2621] 2621 innerTail378_1
    (if 3 ≤ innerCandidates378_0.length then search edge 3 [2621, 2789] innerCandidates378_0 5410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan378_1

theorem rootBranch378 : search edge 4 [2621] rootCandidates378 2621 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2621] 2621 innerTail378_0 best = best
  exact innerScan378_0

def rootCandidates377 : List Nat := [2803, 2833, 2857, 3181, 3571, 4273, 4567, 4801, 5689, 5839, 7213, 7507, 7741, 8161]
set_option maxRecDepth 4096 in
theorem rootCandidates377_checked : candidates edge 5 0 2617 tail378 best.bound = rootCandidates377 := by
  unfold candidates
  have row : edge 2617 = PrimePairCache.smallRow2617.contains := by funext q; rfl
  rw [row]
  decide

def innerTail377_14 : List Nat := []
theorem innerScan377_14 : scan edge (search edge 3) 4 [2617] 2617 innerTail377_14 best = best := rfl

def innerTail377_13 : List Nat := 8161 :: innerTail377_14
def innerCandidates377_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates377_13_checked : candidates edge 4 2617 8161 innerTail377_14 best.bound = innerCandidates377_13 := by
  unfold candidates
  have row : edge 8161 = SearchData.row8161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan377_13 : scan edge (search edge 3) 4 [2617] 2617 innerTail377_13 best = best := by
  rw [innerTail377_13, scan, if_neg (by decide), innerCandidates377_13_checked]
  change scan edge (search edge 3) 4 [2617] 2617 innerTail377_14
    (if 3 ≤ innerCandidates377_13.length then search edge 3 [2617, 8161] innerCandidates377_13 10778 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan377_14

def innerTail377_12 : List Nat := 7741 :: innerTail377_13
def innerCandidates377_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates377_12_checked : candidates edge 4 2617 7741 innerTail377_13 best.bound = innerCandidates377_12 := by
  unfold candidates
  have row : edge 7741 = SearchData.row7741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan377_12 : scan edge (search edge 3) 4 [2617] 2617 innerTail377_12 best = best := by
  rw [innerTail377_12, scan, if_neg (by decide), innerCandidates377_12_checked]
  change scan edge (search edge 3) 4 [2617] 2617 innerTail377_13
    (if 3 ≤ innerCandidates377_12.length then search edge 3 [2617, 7741] innerCandidates377_12 10358 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan377_13

def innerTail377_11 : List Nat := 7507 :: innerTail377_12
def innerCandidates377_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates377_11_checked : candidates edge 4 2617 7507 innerTail377_12 best.bound = innerCandidates377_11 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan377_11 : scan edge (search edge 3) 4 [2617] 2617 innerTail377_11 best = best := by
  rw [innerTail377_11, scan, if_neg (by decide), innerCandidates377_11_checked]
  change scan edge (search edge 3) 4 [2617] 2617 innerTail377_12
    (if 3 ≤ innerCandidates377_11.length then search edge 3 [2617, 7507] innerCandidates377_11 10124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan377_12

def innerTail377_10 : List Nat := 7213 :: innerTail377_11
def innerCandidates377_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates377_10_checked : candidates edge 4 2617 7213 innerTail377_11 best.bound = innerCandidates377_10 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan377_10 : scan edge (search edge 3) 4 [2617] 2617 innerTail377_10 best = best := by
  rw [innerTail377_10, scan, if_neg (by decide), innerCandidates377_10_checked]
  change scan edge (search edge 3) 4 [2617] 2617 innerTail377_11
    (if 3 ≤ innerCandidates377_10.length then search edge 3 [2617, 7213] innerCandidates377_10 9830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan377_11

def innerTail377_9 : List Nat := 5839 :: innerTail377_10
def innerCandidates377_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates377_9_checked : candidates edge 4 2617 5839 innerTail377_10 best.bound = innerCandidates377_9 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan377_9 : scan edge (search edge 3) 4 [2617] 2617 innerTail377_9 best = best := by
  rw [innerTail377_9, scan, if_neg (by decide), innerCandidates377_9_checked]
  change scan edge (search edge 3) 4 [2617] 2617 innerTail377_10
    (if 3 ≤ innerCandidates377_9.length then search edge 3 [2617, 5839] innerCandidates377_9 8456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan377_10

def innerTail377_8 : List Nat := 5689 :: innerTail377_9
def innerCandidates377_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates377_8_checked : candidates edge 4 2617 5689 innerTail377_9 best.bound = innerCandidates377_8 := by
  unfold candidates
  have row : edge 5689 = SearchData.row5689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan377_8 : scan edge (search edge 3) 4 [2617] 2617 innerTail377_8 best = best := by
  rw [innerTail377_8, scan, if_neg (by decide), innerCandidates377_8_checked]
  change scan edge (search edge 3) 4 [2617] 2617 innerTail377_9
    (if 3 ≤ innerCandidates377_8.length then search edge 3 [2617, 5689] innerCandidates377_8 8306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan377_9

def innerTail377_7 : List Nat := 4801 :: innerTail377_8
def innerCandidates377_7 : List Nat := [5839]
set_option maxRecDepth 4096 in
theorem innerCandidates377_7_checked : candidates edge 4 2617 4801 innerTail377_8 best.bound = innerCandidates377_7 := by
  unfold candidates
  have row : edge 4801 = SearchData.row4801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan377_7 : scan edge (search edge 3) 4 [2617] 2617 innerTail377_7 best = best := by
  rw [innerTail377_7, scan, if_neg (by decide), innerCandidates377_7_checked]
  change scan edge (search edge 3) 4 [2617] 2617 innerTail377_8
    (if 3 ≤ innerCandidates377_7.length then search edge 3 [2617, 4801] innerCandidates377_7 7418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan377_8

def innerTail377_6 : List Nat := 4567 :: innerTail377_7
def innerCandidates377_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates377_6_checked : candidates edge 4 2617 4567 innerTail377_7 best.bound = innerCandidates377_6 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan377_6 : scan edge (search edge 3) 4 [2617] 2617 innerTail377_6 best = best := by
  rw [innerTail377_6, scan, if_neg (by decide), innerCandidates377_6_checked]
  change scan edge (search edge 3) 4 [2617] 2617 innerTail377_7
    (if 3 ≤ innerCandidates377_6.length then search edge 3 [2617, 4567] innerCandidates377_6 7184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan377_7

def innerTail377_5 : List Nat := 4273 :: innerTail377_6
def innerCandidates377_5 : List Nat := [4567]
set_option maxRecDepth 4096 in
theorem innerCandidates377_5_checked : candidates edge 4 2617 4273 innerTail377_6 best.bound = innerCandidates377_5 := by
  unfold candidates
  have row : edge 4273 = SearchData.row4273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan377_5 : scan edge (search edge 3) 4 [2617] 2617 innerTail377_5 best = best := by
  rw [innerTail377_5, scan, if_neg (by decide), innerCandidates377_5_checked]
  change scan edge (search edge 3) 4 [2617] 2617 innerTail377_6
    (if 3 ≤ innerCandidates377_5.length then search edge 3 [2617, 4273] innerCandidates377_5 6890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan377_6

def innerTail377_4 : List Nat := 3571 :: innerTail377_5
def innerCandidates377_4 : List Nat := [4801]
set_option maxRecDepth 4096 in
theorem innerCandidates377_4_checked : candidates edge 4 2617 3571 innerTail377_5 best.bound = innerCandidates377_4 := by
  unfold candidates
  have row : edge 3571 = SearchData.row3571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan377_4 : scan edge (search edge 3) 4 [2617] 2617 innerTail377_4 best = best := by
  rw [innerTail377_4, scan, if_neg (by decide), innerCandidates377_4_checked]
  change scan edge (search edge 3) 4 [2617] 2617 innerTail377_5
    (if 3 ≤ innerCandidates377_4.length then search edge 3 [2617, 3571] innerCandidates377_4 6188 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan377_5

def innerTail377_3 : List Nat := 3181 :: innerTail377_4
def innerCandidates377_3 : List Nat := [4273, 7507]
set_option maxRecDepth 4096 in
theorem innerCandidates377_3_checked : candidates edge 4 2617 3181 innerTail377_4 best.bound = innerCandidates377_3 := by
  unfold candidates
  have row : edge 3181 = SearchData.row3181.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan377_3 : scan edge (search edge 3) 4 [2617] 2617 innerTail377_3 best = best := by
  rw [innerTail377_3, scan, if_neg (by decide), innerCandidates377_3_checked]
  change scan edge (search edge 3) 4 [2617] 2617 innerTail377_4
    (if 3 ≤ innerCandidates377_3.length then search edge 3 [2617, 3181] innerCandidates377_3 5798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan377_4

def innerTail377_2 : List Nat := 2857 :: innerTail377_3
def innerCandidates377_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates377_2_checked : candidates edge 4 2617 2857 innerTail377_3 best.bound = innerCandidates377_2 := by
  unfold candidates
  have row : edge 2857 = SearchData.row2857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan377_2 : scan edge (search edge 3) 4 [2617] 2617 innerTail377_2 best = best := by
  rw [innerTail377_2, scan, if_neg (by decide), innerCandidates377_2_checked]
  change scan edge (search edge 3) 4 [2617] 2617 innerTail377_3
    (if 3 ≤ innerCandidates377_2.length then search edge 3 [2617, 2857] innerCandidates377_2 5474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan377_3

def innerTail377_1 : List Nat := 2833 :: innerTail377_2
def innerCandidates377_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates377_1_checked : candidates edge 4 2617 2833 innerTail377_2 best.bound = innerCandidates377_1 := by
  unfold candidates
  have row : edge 2833 = SearchData.row2833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan377_1 : scan edge (search edge 3) 4 [2617] 2617 innerTail377_1 best = best := by
  rw [innerTail377_1, scan, if_neg (by decide), innerCandidates377_1_checked]
  change scan edge (search edge 3) 4 [2617] 2617 innerTail377_2
    (if 3 ≤ innerCandidates377_1.length then search edge 3 [2617, 2833] innerCandidates377_1 5450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan377_2

def innerTail377_0 : List Nat := 2803 :: innerTail377_1
def innerCandidates377_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates377_0_checked : candidates edge 4 2617 2803 innerTail377_1 best.bound = innerCandidates377_0 := by
  unfold candidates
  have row : edge 2803 = SearchData.row2803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan377_0 : scan edge (search edge 3) 4 [2617] 2617 innerTail377_0 best = best := by
  rw [innerTail377_0, scan, if_neg (by decide), innerCandidates377_0_checked]
  change scan edge (search edge 3) 4 [2617] 2617 innerTail377_1
    (if 3 ≤ innerCandidates377_0.length then search edge 3 [2617, 2803] innerCandidates377_0 5420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan377_1

theorem rootBranch377 : search edge 4 [2617] rootCandidates377 2617 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2617] 2617 innerTail377_0 best = best
  exact innerScan377_0

def rootCandidates376 : List Nat := [2777, 3257, 3491, 3803, 4241, 4931, 4973, 5651, 6197, 6569, 6599, 7451, 7829, 8069, 8273, 8297]
set_option maxRecDepth 4096 in
theorem rootCandidates376_checked : candidates edge 5 0 2609 tail377 best.bound = rootCandidates376 := by
  unfold candidates
  have row : edge 2609 = PrimePairCache.smallRow2609.contains := by funext q; rfl
  rw [row]
  decide

def innerTail376_16 : List Nat := []
theorem innerScan376_16 : scan edge (search edge 3) 4 [2609] 2609 innerTail376_16 best = best := rfl

def innerTail376_15 : List Nat := 8297 :: innerTail376_16
def innerCandidates376_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates376_15_checked : candidates edge 4 2609 8297 innerTail376_16 best.bound = innerCandidates376_15 := by
  unfold candidates
  have row : edge 8297 = SearchData.row8297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan376_15 : scan edge (search edge 3) 4 [2609] 2609 innerTail376_15 best = best := by
  rw [innerTail376_15, scan, if_neg (by decide), innerCandidates376_15_checked]
  change scan edge (search edge 3) 4 [2609] 2609 innerTail376_16
    (if 3 ≤ innerCandidates376_15.length then search edge 3 [2609, 8297] innerCandidates376_15 10906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan376_16

def innerTail376_14 : List Nat := 8273 :: innerTail376_15
def innerCandidates376_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates376_14_checked : candidates edge 4 2609 8273 innerTail376_15 best.bound = innerCandidates376_14 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan376_14 : scan edge (search edge 3) 4 [2609] 2609 innerTail376_14 best = best := by
  rw [innerTail376_14, scan, if_neg (by decide), innerCandidates376_14_checked]
  change scan edge (search edge 3) 4 [2609] 2609 innerTail376_15
    (if 3 ≤ innerCandidates376_14.length then search edge 3 [2609, 8273] innerCandidates376_14 10882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan376_15

def innerTail376_13 : List Nat := 8069 :: innerTail376_14
def innerCandidates376_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates376_13_checked : candidates edge 4 2609 8069 innerTail376_14 best.bound = innerCandidates376_13 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan376_13 : scan edge (search edge 3) 4 [2609] 2609 innerTail376_13 best = best := by
  rw [innerTail376_13, scan, if_neg (by decide), innerCandidates376_13_checked]
  change scan edge (search edge 3) 4 [2609] 2609 innerTail376_14
    (if 3 ≤ innerCandidates376_13.length then search edge 3 [2609, 8069] innerCandidates376_13 10678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan376_14

def innerTail376_12 : List Nat := 7829 :: innerTail376_13
def innerCandidates376_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates376_12_checked : candidates edge 4 2609 7829 innerTail376_13 best.bound = innerCandidates376_12 := by
  unfold candidates
  have row : edge 7829 = SearchData.row7829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan376_12 : scan edge (search edge 3) 4 [2609] 2609 innerTail376_12 best = best := by
  rw [innerTail376_12, scan, if_neg (by decide), innerCandidates376_12_checked]
  change scan edge (search edge 3) 4 [2609] 2609 innerTail376_13
    (if 3 ≤ innerCandidates376_12.length then search edge 3 [2609, 7829] innerCandidates376_12 10438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan376_13

def innerTail376_11 : List Nat := 7451 :: innerTail376_12
def innerCandidates376_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates376_11_checked : candidates edge 4 2609 7451 innerTail376_12 best.bound = innerCandidates376_11 := by
  unfold candidates
  have row : edge 7451 = SearchData.row7451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan376_11 : scan edge (search edge 3) 4 [2609] 2609 innerTail376_11 best = best := by
  rw [innerTail376_11, scan, if_neg (by decide), innerCandidates376_11_checked]
  change scan edge (search edge 3) 4 [2609] 2609 innerTail376_12
    (if 3 ≤ innerCandidates376_11.length then search edge 3 [2609, 7451] innerCandidates376_11 10060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan376_12

def innerTail376_10 : List Nat := 6599 :: innerTail376_11
def innerCandidates376_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates376_10_checked : candidates edge 4 2609 6599 innerTail376_11 best.bound = innerCandidates376_10 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan376_10 : scan edge (search edge 3) 4 [2609] 2609 innerTail376_10 best = best := by
  rw [innerTail376_10, scan, if_neg (by decide), innerCandidates376_10_checked]
  change scan edge (search edge 3) 4 [2609] 2609 innerTail376_11
    (if 3 ≤ innerCandidates376_10.length then search edge 3 [2609, 6599] innerCandidates376_10 9208 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan376_11

def innerTail376_9 : List Nat := 6569 :: innerTail376_10
def innerCandidates376_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates376_9_checked : candidates edge 4 2609 6569 innerTail376_10 best.bound = innerCandidates376_9 := by
  unfold candidates
  have row : edge 6569 = SearchData.row6569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan376_9 : scan edge (search edge 3) 4 [2609] 2609 innerTail376_9 best = best := by
  rw [innerTail376_9, scan, if_neg (by decide), innerCandidates376_9_checked]
  change scan edge (search edge 3) 4 [2609] 2609 innerTail376_10
    (if 3 ≤ innerCandidates376_9.length then search edge 3 [2609, 6569] innerCandidates376_9 9178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan376_10

def innerTail376_8 : List Nat := 6197 :: innerTail376_9
def innerCandidates376_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates376_8_checked : candidates edge 4 2609 6197 innerTail376_9 best.bound = innerCandidates376_8 := by
  unfold candidates
  have row : edge 6197 = SearchData.row6197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan376_8 : scan edge (search edge 3) 4 [2609] 2609 innerTail376_8 best = best := by
  rw [innerTail376_8, scan, if_neg (by decide), innerCandidates376_8_checked]
  change scan edge (search edge 3) 4 [2609] 2609 innerTail376_9
    (if 3 ≤ innerCandidates376_8.length then search edge 3 [2609, 6197] innerCandidates376_8 8806 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan376_9

def innerTail376_7 : List Nat := 5651 :: innerTail376_8
def innerCandidates376_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates376_7_checked : candidates edge 4 2609 5651 innerTail376_8 best.bound = innerCandidates376_7 := by
  unfold candidates
  have row : edge 5651 = SearchData.row5651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan376_7 : scan edge (search edge 3) 4 [2609] 2609 innerTail376_7 best = best := by
  rw [innerTail376_7, scan, if_neg (by decide), innerCandidates376_7_checked]
  change scan edge (search edge 3) 4 [2609] 2609 innerTail376_8
    (if 3 ≤ innerCandidates376_7.length then search edge 3 [2609, 5651] innerCandidates376_7 8260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan376_8

def innerTail376_6 : List Nat := 4973 :: innerTail376_7
def innerCandidates376_6 : List Nat := [6569, 7829]
set_option maxRecDepth 4096 in
theorem innerCandidates376_6_checked : candidates edge 4 2609 4973 innerTail376_7 best.bound = innerCandidates376_6 := by
  unfold candidates
  have row : edge 4973 = SearchData.row4973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan376_6 : scan edge (search edge 3) 4 [2609] 2609 innerTail376_6 best = best := by
  rw [innerTail376_6, scan, if_neg (by decide), innerCandidates376_6_checked]
  change scan edge (search edge 3) 4 [2609] 2609 innerTail376_7
    (if 3 ≤ innerCandidates376_6.length then search edge 3 [2609, 4973] innerCandidates376_6 7582 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan376_7

def innerTail376_5 : List Nat := 4931 :: innerTail376_6
def innerCandidates376_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates376_5_checked : candidates edge 4 2609 4931 innerTail376_6 best.bound = innerCandidates376_5 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan376_5 : scan edge (search edge 3) 4 [2609] 2609 innerTail376_5 best = best := by
  rw [innerTail376_5, scan, if_neg (by decide), innerCandidates376_5_checked]
  change scan edge (search edge 3) 4 [2609] 2609 innerTail376_6
    (if 3 ≤ innerCandidates376_5.length then search edge 3 [2609, 4931] innerCandidates376_5 7540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan376_6

def innerTail376_4 : List Nat := 4241 :: innerTail376_5
def innerCandidates376_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates376_4_checked : candidates edge 4 2609 4241 innerTail376_5 best.bound = innerCandidates376_4 := by
  unfold candidates
  have row : edge 4241 = SearchData.row4241.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan376_4 : scan edge (search edge 3) 4 [2609] 2609 innerTail376_4 best = best := by
  rw [innerTail376_4, scan, if_neg (by decide), innerCandidates376_4_checked]
  change scan edge (search edge 3) 4 [2609] 2609 innerTail376_5
    (if 3 ≤ innerCandidates376_4.length then search edge 3 [2609, 4241] innerCandidates376_4 6850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan376_5

def innerTail376_3 : List Nat := 3803 :: innerTail376_4
def innerCandidates376_3 : List Nat := [8069]
set_option maxRecDepth 4096 in
theorem innerCandidates376_3_checked : candidates edge 4 2609 3803 innerTail376_4 best.bound = innerCandidates376_3 := by
  unfold candidates
  have row : edge 3803 = SearchData.row3803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan376_3 : scan edge (search edge 3) 4 [2609] 2609 innerTail376_3 best = best := by
  rw [innerTail376_3, scan, if_neg (by decide), innerCandidates376_3_checked]
  change scan edge (search edge 3) 4 [2609] 2609 innerTail376_4
    (if 3 ≤ innerCandidates376_3.length then search edge 3 [2609, 3803] innerCandidates376_3 6412 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan376_4

def innerTail376_2 : List Nat := 3491 :: innerTail376_3
def innerCandidates376_2 : List Nat := [4931]
set_option maxRecDepth 4096 in
theorem innerCandidates376_2_checked : candidates edge 4 2609 3491 innerTail376_3 best.bound = innerCandidates376_2 := by
  unfold candidates
  have row : edge 3491 = SearchData.row3491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan376_2 : scan edge (search edge 3) 4 [2609] 2609 innerTail376_2 best = best := by
  rw [innerTail376_2, scan, if_neg (by decide), innerCandidates376_2_checked]
  change scan edge (search edge 3) 4 [2609] 2609 innerTail376_3
    (if 3 ≤ innerCandidates376_2.length then search edge 3 [2609, 3491] innerCandidates376_2 6100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan376_3

def innerTail376_1 : List Nat := 3257 :: innerTail376_2
def innerCandidates376_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates376_1_checked : candidates edge 4 2609 3257 innerTail376_2 best.bound = innerCandidates376_1 := by
  unfold candidates
  have row : edge 3257 = SearchData.row3257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan376_1 : scan edge (search edge 3) 4 [2609] 2609 innerTail376_1 best = best := by
  rw [innerTail376_1, scan, if_neg (by decide), innerCandidates376_1_checked]
  change scan edge (search edge 3) 4 [2609] 2609 innerTail376_2
    (if 3 ≤ innerCandidates376_1.length then search edge 3 [2609, 3257] innerCandidates376_1 5866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan376_2

def innerTail376_0 : List Nat := 2777 :: innerTail376_1
def innerCandidates376_0 : List Nat := [4973]
set_option maxRecDepth 4096 in
theorem innerCandidates376_0_checked : candidates edge 4 2609 2777 innerTail376_1 best.bound = innerCandidates376_0 := by
  unfold candidates
  have row : edge 2777 = SearchData.row2777.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan376_0 : scan edge (search edge 3) 4 [2609] 2609 innerTail376_0 best = best := by
  rw [innerTail376_0, scan, if_neg (by decide), innerCandidates376_0_checked]
  change scan edge (search edge 3) 4 [2609] 2609 innerTail376_1
    (if 3 ≤ innerCandidates376_0.length then search edge 3 [2609, 2777] innerCandidates376_0 5386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan376_1

theorem rootBranch376 : search edge 4 [2609] rootCandidates376 2609 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2609] 2609 innerTail376_0 best = best
  exact innerScan376_0

end ProjectEulerSolutions.PrimePairUpper.Search
