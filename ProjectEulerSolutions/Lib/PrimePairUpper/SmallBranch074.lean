import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates507 : List Nat := [4273, 4561, 4987, 5449, 5869, 6073, 6199, 6607]
set_option maxRecDepth 4096 in
theorem rootCandidates507_checked : candidates edge 5 0 3643 tail508 best.bound = rootCandidates507 := by
  unfold candidates
  have row : edge 3643 = SearchData.row3643.contains := by funext q; rfl
  rw [row]
  decide

def innerTail507_8 : List Nat := []
theorem innerScan507_8 : scan edge (search edge 3) 4 [3643] 3643 innerTail507_8 best = best := rfl

def innerTail507_7 : List Nat := 6607 :: innerTail507_8
def innerCandidates507_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates507_7_checked : candidates edge 4 3643 6607 innerTail507_8 best.bound = innerCandidates507_7 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan507_7 : scan edge (search edge 3) 4 [3643] 3643 innerTail507_7 best = best := by
  rw [innerTail507_7, scan, if_neg (by decide), innerCandidates507_7_checked]
  change scan edge (search edge 3) 4 [3643] 3643 innerTail507_8
    (if 3 ≤ innerCandidates507_7.length then search edge 3 [3643, 6607] innerCandidates507_7 10250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan507_8

def innerTail507_6 : List Nat := 6199 :: innerTail507_7
def innerCandidates507_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates507_6_checked : candidates edge 4 3643 6199 innerTail507_7 best.bound = innerCandidates507_6 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan507_6 : scan edge (search edge 3) 4 [3643] 3643 innerTail507_6 best = best := by
  rw [innerTail507_6, scan, if_neg (by decide), innerCandidates507_6_checked]
  change scan edge (search edge 3) 4 [3643] 3643 innerTail507_7
    (if 3 ≤ innerCandidates507_6.length then search edge 3 [3643, 6199] innerCandidates507_6 9842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan507_7

def innerTail507_5 : List Nat := 6073 :: innerTail507_6
def innerCandidates507_5 : List Nat := [6199]
set_option maxRecDepth 4096 in
theorem innerCandidates507_5_checked : candidates edge 4 3643 6073 innerTail507_6 best.bound = innerCandidates507_5 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan507_5 : scan edge (search edge 3) 4 [3643] 3643 innerTail507_5 best = best := by
  rw [innerTail507_5, scan, if_neg (by decide), innerCandidates507_5_checked]
  change scan edge (search edge 3) 4 [3643] 3643 innerTail507_6
    (if 3 ≤ innerCandidates507_5.length then search edge 3 [3643, 6073] innerCandidates507_5 9716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan507_6

def innerTail507_4 : List Nat := 5869 :: innerTail507_5
def innerCandidates507_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates507_4_checked : candidates edge 4 3643 5869 innerTail507_5 best.bound = innerCandidates507_4 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan507_4 : scan edge (search edge 3) 4 [3643] 3643 innerTail507_4 best = best := by
  rw [innerTail507_4, scan, if_neg (by decide), innerCandidates507_4_checked]
  change scan edge (search edge 3) 4 [3643] 3643 innerTail507_5
    (if 3 ≤ innerCandidates507_4.length then search edge 3 [3643, 5869] innerCandidates507_4 9512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan507_5

def innerTail507_3 : List Nat := 5449 :: innerTail507_4
def innerCandidates507_3 : List Nat := [6073]
set_option maxRecDepth 4096 in
theorem innerCandidates507_3_checked : candidates edge 4 3643 5449 innerTail507_4 best.bound = innerCandidates507_3 := by
  unfold candidates
  have row : edge 5449 = SearchData.row5449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan507_3 : scan edge (search edge 3) 4 [3643] 3643 innerTail507_3 best = best := by
  rw [innerTail507_3, scan, if_neg (by decide), innerCandidates507_3_checked]
  change scan edge (search edge 3) 4 [3643] 3643 innerTail507_4
    (if 3 ≤ innerCandidates507_3.length then search edge 3 [3643, 5449] innerCandidates507_3 9092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan507_4

def innerTail507_2 : List Nat := 4987 :: innerTail507_3
def innerCandidates507_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates507_2_checked : candidates edge 4 3643 4987 innerTail507_3 best.bound = innerCandidates507_2 := by
  unfold candidates
  have row : edge 4987 = SearchData.row4987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan507_2 : scan edge (search edge 3) 4 [3643] 3643 innerTail507_2 best = best := by
  rw [innerTail507_2, scan, if_neg (by decide), innerCandidates507_2_checked]
  change scan edge (search edge 3) 4 [3643] 3643 innerTail507_3
    (if 3 ≤ innerCandidates507_2.length then search edge 3 [3643, 4987] innerCandidates507_2 8630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan507_3

def innerTail507_1 : List Nat := 4561 :: innerTail507_2
def innerCandidates507_1 : List Nat := [6073]
set_option maxRecDepth 4096 in
theorem innerCandidates507_1_checked : candidates edge 4 3643 4561 innerTail507_2 best.bound = innerCandidates507_1 := by
  unfold candidates
  have row : edge 4561 = SearchData.row4561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan507_1 : scan edge (search edge 3) 4 [3643] 3643 innerTail507_1 best = best := by
  rw [innerTail507_1, scan, if_neg (by decide), innerCandidates507_1_checked]
  change scan edge (search edge 3) 4 [3643] 3643 innerTail507_2
    (if 3 ≤ innerCandidates507_1.length then search edge 3 [3643, 4561] innerCandidates507_1 8204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan507_2

def innerTail507_0 : List Nat := 4273 :: innerTail507_1
def innerCandidates507_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates507_0_checked : candidates edge 4 3643 4273 innerTail507_1 best.bound = innerCandidates507_0 := by
  unfold candidates
  have row : edge 4273 = SearchData.row4273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan507_0 : scan edge (search edge 3) 4 [3643] 3643 innerTail507_0 best = best := by
  rw [innerTail507_0, scan, if_neg (by decide), innerCandidates507_0_checked]
  change scan edge (search edge 3) 4 [3643] 3643 innerTail507_1
    (if 3 ≤ innerCandidates507_0.length then search edge 3 [3643, 4273] innerCandidates507_0 7916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan507_1

theorem rootBranch507 : search edge 4 [3643] rootCandidates507 3643 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3643] 3643 innerTail507_0 best = best
  exact innerScan507_0

def rootCandidates506 : List Nat := [3643, 5407, 5503, 5659, 5839, 6343, 6841, 6871, 7639, 8059]
set_option maxRecDepth 4096 in
theorem rootCandidates506_checked : candidates edge 5 0 3637 tail507 best.bound = rootCandidates506 := by
  unfold candidates
  have row : edge 3637 = SearchData.row3637.contains := by funext q; rfl
  rw [row]
  decide

def innerTail506_10 : List Nat := []
theorem innerScan506_10 : scan edge (search edge 3) 4 [3637] 3637 innerTail506_10 best = best := rfl

def innerTail506_9 : List Nat := 8059 :: innerTail506_10
def innerCandidates506_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates506_9_checked : candidates edge 4 3637 8059 innerTail506_10 best.bound = innerCandidates506_9 := by
  unfold candidates
  have row : edge 8059 = SearchData.row8059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan506_9 : scan edge (search edge 3) 4 [3637] 3637 innerTail506_9 best = best := by
  rw [innerTail506_9, scan, if_neg (by decide), innerCandidates506_9_checked]
  change scan edge (search edge 3) 4 [3637] 3637 innerTail506_10
    (if 3 ≤ innerCandidates506_9.length then search edge 3 [3637, 8059] innerCandidates506_9 11696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan506_10

def innerTail506_8 : List Nat := 7639 :: innerTail506_9
def innerCandidates506_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates506_8_checked : candidates edge 4 3637 7639 innerTail506_9 best.bound = innerCandidates506_8 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan506_8 : scan edge (search edge 3) 4 [3637] 3637 innerTail506_8 best = best := by
  rw [innerTail506_8, scan, if_neg (by decide), innerCandidates506_8_checked]
  change scan edge (search edge 3) 4 [3637] 3637 innerTail506_9
    (if 3 ≤ innerCandidates506_8.length then search edge 3 [3637, 7639] innerCandidates506_8 11276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan506_9

def innerTail506_7 : List Nat := 6871 :: innerTail506_8
def innerCandidates506_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates506_7_checked : candidates edge 4 3637 6871 innerTail506_8 best.bound = innerCandidates506_7 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan506_7 : scan edge (search edge 3) 4 [3637] 3637 innerTail506_7 best = best := by
  rw [innerTail506_7, scan, if_neg (by decide), innerCandidates506_7_checked]
  change scan edge (search edge 3) 4 [3637] 3637 innerTail506_8
    (if 3 ≤ innerCandidates506_7.length then search edge 3 [3637, 6871] innerCandidates506_7 10508 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan506_8

def innerTail506_6 : List Nat := 6841 :: innerTail506_7
def innerCandidates506_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates506_6_checked : candidates edge 4 3637 6841 innerTail506_7 best.bound = innerCandidates506_6 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan506_6 : scan edge (search edge 3) 4 [3637] 3637 innerTail506_6 best = best := by
  rw [innerTail506_6, scan, if_neg (by decide), innerCandidates506_6_checked]
  change scan edge (search edge 3) 4 [3637] 3637 innerTail506_7
    (if 3 ≤ innerCandidates506_6.length then search edge 3 [3637, 6841] innerCandidates506_6 10478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan506_7

def innerTail506_5 : List Nat := 6343 :: innerTail506_6
def innerCandidates506_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates506_5_checked : candidates edge 4 3637 6343 innerTail506_6 best.bound = innerCandidates506_5 := by
  unfold candidates
  have row : edge 6343 = SearchData.row6343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan506_5 : scan edge (search edge 3) 4 [3637] 3637 innerTail506_5 best = best := by
  rw [innerTail506_5, scan, if_neg (by decide), innerCandidates506_5_checked]
  change scan edge (search edge 3) 4 [3637] 3637 innerTail506_6
    (if 3 ≤ innerCandidates506_5.length then search edge 3 [3637, 6343] innerCandidates506_5 9980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan506_6

def innerTail506_4 : List Nat := 5839 :: innerTail506_5
def innerCandidates506_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates506_4_checked : candidates edge 4 3637 5839 innerTail506_5 best.bound = innerCandidates506_4 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan506_4 : scan edge (search edge 3) 4 [3637] 3637 innerTail506_4 best = best := by
  rw [innerTail506_4, scan, if_neg (by decide), innerCandidates506_4_checked]
  change scan edge (search edge 3) 4 [3637] 3637 innerTail506_5
    (if 3 ≤ innerCandidates506_4.length then search edge 3 [3637, 5839] innerCandidates506_4 9476 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan506_5

def innerTail506_3 : List Nat := 5659 :: innerTail506_4
def innerCandidates506_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates506_3_checked : candidates edge 4 3637 5659 innerTail506_4 best.bound = innerCandidates506_3 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan506_3 : scan edge (search edge 3) 4 [3637] 3637 innerTail506_3 best = best := by
  rw [innerTail506_3, scan, if_neg (by decide), innerCandidates506_3_checked]
  change scan edge (search edge 3) 4 [3637] 3637 innerTail506_4
    (if 3 ≤ innerCandidates506_3.length then search edge 3 [3637, 5659] innerCandidates506_3 9296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan506_4

def innerTail506_2 : List Nat := 5503 :: innerTail506_3
def innerCandidates506_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates506_2_checked : candidates edge 4 3637 5503 innerTail506_3 best.bound = innerCandidates506_2 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan506_2 : scan edge (search edge 3) 4 [3637] 3637 innerTail506_2 best = best := by
  rw [innerTail506_2, scan, if_neg (by decide), innerCandidates506_2_checked]
  change scan edge (search edge 3) 4 [3637] 3637 innerTail506_3
    (if 3 ≤ innerCandidates506_2.length then search edge 3 [3637, 5503] innerCandidates506_2 9140 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan506_3

def innerTail506_1 : List Nat := 5407 :: innerTail506_2
def innerCandidates506_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates506_1_checked : candidates edge 4 3637 5407 innerTail506_2 best.bound = innerCandidates506_1 := by
  unfold candidates
  have row : edge 5407 = SearchData.row5407.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan506_1 : scan edge (search edge 3) 4 [3637] 3637 innerTail506_1 best = best := by
  rw [innerTail506_1, scan, if_neg (by decide), innerCandidates506_1_checked]
  change scan edge (search edge 3) 4 [3637] 3637 innerTail506_2
    (if 3 ≤ innerCandidates506_1.length then search edge 3 [3637, 5407] innerCandidates506_1 9044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan506_2

def innerTail506_0 : List Nat := 3643 :: innerTail506_1
def innerCandidates506_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates506_0_checked : candidates edge 4 3637 3643 innerTail506_1 best.bound = innerCandidates506_0 := by
  unfold candidates
  have row : edge 3643 = SearchData.row3643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan506_0 : scan edge (search edge 3) 4 [3637] 3637 innerTail506_0 best = best := by
  rw [innerTail506_0, scan, if_neg (by decide), innerCandidates506_0_checked]
  change scan edge (search edge 3) 4 [3637] 3637 innerTail506_1
    (if 3 ≤ innerCandidates506_0.length then search edge 3 [3637, 3643] innerCandidates506_0 7280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan506_1

theorem rootBranch506 : search edge 4 [3637] rootCandidates506 3637 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3637] 3637 innerTail506_0 best = best
  exact innerScan506_0

def rootCandidates505 : List Nat := [4513, 5119, 5347, 6547, 6691, 7507, 7789, 8017, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates505_checked : candidates edge 5 0 3631 tail506 best.bound = rootCandidates505 := by
  unfold candidates
  have row : edge 3631 = SearchData.row3631.contains := by funext q; rfl
  rw [row]
  decide

def innerTail505_9 : List Nat := []
theorem innerScan505_9 : scan edge (search edge 3) 4 [3631] 3631 innerTail505_9 best = best := rfl

def innerTail505_8 : List Nat := 8377 :: innerTail505_9
def innerCandidates505_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates505_8_checked : candidates edge 4 3631 8377 innerTail505_9 best.bound = innerCandidates505_8 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan505_8 : scan edge (search edge 3) 4 [3631] 3631 innerTail505_8 best = best := by
  rw [innerTail505_8, scan, if_neg (by decide), innerCandidates505_8_checked]
  change scan edge (search edge 3) 4 [3631] 3631 innerTail505_9
    (if 3 ≤ innerCandidates505_8.length then search edge 3 [3631, 8377] innerCandidates505_8 12008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan505_9

def innerTail505_7 : List Nat := 8017 :: innerTail505_8
def innerCandidates505_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates505_7_checked : candidates edge 4 3631 8017 innerTail505_8 best.bound = innerCandidates505_7 := by
  unfold candidates
  have row : edge 8017 = SearchData.row8017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan505_7 : scan edge (search edge 3) 4 [3631] 3631 innerTail505_7 best = best := by
  rw [innerTail505_7, scan, if_neg (by decide), innerCandidates505_7_checked]
  change scan edge (search edge 3) 4 [3631] 3631 innerTail505_8
    (if 3 ≤ innerCandidates505_7.length then search edge 3 [3631, 8017] innerCandidates505_7 11648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan505_8

def innerTail505_6 : List Nat := 7789 :: innerTail505_7
def innerCandidates505_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates505_6_checked : candidates edge 4 3631 7789 innerTail505_7 best.bound = innerCandidates505_6 := by
  unfold candidates
  have row : edge 7789 = SearchData.row7789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan505_6 : scan edge (search edge 3) 4 [3631] 3631 innerTail505_6 best = best := by
  rw [innerTail505_6, scan, if_neg (by decide), innerCandidates505_6_checked]
  change scan edge (search edge 3) 4 [3631] 3631 innerTail505_7
    (if 3 ≤ innerCandidates505_6.length then search edge 3 [3631, 7789] innerCandidates505_6 11420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan505_7

def innerTail505_5 : List Nat := 7507 :: innerTail505_6
def innerCandidates505_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates505_5_checked : candidates edge 4 3631 7507 innerTail505_6 best.bound = innerCandidates505_5 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan505_5 : scan edge (search edge 3) 4 [3631] 3631 innerTail505_5 best = best := by
  rw [innerTail505_5, scan, if_neg (by decide), innerCandidates505_5_checked]
  change scan edge (search edge 3) 4 [3631] 3631 innerTail505_6
    (if 3 ≤ innerCandidates505_5.length then search edge 3 [3631, 7507] innerCandidates505_5 11138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan505_6

def innerTail505_4 : List Nat := 6691 :: innerTail505_5
def innerCandidates505_4 : List Nat := [7507, 8377]
set_option maxRecDepth 4096 in
theorem innerCandidates505_4_checked : candidates edge 4 3631 6691 innerTail505_5 best.bound = innerCandidates505_4 := by
  unfold candidates
  have row : edge 6691 = SearchData.row6691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan505_4 : scan edge (search edge 3) 4 [3631] 3631 innerTail505_4 best = best := by
  rw [innerTail505_4, scan, if_neg (by decide), innerCandidates505_4_checked]
  change scan edge (search edge 3) 4 [3631] 3631 innerTail505_5
    (if 3 ≤ innerCandidates505_4.length then search edge 3 [3631, 6691] innerCandidates505_4 10322 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan505_5

def innerTail505_3 : List Nat := 6547 :: innerTail505_4
def innerCandidates505_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates505_3_checked : candidates edge 4 3631 6547 innerTail505_4 best.bound = innerCandidates505_3 := by
  unfold candidates
  have row : edge 6547 = SearchData.row6547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan505_3 : scan edge (search edge 3) 4 [3631] 3631 innerTail505_3 best = best := by
  rw [innerTail505_3, scan, if_neg (by decide), innerCandidates505_3_checked]
  change scan edge (search edge 3) 4 [3631] 3631 innerTail505_4
    (if 3 ≤ innerCandidates505_3.length then search edge 3 [3631, 6547] innerCandidates505_3 10178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan505_4

def innerTail505_2 : List Nat := 5347 :: innerTail505_3
def innerCandidates505_2 : List Nat := [7789]
set_option maxRecDepth 4096 in
theorem innerCandidates505_2_checked : candidates edge 4 3631 5347 innerTail505_3 best.bound = innerCandidates505_2 := by
  unfold candidates
  have row : edge 5347 = SearchData.row5347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan505_2 : scan edge (search edge 3) 4 [3631] 3631 innerTail505_2 best = best := by
  rw [innerTail505_2, scan, if_neg (by decide), innerCandidates505_2_checked]
  change scan edge (search edge 3) 4 [3631] 3631 innerTail505_3
    (if 3 ≤ innerCandidates505_2.length then search edge 3 [3631, 5347] innerCandidates505_2 8978 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan505_3

def innerTail505_1 : List Nat := 5119 :: innerTail505_2
def innerCandidates505_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates505_1_checked : candidates edge 4 3631 5119 innerTail505_2 best.bound = innerCandidates505_1 := by
  unfold candidates
  have row : edge 5119 = SearchData.row5119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan505_1 : scan edge (search edge 3) 4 [3631] 3631 innerTail505_1 best = best := by
  rw [innerTail505_1, scan, if_neg (by decide), innerCandidates505_1_checked]
  change scan edge (search edge 3) 4 [3631] 3631 innerTail505_2
    (if 3 ≤ innerCandidates505_1.length then search edge 3 [3631, 5119] innerCandidates505_1 8750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan505_2

def innerTail505_0 : List Nat := 4513 :: innerTail505_1
def innerCandidates505_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates505_0_checked : candidates edge 4 3631 4513 innerTail505_1 best.bound = innerCandidates505_0 := by
  unfold candidates
  have row : edge 4513 = SearchData.row4513.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan505_0 : scan edge (search edge 3) 4 [3631] 3631 innerTail505_0 best = best := by
  rw [innerTail505_0, scan, if_neg (by decide), innerCandidates505_0_checked]
  change scan edge (search edge 3) 4 [3631] 3631 innerTail505_1
    (if 3 ≤ innerCandidates505_0.length then search edge 3 [3631, 4513] innerCandidates505_0 8144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan505_1

theorem rootBranch505 : search edge 4 [3631] rootCandidates505 3631 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3631] 3631 innerTail505_0 best = best
  exact innerScan505_0

def rootCandidates504 : List Nat := [4049, 4091, 4127, 4271, 4967, 5981, 6449, 6521, 8123]
set_option maxRecDepth 4096 in
theorem rootCandidates504_checked : candidates edge 5 0 3623 tail505 best.bound = rootCandidates504 := by
  unfold candidates
  have row : edge 3623 = SearchData.row3623.contains := by funext q; rfl
  rw [row]
  decide

def innerTail504_9 : List Nat := []
theorem innerScan504_9 : scan edge (search edge 3) 4 [3623] 3623 innerTail504_9 best = best := rfl

def innerTail504_8 : List Nat := 8123 :: innerTail504_9
def innerCandidates504_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates504_8_checked : candidates edge 4 3623 8123 innerTail504_9 best.bound = innerCandidates504_8 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan504_8 : scan edge (search edge 3) 4 [3623] 3623 innerTail504_8 best = best := by
  rw [innerTail504_8, scan, if_neg (by decide), innerCandidates504_8_checked]
  change scan edge (search edge 3) 4 [3623] 3623 innerTail504_9
    (if 3 ≤ innerCandidates504_8.length then search edge 3 [3623, 8123] innerCandidates504_8 11746 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan504_9

def innerTail504_7 : List Nat := 6521 :: innerTail504_8
def innerCandidates504_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates504_7_checked : candidates edge 4 3623 6521 innerTail504_8 best.bound = innerCandidates504_7 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan504_7 : scan edge (search edge 3) 4 [3623] 3623 innerTail504_7 best = best := by
  rw [innerTail504_7, scan, if_neg (by decide), innerCandidates504_7_checked]
  change scan edge (search edge 3) 4 [3623] 3623 innerTail504_8
    (if 3 ≤ innerCandidates504_7.length then search edge 3 [3623, 6521] innerCandidates504_7 10144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan504_8

def innerTail504_6 : List Nat := 6449 :: innerTail504_7
def innerCandidates504_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates504_6_checked : candidates edge 4 3623 6449 innerTail504_7 best.bound = innerCandidates504_6 := by
  unfold candidates
  have row : edge 6449 = SearchData.row6449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan504_6 : scan edge (search edge 3) 4 [3623] 3623 innerTail504_6 best = best := by
  rw [innerTail504_6, scan, if_neg (by decide), innerCandidates504_6_checked]
  change scan edge (search edge 3) 4 [3623] 3623 innerTail504_7
    (if 3 ≤ innerCandidates504_6.length then search edge 3 [3623, 6449] innerCandidates504_6 10072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan504_7

def innerTail504_5 : List Nat := 5981 :: innerTail504_6
def innerCandidates504_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates504_5_checked : candidates edge 4 3623 5981 innerTail504_6 best.bound = innerCandidates504_5 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan504_5 : scan edge (search edge 3) 4 [3623] 3623 innerTail504_5 best = best := by
  rw [innerTail504_5, scan, if_neg (by decide), innerCandidates504_5_checked]
  change scan edge (search edge 3) 4 [3623] 3623 innerTail504_6
    (if 3 ≤ innerCandidates504_5.length then search edge 3 [3623, 5981] innerCandidates504_5 9604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan504_6

def innerTail504_4 : List Nat := 4967 :: innerTail504_5
def innerCandidates504_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates504_4_checked : candidates edge 4 3623 4967 innerTail504_5 best.bound = innerCandidates504_4 := by
  unfold candidates
  have row : edge 4967 = SearchData.row4967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan504_4 : scan edge (search edge 3) 4 [3623] 3623 innerTail504_4 best = best := by
  rw [innerTail504_4, scan, if_neg (by decide), innerCandidates504_4_checked]
  change scan edge (search edge 3) 4 [3623] 3623 innerTail504_5
    (if 3 ≤ innerCandidates504_4.length then search edge 3 [3623, 4967] innerCandidates504_4 8590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan504_5

def innerTail504_3 : List Nat := 4271 :: innerTail504_4
def innerCandidates504_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates504_3_checked : candidates edge 4 3623 4271 innerTail504_4 best.bound = innerCandidates504_3 := by
  unfold candidates
  have row : edge 4271 = SearchData.row4271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan504_3 : scan edge (search edge 3) 4 [3623] 3623 innerTail504_3 best = best := by
  rw [innerTail504_3, scan, if_neg (by decide), innerCandidates504_3_checked]
  change scan edge (search edge 3) 4 [3623] 3623 innerTail504_4
    (if 3 ≤ innerCandidates504_3.length then search edge 3 [3623, 4271] innerCandidates504_3 7894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan504_4

def innerTail504_2 : List Nat := 4127 :: innerTail504_3
def innerCandidates504_2 : List Nat := [4967]
set_option maxRecDepth 4096 in
theorem innerCandidates504_2_checked : candidates edge 4 3623 4127 innerTail504_3 best.bound = innerCandidates504_2 := by
  unfold candidates
  have row : edge 4127 = SearchData.row4127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan504_2 : scan edge (search edge 3) 4 [3623] 3623 innerTail504_2 best = best := by
  rw [innerTail504_2, scan, if_neg (by decide), innerCandidates504_2_checked]
  change scan edge (search edge 3) 4 [3623] 3623 innerTail504_3
    (if 3 ≤ innerCandidates504_2.length then search edge 3 [3623, 4127] innerCandidates504_2 7750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan504_3

def innerTail504_1 : List Nat := 4091 :: innerTail504_2
def innerCandidates504_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates504_1_checked : candidates edge 4 3623 4091 innerTail504_2 best.bound = innerCandidates504_1 := by
  unfold candidates
  have row : edge 4091 = SearchData.row4091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan504_1 : scan edge (search edge 3) 4 [3623] 3623 innerTail504_1 best = best := by
  rw [innerTail504_1, scan, if_neg (by decide), innerCandidates504_1_checked]
  change scan edge (search edge 3) 4 [3623] 3623 innerTail504_2
    (if 3 ≤ innerCandidates504_1.length then search edge 3 [3623, 4091] innerCandidates504_1 7714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan504_2

def innerTail504_0 : List Nat := 4049 :: innerTail504_1
def innerCandidates504_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates504_0_checked : candidates edge 4 3623 4049 innerTail504_1 best.bound = innerCandidates504_0 := by
  unfold candidates
  have row : edge 4049 = SearchData.row4049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan504_0 : scan edge (search edge 3) 4 [3623] 3623 innerTail504_0 best = best := by
  rw [innerTail504_0, scan, if_neg (by decide), innerCandidates504_0_checked]
  change scan edge (search edge 3) 4 [3623] 3623 innerTail504_1
    (if 3 ≤ innerCandidates504_0.length then search edge 3 [3623, 4049] innerCandidates504_0 7672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan504_1

theorem rootBranch504 : search edge 4 [3623] rootCandidates504 3623 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3623] 3623 innerTail504_0 best = best
  exact innerScan504_0

def rootCandidates503 : List Nat := [3677, 3803, 3923, 4517, 4643, 4703, 5003, 5333, 5651, 5669, 6089, 7607, 7901, 8231, 8291]
set_option maxRecDepth 4096 in
theorem rootCandidates503_checked : candidates edge 5 0 3617 tail504 best.bound = rootCandidates503 := by
  unfold candidates
  have row : edge 3617 = SearchData.row3617.contains := by funext q; rfl
  rw [row]
  decide

def innerTail503_15 : List Nat := []
theorem innerScan503_15 : scan edge (search edge 3) 4 [3617] 3617 innerTail503_15 best = best := rfl

def innerTail503_14 : List Nat := 8291 :: innerTail503_15
def innerCandidates503_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates503_14_checked : candidates edge 4 3617 8291 innerTail503_15 best.bound = innerCandidates503_14 := by
  unfold candidates
  have row : edge 8291 = SearchData.row8291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan503_14 : scan edge (search edge 3) 4 [3617] 3617 innerTail503_14 best = best := by
  rw [innerTail503_14, scan, if_neg (by decide), innerCandidates503_14_checked]
  change scan edge (search edge 3) 4 [3617] 3617 innerTail503_15
    (if 3 ≤ innerCandidates503_14.length then search edge 3 [3617, 8291] innerCandidates503_14 11908 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan503_15

def innerTail503_13 : List Nat := 8231 :: innerTail503_14
def innerCandidates503_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates503_13_checked : candidates edge 4 3617 8231 innerTail503_14 best.bound = innerCandidates503_13 := by
  unfold candidates
  have row : edge 8231 = SearchData.row8231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan503_13 : scan edge (search edge 3) 4 [3617] 3617 innerTail503_13 best = best := by
  rw [innerTail503_13, scan, if_neg (by decide), innerCandidates503_13_checked]
  change scan edge (search edge 3) 4 [3617] 3617 innerTail503_14
    (if 3 ≤ innerCandidates503_13.length then search edge 3 [3617, 8231] innerCandidates503_13 11848 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan503_14

def innerTail503_12 : List Nat := 7901 :: innerTail503_13
def innerCandidates503_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates503_12_checked : candidates edge 4 3617 7901 innerTail503_13 best.bound = innerCandidates503_12 := by
  unfold candidates
  have row : edge 7901 = SearchData.row7901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan503_12 : scan edge (search edge 3) 4 [3617] 3617 innerTail503_12 best = best := by
  rw [innerTail503_12, scan, if_neg (by decide), innerCandidates503_12_checked]
  change scan edge (search edge 3) 4 [3617] 3617 innerTail503_13
    (if 3 ≤ innerCandidates503_12.length then search edge 3 [3617, 7901] innerCandidates503_12 11518 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan503_13

def innerTail503_11 : List Nat := 7607 :: innerTail503_12
def innerCandidates503_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates503_11_checked : candidates edge 4 3617 7607 innerTail503_12 best.bound = innerCandidates503_11 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan503_11 : scan edge (search edge 3) 4 [3617] 3617 innerTail503_11 best = best := by
  rw [innerTail503_11, scan, if_neg (by decide), innerCandidates503_11_checked]
  change scan edge (search edge 3) 4 [3617] 3617 innerTail503_12
    (if 3 ≤ innerCandidates503_11.length then search edge 3 [3617, 7607] innerCandidates503_11 11224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan503_12

def innerTail503_10 : List Nat := 6089 :: innerTail503_11
def innerCandidates503_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates503_10_checked : candidates edge 4 3617 6089 innerTail503_11 best.bound = innerCandidates503_10 := by
  unfold candidates
  have row : edge 6089 = SearchData.row6089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan503_10 : scan edge (search edge 3) 4 [3617] 3617 innerTail503_10 best = best := by
  rw [innerTail503_10, scan, if_neg (by decide), innerCandidates503_10_checked]
  change scan edge (search edge 3) 4 [3617] 3617 innerTail503_11
    (if 3 ≤ innerCandidates503_10.length then search edge 3 [3617, 6089] innerCandidates503_10 9706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan503_11

def innerTail503_9 : List Nat := 5669 :: innerTail503_10
def innerCandidates503_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates503_9_checked : candidates edge 4 3617 5669 innerTail503_10 best.bound = innerCandidates503_9 := by
  unfold candidates
  have row : edge 5669 = SearchData.row5669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan503_9 : scan edge (search edge 3) 4 [3617] 3617 innerTail503_9 best = best := by
  rw [innerTail503_9, scan, if_neg (by decide), innerCandidates503_9_checked]
  change scan edge (search edge 3) 4 [3617] 3617 innerTail503_10
    (if 3 ≤ innerCandidates503_9.length then search edge 3 [3617, 5669] innerCandidates503_9 9286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan503_10

def innerTail503_8 : List Nat := 5651 :: innerTail503_9
def innerCandidates503_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates503_8_checked : candidates edge 4 3617 5651 innerTail503_9 best.bound = innerCandidates503_8 := by
  unfold candidates
  have row : edge 5651 = SearchData.row5651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan503_8 : scan edge (search edge 3) 4 [3617] 3617 innerTail503_8 best = best := by
  rw [innerTail503_8, scan, if_neg (by decide), innerCandidates503_8_checked]
  change scan edge (search edge 3) 4 [3617] 3617 innerTail503_9
    (if 3 ≤ innerCandidates503_8.length then search edge 3 [3617, 5651] innerCandidates503_8 9268 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan503_9

def innerTail503_7 : List Nat := 5333 :: innerTail503_8
def innerCandidates503_7 : List Nat := [7901]
set_option maxRecDepth 4096 in
theorem innerCandidates503_7_checked : candidates edge 4 3617 5333 innerTail503_8 best.bound = innerCandidates503_7 := by
  unfold candidates
  have row : edge 5333 = SearchData.row5333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan503_7 : scan edge (search edge 3) 4 [3617] 3617 innerTail503_7 best = best := by
  rw [innerTail503_7, scan, if_neg (by decide), innerCandidates503_7_checked]
  change scan edge (search edge 3) 4 [3617] 3617 innerTail503_8
    (if 3 ≤ innerCandidates503_7.length then search edge 3 [3617, 5333] innerCandidates503_7 8950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan503_8

def innerTail503_6 : List Nat := 5003 :: innerTail503_7
def innerCandidates503_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates503_6_checked : candidates edge 4 3617 5003 innerTail503_7 best.bound = innerCandidates503_6 := by
  unfold candidates
  have row : edge 5003 = SearchData.row5003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan503_6 : scan edge (search edge 3) 4 [3617] 3617 innerTail503_6 best = best := by
  rw [innerTail503_6, scan, if_neg (by decide), innerCandidates503_6_checked]
  change scan edge (search edge 3) 4 [3617] 3617 innerTail503_7
    (if 3 ≤ innerCandidates503_6.length then search edge 3 [3617, 5003] innerCandidates503_6 8620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan503_7

def innerTail503_5 : List Nat := 4703 :: innerTail503_6
def innerCandidates503_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates503_5_checked : candidates edge 4 3617 4703 innerTail503_6 best.bound = innerCandidates503_5 := by
  unfold candidates
  have row : edge 4703 = SearchData.row4703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan503_5 : scan edge (search edge 3) 4 [3617] 3617 innerTail503_5 best = best := by
  rw [innerTail503_5, scan, if_neg (by decide), innerCandidates503_5_checked]
  change scan edge (search edge 3) 4 [3617] 3617 innerTail503_6
    (if 3 ≤ innerCandidates503_5.length then search edge 3 [3617, 4703] innerCandidates503_5 8320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan503_6

def innerTail503_4 : List Nat := 4643 :: innerTail503_5
def innerCandidates503_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates503_4_checked : candidates edge 4 3617 4643 innerTail503_5 best.bound = innerCandidates503_4 := by
  unfold candidates
  have row : edge 4643 = SearchData.row4643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan503_4 : scan edge (search edge 3) 4 [3617] 3617 innerTail503_4 best = best := by
  rw [innerTail503_4, scan, if_neg (by decide), innerCandidates503_4_checked]
  change scan edge (search edge 3) 4 [3617] 3617 innerTail503_5
    (if 3 ≤ innerCandidates503_4.length then search edge 3 [3617, 4643] innerCandidates503_4 8260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan503_5

def innerTail503_3 : List Nat := 4517 :: innerTail503_4
def innerCandidates503_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates503_3_checked : candidates edge 4 3617 4517 innerTail503_4 best.bound = innerCandidates503_3 := by
  unfold candidates
  have row : edge 4517 = SearchData.row4517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan503_3 : scan edge (search edge 3) 4 [3617] 3617 innerTail503_3 best = best := by
  rw [innerTail503_3, scan, if_neg (by decide), innerCandidates503_3_checked]
  change scan edge (search edge 3) 4 [3617] 3617 innerTail503_4
    (if 3 ≤ innerCandidates503_3.length then search edge 3 [3617, 4517] innerCandidates503_3 8134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan503_4

def innerTail503_2 : List Nat := 3923 :: innerTail503_3
def innerCandidates503_2 : List Nat := [4703]
set_option maxRecDepth 4096 in
theorem innerCandidates503_2_checked : candidates edge 4 3617 3923 innerTail503_3 best.bound = innerCandidates503_2 := by
  unfold candidates
  have row : edge 3923 = SearchData.row3923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan503_2 : scan edge (search edge 3) 4 [3617] 3617 innerTail503_2 best = best := by
  rw [innerTail503_2, scan, if_neg (by decide), innerCandidates503_2_checked]
  change scan edge (search edge 3) 4 [3617] 3617 innerTail503_3
    (if 3 ≤ innerCandidates503_2.length then search edge 3 [3617, 3923] innerCandidates503_2 7540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan503_3

def innerTail503_1 : List Nat := 3803 :: innerTail503_2
def innerCandidates503_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates503_1_checked : candidates edge 4 3617 3803 innerTail503_2 best.bound = innerCandidates503_1 := by
  unfold candidates
  have row : edge 3803 = SearchData.row3803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan503_1 : scan edge (search edge 3) 4 [3617] 3617 innerTail503_1 best = best := by
  rw [innerTail503_1, scan, if_neg (by decide), innerCandidates503_1_checked]
  change scan edge (search edge 3) 4 [3617] 3617 innerTail503_2
    (if 3 ≤ innerCandidates503_1.length then search edge 3 [3617, 3803] innerCandidates503_1 7420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan503_2

def innerTail503_0 : List Nat := 3677 :: innerTail503_1
def innerCandidates503_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates503_0_checked : candidates edge 4 3617 3677 innerTail503_1 best.bound = innerCandidates503_0 := by
  unfold candidates
  have row : edge 3677 = SearchData.row3677.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan503_0 : scan edge (search edge 3) 4 [3617] 3617 innerTail503_0 best = best := by
  rw [innerTail503_0, scan, if_neg (by decide), innerCandidates503_0_checked]
  change scan edge (search edge 3) 4 [3617] 3617 innerTail503_1
    (if 3 ≤ innerCandidates503_0.length then search edge 3 [3617, 3677] innerCandidates503_0 7294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan503_1

theorem rootBranch503 : search edge 4 [3617] rootCandidates503 3617 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3617] 3617 innerTail503_0 best = best
  exact innerScan503_0

def rootCandidates502 : List Nat := [3697, 4831, 5023, 5233, 5623, 5683, 5821, 8353]
set_option maxRecDepth 4096 in
theorem rootCandidates502_checked : candidates edge 5 0 3613 tail503 best.bound = rootCandidates502 := by
  unfold candidates
  have row : edge 3613 = SearchData.row3613.contains := by funext q; rfl
  rw [row]
  decide

def innerTail502_8 : List Nat := []
theorem innerScan502_8 : scan edge (search edge 3) 4 [3613] 3613 innerTail502_8 best = best := rfl

def innerTail502_7 : List Nat := 8353 :: innerTail502_8
def innerCandidates502_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates502_7_checked : candidates edge 4 3613 8353 innerTail502_8 best.bound = innerCandidates502_7 := by
  unfold candidates
  have row : edge 8353 = SearchData.row8353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan502_7 : scan edge (search edge 3) 4 [3613] 3613 innerTail502_7 best = best := by
  rw [innerTail502_7, scan, if_neg (by decide), innerCandidates502_7_checked]
  change scan edge (search edge 3) 4 [3613] 3613 innerTail502_8
    (if 3 ≤ innerCandidates502_7.length then search edge 3 [3613, 8353] innerCandidates502_7 11966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan502_8

def innerTail502_6 : List Nat := 5821 :: innerTail502_7
def innerCandidates502_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates502_6_checked : candidates edge 4 3613 5821 innerTail502_7 best.bound = innerCandidates502_6 := by
  unfold candidates
  have row : edge 5821 = SearchData.row5821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan502_6 : scan edge (search edge 3) 4 [3613] 3613 innerTail502_6 best = best := by
  rw [innerTail502_6, scan, if_neg (by decide), innerCandidates502_6_checked]
  change scan edge (search edge 3) 4 [3613] 3613 innerTail502_7
    (if 3 ≤ innerCandidates502_6.length then search edge 3 [3613, 5821] innerCandidates502_6 9434 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan502_7

def innerTail502_5 : List Nat := 5683 :: innerTail502_6
def innerCandidates502_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates502_5_checked : candidates edge 4 3613 5683 innerTail502_6 best.bound = innerCandidates502_5 := by
  unfold candidates
  have row : edge 5683 = SearchData.row5683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan502_5 : scan edge (search edge 3) 4 [3613] 3613 innerTail502_5 best = best := by
  rw [innerTail502_5, scan, if_neg (by decide), innerCandidates502_5_checked]
  change scan edge (search edge 3) 4 [3613] 3613 innerTail502_6
    (if 3 ≤ innerCandidates502_5.length then search edge 3 [3613, 5683] innerCandidates502_5 9296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan502_6

def innerTail502_4 : List Nat := 5623 :: innerTail502_5
def innerCandidates502_4 : List Nat := [5821]
set_option maxRecDepth 4096 in
theorem innerCandidates502_4_checked : candidates edge 4 3613 5623 innerTail502_5 best.bound = innerCandidates502_4 := by
  unfold candidates
  have row : edge 5623 = SearchData.row5623.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan502_4 : scan edge (search edge 3) 4 [3613] 3613 innerTail502_4 best = best := by
  rw [innerTail502_4, scan, if_neg (by decide), innerCandidates502_4_checked]
  change scan edge (search edge 3) 4 [3613] 3613 innerTail502_5
    (if 3 ≤ innerCandidates502_4.length then search edge 3 [3613, 5623] innerCandidates502_4 9236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan502_5

def innerTail502_3 : List Nat := 5233 :: innerTail502_4
def innerCandidates502_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates502_3_checked : candidates edge 4 3613 5233 innerTail502_4 best.bound = innerCandidates502_3 := by
  unfold candidates
  have row : edge 5233 = SearchData.row5233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan502_3 : scan edge (search edge 3) 4 [3613] 3613 innerTail502_3 best = best := by
  rw [innerTail502_3, scan, if_neg (by decide), innerCandidates502_3_checked]
  change scan edge (search edge 3) 4 [3613] 3613 innerTail502_4
    (if 3 ≤ innerCandidates502_3.length then search edge 3 [3613, 5233] innerCandidates502_3 8846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan502_4

def innerTail502_2 : List Nat := 5023 :: innerTail502_3
def innerCandidates502_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates502_2_checked : candidates edge 4 3613 5023 innerTail502_3 best.bound = innerCandidates502_2 := by
  unfold candidates
  have row : edge 5023 = SearchData.row5023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan502_2 : scan edge (search edge 3) 4 [3613] 3613 innerTail502_2 best = best := by
  rw [innerTail502_2, scan, if_neg (by decide), innerCandidates502_2_checked]
  change scan edge (search edge 3) 4 [3613] 3613 innerTail502_3
    (if 3 ≤ innerCandidates502_2.length then search edge 3 [3613, 5023] innerCandidates502_2 8636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan502_3

def innerTail502_1 : List Nat := 4831 :: innerTail502_2
def innerCandidates502_1 : List Nat := [5623]
set_option maxRecDepth 4096 in
theorem innerCandidates502_1_checked : candidates edge 4 3613 4831 innerTail502_2 best.bound = innerCandidates502_1 := by
  unfold candidates
  have row : edge 4831 = SearchData.row4831.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan502_1 : scan edge (search edge 3) 4 [3613] 3613 innerTail502_1 best = best := by
  rw [innerTail502_1, scan, if_neg (by decide), innerCandidates502_1_checked]
  change scan edge (search edge 3) 4 [3613] 3613 innerTail502_2
    (if 3 ≤ innerCandidates502_1.length then search edge 3 [3613, 4831] innerCandidates502_1 8444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan502_2

def innerTail502_0 : List Nat := 3697 :: innerTail502_1
def innerCandidates502_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates502_0_checked : candidates edge 4 3613 3697 innerTail502_1 best.bound = innerCandidates502_0 := by
  unfold candidates
  have row : edge 3697 = SearchData.row3697.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan502_0 : scan edge (search edge 3) 4 [3613] 3613 innerTail502_0 best = best := by
  rw [innerTail502_0, scan, if_neg (by decide), innerCandidates502_0_checked]
  change scan edge (search edge 3) 4 [3613] 3613 innerTail502_1
    (if 3 ≤ innerCandidates502_0.length then search edge 3 [3613, 3697] innerCandidates502_0 7310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan502_1

theorem rootBranch502 : search edge 4 [3613] rootCandidates502 3613 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3613] 3613 innerTail502_0 best = best
  exact innerScan502_0

end ProjectEulerSolutions.PrimePairUpper.Search
