import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates261 : List Nat := [1753, 2371, 2551, 2593, 2689, 3181, 3373, 3739, 3919, 4327, 4441, 4639, 6217, 6991, 7309, 7477, 7489, 7927]
set_option maxRecDepth 4096 in
theorem rootCandidates261_checked : candidates edge 5 0 1693 tail262 best.bound = rootCandidates261 := by
  unfold candidates
  have row : edge 1693 = PrimePairCache.smallRow1693.contains := by funext q; rfl
  rw [row]
  decide

def innerTail261_18 : List Nat := []
theorem innerScan261_18 : scan edge (search edge 3) 4 [1693] 1693 innerTail261_18 best = best := rfl

def innerTail261_17 : List Nat := 7927 :: innerTail261_18
def innerCandidates261_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates261_17_checked : candidates edge 4 1693 7927 innerTail261_18 best.bound = innerCandidates261_17 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan261_17 : scan edge (search edge 3) 4 [1693] 1693 innerTail261_17 best = best := by
  rw [innerTail261_17, scan, if_neg (by decide), innerCandidates261_17_checked]
  change scan edge (search edge 3) 4 [1693] 1693 innerTail261_18
    (if 3 ≤ innerCandidates261_17.length then search edge 3 [1693, 7927] innerCandidates261_17 9620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan261_18

def innerTail261_16 : List Nat := 7489 :: innerTail261_17
def innerCandidates261_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates261_16_checked : candidates edge 4 1693 7489 innerTail261_17 best.bound = innerCandidates261_16 := by
  unfold candidates
  have row : edge 7489 = SearchData.row7489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan261_16 : scan edge (search edge 3) 4 [1693] 1693 innerTail261_16 best = best := by
  rw [innerTail261_16, scan, if_neg (by decide), innerCandidates261_16_checked]
  change scan edge (search edge 3) 4 [1693] 1693 innerTail261_17
    (if 3 ≤ innerCandidates261_16.length then search edge 3 [1693, 7489] innerCandidates261_16 9182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan261_17

def innerTail261_15 : List Nat := 7477 :: innerTail261_16
def innerCandidates261_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates261_15_checked : candidates edge 4 1693 7477 innerTail261_16 best.bound = innerCandidates261_15 := by
  unfold candidates
  have row : edge 7477 = SearchData.row7477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan261_15 : scan edge (search edge 3) 4 [1693] 1693 innerTail261_15 best = best := by
  rw [innerTail261_15, scan, if_neg (by decide), innerCandidates261_15_checked]
  change scan edge (search edge 3) 4 [1693] 1693 innerTail261_16
    (if 3 ≤ innerCandidates261_15.length then search edge 3 [1693, 7477] innerCandidates261_15 9170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan261_16

def innerTail261_14 : List Nat := 7309 :: innerTail261_15
def innerCandidates261_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates261_14_checked : candidates edge 4 1693 7309 innerTail261_15 best.bound = innerCandidates261_14 := by
  unfold candidates
  have row : edge 7309 = SearchData.row7309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan261_14 : scan edge (search edge 3) 4 [1693] 1693 innerTail261_14 best = best := by
  rw [innerTail261_14, scan, if_neg (by decide), innerCandidates261_14_checked]
  change scan edge (search edge 3) 4 [1693] 1693 innerTail261_15
    (if 3 ≤ innerCandidates261_14.length then search edge 3 [1693, 7309] innerCandidates261_14 9002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan261_15

def innerTail261_13 : List Nat := 6991 :: innerTail261_14
def innerCandidates261_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates261_13_checked : candidates edge 4 1693 6991 innerTail261_14 best.bound = innerCandidates261_13 := by
  unfold candidates
  have row : edge 6991 = SearchData.row6991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan261_13 : scan edge (search edge 3) 4 [1693] 1693 innerTail261_13 best = best := by
  rw [innerTail261_13, scan, if_neg (by decide), innerCandidates261_13_checked]
  change scan edge (search edge 3) 4 [1693] 1693 innerTail261_14
    (if 3 ≤ innerCandidates261_13.length then search edge 3 [1693, 6991] innerCandidates261_13 8684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan261_14

def innerTail261_12 : List Nat := 6217 :: innerTail261_13
def innerCandidates261_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates261_12_checked : candidates edge 4 1693 6217 innerTail261_13 best.bound = innerCandidates261_12 := by
  unfold candidates
  have row : edge 6217 = SearchData.row6217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan261_12 : scan edge (search edge 3) 4 [1693] 1693 innerTail261_12 best = best := by
  rw [innerTail261_12, scan, if_neg (by decide), innerCandidates261_12_checked]
  change scan edge (search edge 3) 4 [1693] 1693 innerTail261_13
    (if 3 ≤ innerCandidates261_12.length then search edge 3 [1693, 6217] innerCandidates261_12 7910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan261_13

def innerTail261_11 : List Nat := 4639 :: innerTail261_12
def innerCandidates261_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates261_11_checked : candidates edge 4 1693 4639 innerTail261_12 best.bound = innerCandidates261_11 := by
  unfold candidates
  have row : edge 4639 = SearchData.row4639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan261_11 : scan edge (search edge 3) 4 [1693] 1693 innerTail261_11 best = best := by
  rw [innerTail261_11, scan, if_neg (by decide), innerCandidates261_11_checked]
  change scan edge (search edge 3) 4 [1693] 1693 innerTail261_12
    (if 3 ≤ innerCandidates261_11.length then search edge 3 [1693, 4639] innerCandidates261_11 6332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan261_12

def innerTail261_10 : List Nat := 4441 :: innerTail261_11
def innerCandidates261_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates261_10_checked : candidates edge 4 1693 4441 innerTail261_11 best.bound = innerCandidates261_10 := by
  unfold candidates
  have row : edge 4441 = SearchData.row4441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan261_10 : scan edge (search edge 3) 4 [1693] 1693 innerTail261_10 best = best := by
  rw [innerTail261_10, scan, if_neg (by decide), innerCandidates261_10_checked]
  change scan edge (search edge 3) 4 [1693] 1693 innerTail261_11
    (if 3 ≤ innerCandidates261_10.length then search edge 3 [1693, 4441] innerCandidates261_10 6134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan261_11

def innerTail261_9 : List Nat := 4327 :: innerTail261_10
def innerCandidates261_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates261_9_checked : candidates edge 4 1693 4327 innerTail261_10 best.bound = innerCandidates261_9 := by
  unfold candidates
  have row : edge 4327 = SearchData.row4327.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan261_9 : scan edge (search edge 3) 4 [1693] 1693 innerTail261_9 best = best := by
  rw [innerTail261_9, scan, if_neg (by decide), innerCandidates261_9_checked]
  change scan edge (search edge 3) 4 [1693] 1693 innerTail261_10
    (if 3 ≤ innerCandidates261_9.length then search edge 3 [1693, 4327] innerCandidates261_9 6020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan261_10

def innerTail261_8 : List Nat := 3919 :: innerTail261_9
def innerCandidates261_8 : List Nat := [7309, 7489]
set_option maxRecDepth 4096 in
theorem innerCandidates261_8_checked : candidates edge 4 1693 3919 innerTail261_9 best.bound = innerCandidates261_8 := by
  unfold candidates
  have row : edge 3919 = SearchData.row3919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan261_8 : scan edge (search edge 3) 4 [1693] 1693 innerTail261_8 best = best := by
  rw [innerTail261_8, scan, if_neg (by decide), innerCandidates261_8_checked]
  change scan edge (search edge 3) 4 [1693] 1693 innerTail261_9
    (if 3 ≤ innerCandidates261_8.length then search edge 3 [1693, 3919] innerCandidates261_8 5612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan261_9

def innerTail261_7 : List Nat := 3739 :: innerTail261_8
def innerCandidates261_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates261_7_checked : candidates edge 4 1693 3739 innerTail261_8 best.bound = innerCandidates261_7 := by
  unfold candidates
  have row : edge 3739 = SearchData.row3739.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan261_7 : scan edge (search edge 3) 4 [1693] 1693 innerTail261_7 best = best := by
  rw [innerTail261_7, scan, if_neg (by decide), innerCandidates261_7_checked]
  change scan edge (search edge 3) 4 [1693] 1693 innerTail261_8
    (if 3 ≤ innerCandidates261_7.length then search edge 3 [1693, 3739] innerCandidates261_7 5432 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan261_8

def innerTail261_6 : List Nat := 3373 :: innerTail261_7
def innerCandidates261_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates261_6_checked : candidates edge 4 1693 3373 innerTail261_7 best.bound = innerCandidates261_6 := by
  unfold candidates
  have row : edge 3373 = SearchData.row3373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan261_6 : scan edge (search edge 3) 4 [1693] 1693 innerTail261_6 best = best := by
  rw [innerTail261_6, scan, if_neg (by decide), innerCandidates261_6_checked]
  change scan edge (search edge 3) 4 [1693] 1693 innerTail261_7
    (if 3 ≤ innerCandidates261_6.length then search edge 3 [1693, 3373] innerCandidates261_6 5066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan261_7

def innerTail261_5 : List Nat := 3181 :: innerTail261_6
def innerCandidates261_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates261_5_checked : candidates edge 4 1693 3181 innerTail261_6 best.bound = innerCandidates261_5 := by
  unfold candidates
  have row : edge 3181 = SearchData.row3181.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan261_5 : scan edge (search edge 3) 4 [1693] 1693 innerTail261_5 best = best := by
  rw [innerTail261_5, scan, if_neg (by decide), innerCandidates261_5_checked]
  change scan edge (search edge 3) 4 [1693] 1693 innerTail261_6
    (if 3 ≤ innerCandidates261_5.length then search edge 3 [1693, 3181] innerCandidates261_5 4874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan261_6

def innerTail261_4 : List Nat := 2689 :: innerTail261_5
def innerCandidates261_4 : List Nat := [7477]
set_option maxRecDepth 4096 in
theorem innerCandidates261_4_checked : candidates edge 4 1693 2689 innerTail261_5 best.bound = innerCandidates261_4 := by
  unfold candidates
  have row : edge 2689 = SearchData.row2689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan261_4 : scan edge (search edge 3) 4 [1693] 1693 innerTail261_4 best = best := by
  rw [innerTail261_4, scan, if_neg (by decide), innerCandidates261_4_checked]
  change scan edge (search edge 3) 4 [1693] 1693 innerTail261_5
    (if 3 ≤ innerCandidates261_4.length then search edge 3 [1693, 2689] innerCandidates261_4 4382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan261_5

def innerTail261_3 : List Nat := 2593 :: innerTail261_4
def innerCandidates261_3 : List Nat := [3373, 7927]
set_option maxRecDepth 4096 in
theorem innerCandidates261_3_checked : candidates edge 4 1693 2593 innerTail261_4 best.bound = innerCandidates261_3 := by
  unfold candidates
  have row : edge 2593 = PrimePairCache.smallRow2593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan261_3 : scan edge (search edge 3) 4 [1693] 1693 innerTail261_3 best = best := by
  rw [innerTail261_3, scan, if_neg (by decide), innerCandidates261_3_checked]
  change scan edge (search edge 3) 4 [1693] 1693 innerTail261_4
    (if 3 ≤ innerCandidates261_3.length then search edge 3 [1693, 2593] innerCandidates261_3 4286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan261_4

def innerTail261_2 : List Nat := 2551 :: innerTail261_3
def innerCandidates261_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates261_2_checked : candidates edge 4 1693 2551 innerTail261_3 best.bound = innerCandidates261_2 := by
  unfold candidates
  have row : edge 2551 = PrimePairCache.smallRow2551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan261_2 : scan edge (search edge 3) 4 [1693] 1693 innerTail261_2 best = best := by
  rw [innerTail261_2, scan, if_neg (by decide), innerCandidates261_2_checked]
  change scan edge (search edge 3) 4 [1693] 1693 innerTail261_3
    (if 3 ≤ innerCandidates261_2.length then search edge 3 [1693, 2551] innerCandidates261_2 4244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan261_3

def innerTail261_1 : List Nat := 2371 :: innerTail261_2
def innerCandidates261_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates261_1_checked : candidates edge 4 1693 2371 innerTail261_2 best.bound = innerCandidates261_1 := by
  unfold candidates
  have row : edge 2371 = PrimePairCache.smallRow2371.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan261_1 : scan edge (search edge 3) 4 [1693] 1693 innerTail261_1 best = best := by
  rw [innerTail261_1, scan, if_neg (by decide), innerCandidates261_1_checked]
  change scan edge (search edge 3) 4 [1693] 1693 innerTail261_2
    (if 3 ≤ innerCandidates261_1.length then search edge 3 [1693, 2371] innerCandidates261_1 4064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan261_2

def innerTail261_0 : List Nat := 1753 :: innerTail261_1
def innerCandidates261_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates261_0_checked : candidates edge 4 1693 1753 innerTail261_1 best.bound = innerCandidates261_0 := by
  unfold candidates
  have row : edge 1753 = PrimePairCache.smallRow1753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan261_0 : scan edge (search edge 3) 4 [1693] 1693 innerTail261_0 best = best := by
  rw [innerTail261_0, scan, if_neg (by decide), innerCandidates261_0_checked]
  change scan edge (search edge 3) 4 [1693] 1693 innerTail261_1
    (if 3 ≤ innerCandidates261_0.length then search edge 3 [1693, 1753] innerCandidates261_0 3446 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan261_1

theorem rootBranch261 : search edge 4 [1693] rootCandidates261 1693 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1693] 1693 innerTail261_0 best = best
  exact innerScan261_0

def rootCandidates260 : List Nat := [2593, 2707, 2791, 3181, 3301, 3499, 3511, 4273, 4441, 4651, 5659, 5743, 6577, 6661, 6709, 6733, 7669, 7867, 7873]
set_option maxRecDepth 4096 in
theorem rootCandidates260_checked : candidates edge 5 0 1669 tail261 best.bound = rootCandidates260 := by
  unfold candidates
  have row : edge 1669 = PrimePairCache.smallRow1669.contains := by funext q; rfl
  rw [row]
  decide

def innerTail260_19 : List Nat := []
theorem innerScan260_19 : scan edge (search edge 3) 4 [1669] 1669 innerTail260_19 best = best := rfl

def innerTail260_18 : List Nat := 7873 :: innerTail260_19
def innerCandidates260_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates260_18_checked : candidates edge 4 1669 7873 innerTail260_19 best.bound = innerCandidates260_18 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan260_18 : scan edge (search edge 3) 4 [1669] 1669 innerTail260_18 best = best := by
  rw [innerTail260_18, scan, if_neg (by decide), innerCandidates260_18_checked]
  change scan edge (search edge 3) 4 [1669] 1669 innerTail260_19
    (if 3 ≤ innerCandidates260_18.length then search edge 3 [1669, 7873] innerCandidates260_18 9542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan260_19

def innerTail260_17 : List Nat := 7867 :: innerTail260_18
def innerCandidates260_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates260_17_checked : candidates edge 4 1669 7867 innerTail260_18 best.bound = innerCandidates260_17 := by
  unfold candidates
  have row : edge 7867 = SearchData.row7867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan260_17 : scan edge (search edge 3) 4 [1669] 1669 innerTail260_17 best = best := by
  rw [innerTail260_17, scan, if_neg (by decide), innerCandidates260_17_checked]
  change scan edge (search edge 3) 4 [1669] 1669 innerTail260_18
    (if 3 ≤ innerCandidates260_17.length then search edge 3 [1669, 7867] innerCandidates260_17 9536 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan260_18

def innerTail260_16 : List Nat := 7669 :: innerTail260_17
def innerCandidates260_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates260_16_checked : candidates edge 4 1669 7669 innerTail260_17 best.bound = innerCandidates260_16 := by
  unfold candidates
  have row : edge 7669 = SearchData.row7669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan260_16 : scan edge (search edge 3) 4 [1669] 1669 innerTail260_16 best = best := by
  rw [innerTail260_16, scan, if_neg (by decide), innerCandidates260_16_checked]
  change scan edge (search edge 3) 4 [1669] 1669 innerTail260_17
    (if 3 ≤ innerCandidates260_16.length then search edge 3 [1669, 7669] innerCandidates260_16 9338 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan260_17

def innerTail260_15 : List Nat := 6733 :: innerTail260_16
def innerCandidates260_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates260_15_checked : candidates edge 4 1669 6733 innerTail260_16 best.bound = innerCandidates260_15 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan260_15 : scan edge (search edge 3) 4 [1669] 1669 innerTail260_15 best = best := by
  rw [innerTail260_15, scan, if_neg (by decide), innerCandidates260_15_checked]
  change scan edge (search edge 3) 4 [1669] 1669 innerTail260_16
    (if 3 ≤ innerCandidates260_15.length then search edge 3 [1669, 6733] innerCandidates260_15 8402 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan260_16

def innerTail260_14 : List Nat := 6709 :: innerTail260_15
def innerCandidates260_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates260_14_checked : candidates edge 4 1669 6709 innerTail260_15 best.bound = innerCandidates260_14 := by
  unfold candidates
  have row : edge 6709 = SearchData.row6709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan260_14 : scan edge (search edge 3) 4 [1669] 1669 innerTail260_14 best = best := by
  rw [innerTail260_14, scan, if_neg (by decide), innerCandidates260_14_checked]
  change scan edge (search edge 3) 4 [1669] 1669 innerTail260_15
    (if 3 ≤ innerCandidates260_14.length then search edge 3 [1669, 6709] innerCandidates260_14 8378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan260_15

def innerTail260_13 : List Nat := 6661 :: innerTail260_14
def innerCandidates260_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates260_13_checked : candidates edge 4 1669 6661 innerTail260_14 best.bound = innerCandidates260_13 := by
  unfold candidates
  have row : edge 6661 = SearchData.row6661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan260_13 : scan edge (search edge 3) 4 [1669] 1669 innerTail260_13 best = best := by
  rw [innerTail260_13, scan, if_neg (by decide), innerCandidates260_13_checked]
  change scan edge (search edge 3) 4 [1669] 1669 innerTail260_14
    (if 3 ≤ innerCandidates260_13.length then search edge 3 [1669, 6661] innerCandidates260_13 8330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan260_14

def innerTail260_12 : List Nat := 6577 :: innerTail260_13
def innerCandidates260_12 : List Nat := [6661, 7669]
set_option maxRecDepth 4096 in
theorem innerCandidates260_12_checked : candidates edge 4 1669 6577 innerTail260_13 best.bound = innerCandidates260_12 := by
  unfold candidates
  have row : edge 6577 = SearchData.row6577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan260_12 : scan edge (search edge 3) 4 [1669] 1669 innerTail260_12 best = best := by
  rw [innerTail260_12, scan, if_neg (by decide), innerCandidates260_12_checked]
  change scan edge (search edge 3) 4 [1669] 1669 innerTail260_13
    (if 3 ≤ innerCandidates260_12.length then search edge 3 [1669, 6577] innerCandidates260_12 8246 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan260_13

def innerTail260_11 : List Nat := 5743 :: innerTail260_12
def innerCandidates260_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates260_11_checked : candidates edge 4 1669 5743 innerTail260_12 best.bound = innerCandidates260_11 := by
  unfold candidates
  have row : edge 5743 = SearchData.row5743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan260_11 : scan edge (search edge 3) 4 [1669] 1669 innerTail260_11 best = best := by
  rw [innerTail260_11, scan, if_neg (by decide), innerCandidates260_11_checked]
  change scan edge (search edge 3) 4 [1669] 1669 innerTail260_12
    (if 3 ≤ innerCandidates260_11.length then search edge 3 [1669, 5743] innerCandidates260_11 7412 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan260_12

def innerTail260_10 : List Nat := 5659 :: innerTail260_11
def innerCandidates260_10 : List Nat := [6577, 7867]
set_option maxRecDepth 4096 in
theorem innerCandidates260_10_checked : candidates edge 4 1669 5659 innerTail260_11 best.bound = innerCandidates260_10 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan260_10 : scan edge (search edge 3) 4 [1669] 1669 innerTail260_10 best = best := by
  rw [innerTail260_10, scan, if_neg (by decide), innerCandidates260_10_checked]
  change scan edge (search edge 3) 4 [1669] 1669 innerTail260_11
    (if 3 ≤ innerCandidates260_10.length then search edge 3 [1669, 5659] innerCandidates260_10 7328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan260_11

def innerTail260_9 : List Nat := 4651 :: innerTail260_10
def innerCandidates260_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates260_9_checked : candidates edge 4 1669 4651 innerTail260_10 best.bound = innerCandidates260_9 := by
  unfold candidates
  have row : edge 4651 = SearchData.row4651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan260_9 : scan edge (search edge 3) 4 [1669] 1669 innerTail260_9 best = best := by
  rw [innerTail260_9, scan, if_neg (by decide), innerCandidates260_9_checked]
  change scan edge (search edge 3) 4 [1669] 1669 innerTail260_10
    (if 3 ≤ innerCandidates260_9.length then search edge 3 [1669, 4651] innerCandidates260_9 6320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan260_10

def innerTail260_8 : List Nat := 4441 :: innerTail260_9
def innerCandidates260_8 : List Nat := [5743, 7873]
set_option maxRecDepth 4096 in
theorem innerCandidates260_8_checked : candidates edge 4 1669 4441 innerTail260_9 best.bound = innerCandidates260_8 := by
  unfold candidates
  have row : edge 4441 = SearchData.row4441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan260_8 : scan edge (search edge 3) 4 [1669] 1669 innerTail260_8 best = best := by
  rw [innerTail260_8, scan, if_neg (by decide), innerCandidates260_8_checked]
  change scan edge (search edge 3) 4 [1669] 1669 innerTail260_9
    (if 3 ≤ innerCandidates260_8.length then search edge 3 [1669, 4441] innerCandidates260_8 6110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan260_9

def innerTail260_7 : List Nat := 4273 :: innerTail260_8
def innerCandidates260_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates260_7_checked : candidates edge 4 1669 4273 innerTail260_8 best.bound = innerCandidates260_7 := by
  unfold candidates
  have row : edge 4273 = SearchData.row4273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan260_7 : scan edge (search edge 3) 4 [1669] 1669 innerTail260_7 best = best := by
  rw [innerTail260_7, scan, if_neg (by decide), innerCandidates260_7_checked]
  change scan edge (search edge 3) 4 [1669] 1669 innerTail260_8
    (if 3 ≤ innerCandidates260_7.length then search edge 3 [1669, 4273] innerCandidates260_7 5942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan260_8

def innerTail260_6 : List Nat := 3511 :: innerTail260_7
def innerCandidates260_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates260_6_checked : candidates edge 4 1669 3511 innerTail260_7 best.bound = innerCandidates260_6 := by
  unfold candidates
  have row : edge 3511 = SearchData.row3511.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan260_6 : scan edge (search edge 3) 4 [1669] 1669 innerTail260_6 best = best := by
  rw [innerTail260_6, scan, if_neg (by decide), innerCandidates260_6_checked]
  change scan edge (search edge 3) 4 [1669] 1669 innerTail260_7
    (if 3 ≤ innerCandidates260_6.length then search edge 3 [1669, 3511] innerCandidates260_6 5180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan260_7

def innerTail260_5 : List Nat := 3499 :: innerTail260_6
def innerCandidates260_5 : List Nat := [4651]
set_option maxRecDepth 4096 in
theorem innerCandidates260_5_checked : candidates edge 4 1669 3499 innerTail260_6 best.bound = innerCandidates260_5 := by
  unfold candidates
  have row : edge 3499 = SearchData.row3499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan260_5 : scan edge (search edge 3) 4 [1669] 1669 innerTail260_5 best = best := by
  rw [innerTail260_5, scan, if_neg (by decide), innerCandidates260_5_checked]
  change scan edge (search edge 3) 4 [1669] 1669 innerTail260_6
    (if 3 ≤ innerCandidates260_5.length then search edge 3 [1669, 3499] innerCandidates260_5 5168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan260_6

def innerTail260_4 : List Nat := 3301 :: innerTail260_5
def innerCandidates260_4 : List Nat := [4651]
set_option maxRecDepth 4096 in
theorem innerCandidates260_4_checked : candidates edge 4 1669 3301 innerTail260_5 best.bound = innerCandidates260_4 := by
  unfold candidates
  have row : edge 3301 = SearchData.row3301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan260_4 : scan edge (search edge 3) 4 [1669] 1669 innerTail260_4 best = best := by
  rw [innerTail260_4, scan, if_neg (by decide), innerCandidates260_4_checked]
  change scan edge (search edge 3) 4 [1669] 1669 innerTail260_5
    (if 3 ≤ innerCandidates260_4.length then search edge 3 [1669, 3301] innerCandidates260_4 4970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan260_5

def innerTail260_3 : List Nat := 3181 :: innerTail260_4
def innerCandidates260_3 : List Nat := [4273, 6709]
set_option maxRecDepth 4096 in
theorem innerCandidates260_3_checked : candidates edge 4 1669 3181 innerTail260_4 best.bound = innerCandidates260_3 := by
  unfold candidates
  have row : edge 3181 = SearchData.row3181.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan260_3 : scan edge (search edge 3) 4 [1669] 1669 innerTail260_3 best = best := by
  rw [innerTail260_3, scan, if_neg (by decide), innerCandidates260_3_checked]
  change scan edge (search edge 3) 4 [1669] 1669 innerTail260_4
    (if 3 ≤ innerCandidates260_3.length then search edge 3 [1669, 3181] innerCandidates260_3 4850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan260_4

def innerTail260_2 : List Nat := 2791 :: innerTail260_3
def innerCandidates260_2 : List Nat := [3499]
set_option maxRecDepth 4096 in
theorem innerCandidates260_2_checked : candidates edge 4 1669 2791 innerTail260_3 best.bound = innerCandidates260_2 := by
  unfold candidates
  have row : edge 2791 = SearchData.row2791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan260_2 : scan edge (search edge 3) 4 [1669] 1669 innerTail260_2 best = best := by
  rw [innerTail260_2, scan, if_neg (by decide), innerCandidates260_2_checked]
  change scan edge (search edge 3) 4 [1669] 1669 innerTail260_3
    (if 3 ≤ innerCandidates260_2.length then search edge 3 [1669, 2791] innerCandidates260_2 4460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan260_3

def innerTail260_1 : List Nat := 2707 :: innerTail260_2
def innerCandidates260_1 : List Nat := [2791, 4273]
set_option maxRecDepth 4096 in
theorem innerCandidates260_1_checked : candidates edge 4 1669 2707 innerTail260_2 best.bound = innerCandidates260_1 := by
  unfold candidates
  have row : edge 2707 = SearchData.row2707.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan260_1 : scan edge (search edge 3) 4 [1669] 1669 innerTail260_1 best = best := by
  rw [innerTail260_1, scan, if_neg (by decide), innerCandidates260_1_checked]
  change scan edge (search edge 3) 4 [1669] 1669 innerTail260_2
    (if 3 ≤ innerCandidates260_1.length then search edge 3 [1669, 2707] innerCandidates260_1 4376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan260_2

def innerTail260_0 : List Nat := 2593 :: innerTail260_1
def innerCandidates260_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates260_0_checked : candidates edge 4 1669 2593 innerTail260_1 best.bound = innerCandidates260_0 := by
  unfold candidates
  have row : edge 2593 = PrimePairCache.smallRow2593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan260_0 : scan edge (search edge 3) 4 [1669] 1669 innerTail260_0 best = best := by
  rw [innerTail260_0, scan, if_neg (by decide), innerCandidates260_0_checked]
  change scan edge (search edge 3) 4 [1669] 1669 innerTail260_1
    (if 3 ≤ innerCandidates260_0.length then search edge 3 [1669, 2593] innerCandidates260_0 4262 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan260_1

theorem rootBranch260 : search edge 4 [1669] rootCandidates260 1669 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1669] 1669 innerTail260_0 best = best
  exact innerScan260_0

def rootCandidates259 : List Nat := [1901, 1973, 2153, 2267, 2657, 3623, 3929, 4157, 4481, 4703, 4817, 4967, 5051, 5531, 5573, 5669, 6287, 6323, 6329, 6353, 6701]
set_option maxRecDepth 4096 in
theorem rootCandidates259_checked : candidates edge 5 0 1667 tail260 best.bound = rootCandidates259 := by
  unfold candidates
  have row : edge 1667 = PrimePairCache.smallRow1667.contains := by funext q; rfl
  rw [row]
  decide

def innerTail259_21 : List Nat := []
theorem innerScan259_21 : scan edge (search edge 3) 4 [1667] 1667 innerTail259_21 best = best := rfl

def innerTail259_20 : List Nat := 6701 :: innerTail259_21
def innerCandidates259_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates259_20_checked : candidates edge 4 1667 6701 innerTail259_21 best.bound = innerCandidates259_20 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan259_20 : scan edge (search edge 3) 4 [1667] 1667 innerTail259_20 best = best := by
  rw [innerTail259_20, scan, if_neg (by decide), innerCandidates259_20_checked]
  change scan edge (search edge 3) 4 [1667] 1667 innerTail259_21
    (if 3 ≤ innerCandidates259_20.length then search edge 3 [1667, 6701] innerCandidates259_20 8368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan259_21

def innerTail259_19 : List Nat := 6353 :: innerTail259_20
def innerCandidates259_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates259_19_checked : candidates edge 4 1667 6353 innerTail259_20 best.bound = innerCandidates259_19 := by
  unfold candidates
  have row : edge 6353 = SearchData.row6353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan259_19 : scan edge (search edge 3) 4 [1667] 1667 innerTail259_19 best = best := by
  rw [innerTail259_19, scan, if_neg (by decide), innerCandidates259_19_checked]
  change scan edge (search edge 3) 4 [1667] 1667 innerTail259_20
    (if 3 ≤ innerCandidates259_19.length then search edge 3 [1667, 6353] innerCandidates259_19 8020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan259_20

def innerTail259_18 : List Nat := 6329 :: innerTail259_19
def innerCandidates259_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates259_18_checked : candidates edge 4 1667 6329 innerTail259_19 best.bound = innerCandidates259_18 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan259_18 : scan edge (search edge 3) 4 [1667] 1667 innerTail259_18 best = best := by
  rw [innerTail259_18, scan, if_neg (by decide), innerCandidates259_18_checked]
  change scan edge (search edge 3) 4 [1667] 1667 innerTail259_19
    (if 3 ≤ innerCandidates259_18.length then search edge 3 [1667, 6329] innerCandidates259_18 7996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan259_19

def innerTail259_17 : List Nat := 6323 :: innerTail259_18
def innerCandidates259_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates259_17_checked : candidates edge 4 1667 6323 innerTail259_18 best.bound = innerCandidates259_17 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan259_17 : scan edge (search edge 3) 4 [1667] 1667 innerTail259_17 best = best := by
  rw [innerTail259_17, scan, if_neg (by decide), innerCandidates259_17_checked]
  change scan edge (search edge 3) 4 [1667] 1667 innerTail259_18
    (if 3 ≤ innerCandidates259_17.length then search edge 3 [1667, 6323] innerCandidates259_17 7990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan259_18

def innerTail259_16 : List Nat := 6287 :: innerTail259_17
def innerCandidates259_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates259_16_checked : candidates edge 4 1667 6287 innerTail259_17 best.bound = innerCandidates259_16 := by
  unfold candidates
  have row : edge 6287 = SearchData.row6287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan259_16 : scan edge (search edge 3) 4 [1667] 1667 innerTail259_16 best = best := by
  rw [innerTail259_16, scan, if_neg (by decide), innerCandidates259_16_checked]
  change scan edge (search edge 3) 4 [1667] 1667 innerTail259_17
    (if 3 ≤ innerCandidates259_16.length then search edge 3 [1667, 6287] innerCandidates259_16 7954 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan259_17

def innerTail259_15 : List Nat := 5669 :: innerTail259_16
def innerCandidates259_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates259_15_checked : candidates edge 4 1667 5669 innerTail259_16 best.bound = innerCandidates259_15 := by
  unfold candidates
  have row : edge 5669 = SearchData.row5669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan259_15 : scan edge (search edge 3) 4 [1667] 1667 innerTail259_15 best = best := by
  rw [innerTail259_15, scan, if_neg (by decide), innerCandidates259_15_checked]
  change scan edge (search edge 3) 4 [1667] 1667 innerTail259_16
    (if 3 ≤ innerCandidates259_15.length then search edge 3 [1667, 5669] innerCandidates259_15 7336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan259_16

def innerTail259_14 : List Nat := 5573 :: innerTail259_15
def innerCandidates259_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates259_14_checked : candidates edge 4 1667 5573 innerTail259_15 best.bound = innerCandidates259_14 := by
  unfold candidates
  have row : edge 5573 = SearchData.row5573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan259_14 : scan edge (search edge 3) 4 [1667] 1667 innerTail259_14 best = best := by
  rw [innerTail259_14, scan, if_neg (by decide), innerCandidates259_14_checked]
  change scan edge (search edge 3) 4 [1667] 1667 innerTail259_15
    (if 3 ≤ innerCandidates259_14.length then search edge 3 [1667, 5573] innerCandidates259_14 7240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan259_15

def innerTail259_13 : List Nat := 5531 :: innerTail259_14
def innerCandidates259_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates259_13_checked : candidates edge 4 1667 5531 innerTail259_14 best.bound = innerCandidates259_13 := by
  unfold candidates
  have row : edge 5531 = SearchData.row5531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan259_13 : scan edge (search edge 3) 4 [1667] 1667 innerTail259_13 best = best := by
  rw [innerTail259_13, scan, if_neg (by decide), innerCandidates259_13_checked]
  change scan edge (search edge 3) 4 [1667] 1667 innerTail259_14
    (if 3 ≤ innerCandidates259_13.length then search edge 3 [1667, 5531] innerCandidates259_13 7198 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan259_14

def innerTail259_12 : List Nat := 5051 :: innerTail259_13
def innerCandidates259_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates259_12_checked : candidates edge 4 1667 5051 innerTail259_13 best.bound = innerCandidates259_12 := by
  unfold candidates
  have row : edge 5051 = SearchData.row5051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan259_12 : scan edge (search edge 3) 4 [1667] 1667 innerTail259_12 best = best := by
  rw [innerTail259_12, scan, if_neg (by decide), innerCandidates259_12_checked]
  change scan edge (search edge 3) 4 [1667] 1667 innerTail259_13
    (if 3 ≤ innerCandidates259_12.length then search edge 3 [1667, 5051] innerCandidates259_12 6718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan259_13

def innerTail259_11 : List Nat := 4967 :: innerTail259_12
def innerCandidates259_11 : List Nat := [5531, 5573]
set_option maxRecDepth 4096 in
theorem innerCandidates259_11_checked : candidates edge 4 1667 4967 innerTail259_12 best.bound = innerCandidates259_11 := by
  unfold candidates
  have row : edge 4967 = SearchData.row4967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan259_11 : scan edge (search edge 3) 4 [1667] 1667 innerTail259_11 best = best := by
  rw [innerTail259_11, scan, if_neg (by decide), innerCandidates259_11_checked]
  change scan edge (search edge 3) 4 [1667] 1667 innerTail259_12
    (if 3 ≤ innerCandidates259_11.length then search edge 3 [1667, 4967] innerCandidates259_11 6634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan259_12

def innerTail259_10 : List Nat := 4817 :: innerTail259_11
def innerCandidates259_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates259_10_checked : candidates edge 4 1667 4817 innerTail259_11 best.bound = innerCandidates259_10 := by
  unfold candidates
  have row : edge 4817 = SearchData.row4817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan259_10 : scan edge (search edge 3) 4 [1667] 1667 innerTail259_10 best = best := by
  rw [innerTail259_10, scan, if_neg (by decide), innerCandidates259_10_checked]
  change scan edge (search edge 3) 4 [1667] 1667 innerTail259_11
    (if 3 ≤ innerCandidates259_10.length then search edge 3 [1667, 4817] innerCandidates259_10 6484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan259_11

def innerTail259_9 : List Nat := 4703 :: innerTail259_10
def innerCandidates259_9 : List Nat := [5531, 6329]
set_option maxRecDepth 4096 in
theorem innerCandidates259_9_checked : candidates edge 4 1667 4703 innerTail259_10 best.bound = innerCandidates259_9 := by
  unfold candidates
  have row : edge 4703 = SearchData.row4703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan259_9 : scan edge (search edge 3) 4 [1667] 1667 innerTail259_9 best = best := by
  rw [innerTail259_9, scan, if_neg (by decide), innerCandidates259_9_checked]
  change scan edge (search edge 3) 4 [1667] 1667 innerTail259_10
    (if 3 ≤ innerCandidates259_9.length then search edge 3 [1667, 4703] innerCandidates259_9 6370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan259_10

def innerTail259_8 : List Nat := 4481 :: innerTail259_9
def innerCandidates259_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates259_8_checked : candidates edge 4 1667 4481 innerTail259_9 best.bound = innerCandidates259_8 := by
  unfold candidates
  have row : edge 4481 = SearchData.row4481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan259_8 : scan edge (search edge 3) 4 [1667] 1667 innerTail259_8 best = best := by
  rw [innerTail259_8, scan, if_neg (by decide), innerCandidates259_8_checked]
  change scan edge (search edge 3) 4 [1667] 1667 innerTail259_9
    (if 3 ≤ innerCandidates259_8.length then search edge 3 [1667, 4481] innerCandidates259_8 6148 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan259_9

def innerTail259_7 : List Nat := 4157 :: innerTail259_8
def innerCandidates259_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates259_7_checked : candidates edge 4 1667 4157 innerTail259_8 best.bound = innerCandidates259_7 := by
  unfold candidates
  have row : edge 4157 = SearchData.row4157.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan259_7 : scan edge (search edge 3) 4 [1667] 1667 innerTail259_7 best = best := by
  rw [innerTail259_7, scan, if_neg (by decide), innerCandidates259_7_checked]
  change scan edge (search edge 3) 4 [1667] 1667 innerTail259_8
    (if 3 ≤ innerCandidates259_7.length then search edge 3 [1667, 4157] innerCandidates259_7 5824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan259_8

def innerTail259_6 : List Nat := 3929 :: innerTail259_7
def innerCandidates259_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates259_6_checked : candidates edge 4 1667 3929 innerTail259_7 best.bound = innerCandidates259_6 := by
  unfold candidates
  have row : edge 3929 = SearchData.row3929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan259_6 : scan edge (search edge 3) 4 [1667] 1667 innerTail259_6 best = best := by
  rw [innerTail259_6, scan, if_neg (by decide), innerCandidates259_6_checked]
  change scan edge (search edge 3) 4 [1667] 1667 innerTail259_7
    (if 3 ≤ innerCandidates259_6.length then search edge 3 [1667, 3929] innerCandidates259_6 5596 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan259_7

def innerTail259_5 : List Nat := 3623 :: innerTail259_6
def innerCandidates259_5 : List Nat := [4967]
set_option maxRecDepth 4096 in
theorem innerCandidates259_5_checked : candidates edge 4 1667 3623 innerTail259_6 best.bound = innerCandidates259_5 := by
  unfold candidates
  have row : edge 3623 = SearchData.row3623.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan259_5 : scan edge (search edge 3) 4 [1667] 1667 innerTail259_5 best = best := by
  rw [innerTail259_5, scan, if_neg (by decide), innerCandidates259_5_checked]
  change scan edge (search edge 3) 4 [1667] 1667 innerTail259_6
    (if 3 ≤ innerCandidates259_5.length then search edge 3 [1667, 3623] innerCandidates259_5 5290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan259_6

def innerTail259_4 : List Nat := 2657 :: innerTail259_5
def innerCandidates259_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates259_4_checked : candidates edge 4 1667 2657 innerTail259_5 best.bound = innerCandidates259_4 := by
  unfold candidates
  have row : edge 2657 = PrimePairCache.smallRow2657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan259_4 : scan edge (search edge 3) 4 [1667] 1667 innerTail259_4 best = best := by
  rw [innerTail259_4, scan, if_neg (by decide), innerCandidates259_4_checked]
  change scan edge (search edge 3) 4 [1667] 1667 innerTail259_5
    (if 3 ≤ innerCandidates259_4.length then search edge 3 [1667, 2657] innerCandidates259_4 4324 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan259_5

def innerTail259_3 : List Nat := 2267 :: innerTail259_4
def innerCandidates259_3 : List Nat := [5669]
set_option maxRecDepth 4096 in
theorem innerCandidates259_3_checked : candidates edge 4 1667 2267 innerTail259_4 best.bound = innerCandidates259_3 := by
  unfold candidates
  have row : edge 2267 = PrimePairCache.smallRow2267.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan259_3 : scan edge (search edge 3) 4 [1667] 1667 innerTail259_3 best = best := by
  rw [innerTail259_3, scan, if_neg (by decide), innerCandidates259_3_checked]
  change scan edge (search edge 3) 4 [1667] 1667 innerTail259_4
    (if 3 ≤ innerCandidates259_3.length then search edge 3 [1667, 2267] innerCandidates259_3 3934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan259_4

def innerTail259_2 : List Nat := 2153 :: innerTail259_3
def innerCandidates259_2 : List Nat := [4967]
set_option maxRecDepth 4096 in
theorem innerCandidates259_2_checked : candidates edge 4 1667 2153 innerTail259_3 best.bound = innerCandidates259_2 := by
  unfold candidates
  have row : edge 2153 = PrimePairCache.smallRow2153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan259_2 : scan edge (search edge 3) 4 [1667] 1667 innerTail259_2 best = best := by
  rw [innerTail259_2, scan, if_neg (by decide), innerCandidates259_2_checked]
  change scan edge (search edge 3) 4 [1667] 1667 innerTail259_3
    (if 3 ≤ innerCandidates259_2.length then search edge 3 [1667, 2153] innerCandidates259_2 3820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan259_3

def innerTail259_1 : List Nat := 1973 :: innerTail259_2
def innerCandidates259_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates259_1_checked : candidates edge 4 1667 1973 innerTail259_2 best.bound = innerCandidates259_1 := by
  unfold candidates
  have row : edge 1973 = PrimePairCache.smallRow1973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan259_1 : scan edge (search edge 3) 4 [1667] 1667 innerTail259_1 best = best := by
  rw [innerTail259_1, scan, if_neg (by decide), innerCandidates259_1_checked]
  change scan edge (search edge 3) 4 [1667] 1667 innerTail259_2
    (if 3 ≤ innerCandidates259_1.length then search edge 3 [1667, 1973] innerCandidates259_1 3640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan259_2

def innerTail259_0 : List Nat := 1901 :: innerTail259_1
def innerCandidates259_0 : List Nat := [2153, 4817]
set_option maxRecDepth 4096 in
theorem innerCandidates259_0_checked : candidates edge 4 1667 1901 innerTail259_1 best.bound = innerCandidates259_0 := by
  unfold candidates
  have row : edge 1901 = PrimePairCache.smallRow1901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan259_0 : scan edge (search edge 3) 4 [1667] 1667 innerTail259_0 best = best := by
  rw [innerTail259_0, scan, if_neg (by decide), innerCandidates259_0_checked]
  change scan edge (search edge 3) 4 [1667] 1667 innerTail259_1
    (if 3 ≤ innerCandidates259_0.length then search edge 3 [1667, 1901] innerCandidates259_0 3568 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan259_1

theorem rootBranch259 : search edge 4 [1667] rootCandidates259 1667 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1667] 1667 innerTail259_0 best = best
  exact innerScan259_0

end ProjectEulerSolutions.PrimePairUpper.Search
