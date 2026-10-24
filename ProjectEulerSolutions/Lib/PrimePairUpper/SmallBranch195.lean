import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates113 : List Nat := [821, 863, 881, 929, 2543, 2801, 2957, 4013, 4787, 4937, 5237, 5441, 5693, 6197, 6203, 6653, 7247, 7457]
set_option maxRecDepth 4096 in
theorem rootCandidates113_checked : candidates edge 5 0 641 tail114 best.bound = rootCandidates113 := by
  unfold candidates
  have row : edge 641 = PrimePairCache.smallRow641.contains := by funext q; rfl
  rw [row]
  decide

def innerTail113_18 : List Nat := []
theorem innerScan113_18 : scan edge (search edge 3) 4 [641] 641 innerTail113_18 best = best := rfl

def innerTail113_17 : List Nat := 7457 :: innerTail113_18
def innerCandidates113_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates113_17_checked : candidates edge 4 641 7457 innerTail113_18 best.bound = innerCandidates113_17 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan113_17 : scan edge (search edge 3) 4 [641] 641 innerTail113_17 best = best := by
  rw [innerTail113_17, scan, if_neg (by decide), innerCandidates113_17_checked]
  change scan edge (search edge 3) 4 [641] 641 innerTail113_18
    (if 3 ≤ innerCandidates113_17.length then search edge 3 [641, 7457] innerCandidates113_17 8098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan113_18

def innerTail113_16 : List Nat := 7247 :: innerTail113_17
def innerCandidates113_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates113_16_checked : candidates edge 4 641 7247 innerTail113_17 best.bound = innerCandidates113_16 := by
  unfold candidates
  have row : edge 7247 = SearchData.row7247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan113_16 : scan edge (search edge 3) 4 [641] 641 innerTail113_16 best = best := by
  rw [innerTail113_16, scan, if_neg (by decide), innerCandidates113_16_checked]
  change scan edge (search edge 3) 4 [641] 641 innerTail113_17
    (if 3 ≤ innerCandidates113_16.length then search edge 3 [641, 7247] innerCandidates113_16 7888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan113_17

def innerTail113_15 : List Nat := 6653 :: innerTail113_16
def innerCandidates113_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates113_15_checked : candidates edge 4 641 6653 innerTail113_16 best.bound = innerCandidates113_15 := by
  unfold candidates
  have row : edge 6653 = SearchData.row6653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan113_15 : scan edge (search edge 3) 4 [641] 641 innerTail113_15 best = best := by
  rw [innerTail113_15, scan, if_neg (by decide), innerCandidates113_15_checked]
  change scan edge (search edge 3) 4 [641] 641 innerTail113_16
    (if 3 ≤ innerCandidates113_15.length then search edge 3 [641, 6653] innerCandidates113_15 7294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan113_16

def innerTail113_14 : List Nat := 6203 :: innerTail113_15
def innerCandidates113_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates113_14_checked : candidates edge 4 641 6203 innerTail113_15 best.bound = innerCandidates113_14 := by
  unfold candidates
  have row : edge 6203 = SearchData.row6203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan113_14 : scan edge (search edge 3) 4 [641] 641 innerTail113_14 best = best := by
  rw [innerTail113_14, scan, if_neg (by decide), innerCandidates113_14_checked]
  change scan edge (search edge 3) 4 [641] 641 innerTail113_15
    (if 3 ≤ innerCandidates113_14.length then search edge 3 [641, 6203] innerCandidates113_14 6844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan113_15

def innerTail113_13 : List Nat := 6197 :: innerTail113_14
def innerCandidates113_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates113_13_checked : candidates edge 4 641 6197 innerTail113_14 best.bound = innerCandidates113_13 := by
  unfold candidates
  have row : edge 6197 = SearchData.row6197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan113_13 : scan edge (search edge 3) 4 [641] 641 innerTail113_13 best = best := by
  rw [innerTail113_13, scan, if_neg (by decide), innerCandidates113_13_checked]
  change scan edge (search edge 3) 4 [641] 641 innerTail113_14
    (if 3 ≤ innerCandidates113_13.length then search edge 3 [641, 6197] innerCandidates113_13 6838 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan113_14

def innerTail113_12 : List Nat := 5693 :: innerTail113_13
def innerCandidates113_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates113_12_checked : candidates edge 4 641 5693 innerTail113_13 best.bound = innerCandidates113_12 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan113_12 : scan edge (search edge 3) 4 [641] 641 innerTail113_12 best = best := by
  rw [innerTail113_12, scan, if_neg (by decide), innerCandidates113_12_checked]
  change scan edge (search edge 3) 4 [641] 641 innerTail113_13
    (if 3 ≤ innerCandidates113_12.length then search edge 3 [641, 5693] innerCandidates113_12 6334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan113_13

def innerTail113_11 : List Nat := 5441 :: innerTail113_12
def innerCandidates113_11 : List Nat := [5693, 7457]
set_option maxRecDepth 4096 in
theorem innerCandidates113_11_checked : candidates edge 4 641 5441 innerTail113_12 best.bound = innerCandidates113_11 := by
  unfold candidates
  have row : edge 5441 = SearchData.row5441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan113_11 : scan edge (search edge 3) 4 [641] 641 innerTail113_11 best = best := by
  rw [innerTail113_11, scan, if_neg (by decide), innerCandidates113_11_checked]
  change scan edge (search edge 3) 4 [641] 641 innerTail113_12
    (if 3 ≤ innerCandidates113_11.length then search edge 3 [641, 5441] innerCandidates113_11 6082 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan113_12

def innerTail113_10 : List Nat := 5237 :: innerTail113_11
def innerCandidates113_10 : List Nat := [5693]
set_option maxRecDepth 4096 in
theorem innerCandidates113_10_checked : candidates edge 4 641 5237 innerTail113_11 best.bound = innerCandidates113_10 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan113_10 : scan edge (search edge 3) 4 [641] 641 innerTail113_10 best = best := by
  rw [innerTail113_10, scan, if_neg (by decide), innerCandidates113_10_checked]
  change scan edge (search edge 3) 4 [641] 641 innerTail113_11
    (if 3 ≤ innerCandidates113_10.length then search edge 3 [641, 5237] innerCandidates113_10 5878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan113_11

def innerTail113_9 : List Nat := 4937 :: innerTail113_10
def innerCandidates113_9 : List Nat := [7247]
set_option maxRecDepth 4096 in
theorem innerCandidates113_9_checked : candidates edge 4 641 4937 innerTail113_10 best.bound = innerCandidates113_9 := by
  unfold candidates
  have row : edge 4937 = SearchData.row4937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan113_9 : scan edge (search edge 3) 4 [641] 641 innerTail113_9 best = best := by
  rw [innerTail113_9, scan, if_neg (by decide), innerCandidates113_9_checked]
  change scan edge (search edge 3) 4 [641] 641 innerTail113_10
    (if 3 ≤ innerCandidates113_9.length then search edge 3 [641, 4937] innerCandidates113_9 5578 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan113_10

def innerTail113_8 : List Nat := 4787 :: innerTail113_9
def innerCandidates113_8 : List Nat := [5237]
set_option maxRecDepth 4096 in
theorem innerCandidates113_8_checked : candidates edge 4 641 4787 innerTail113_9 best.bound = innerCandidates113_8 := by
  unfold candidates
  have row : edge 4787 = SearchData.row4787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan113_8 : scan edge (search edge 3) 4 [641] 641 innerTail113_8 best = best := by
  rw [innerTail113_8, scan, if_neg (by decide), innerCandidates113_8_checked]
  change scan edge (search edge 3) 4 [641] 641 innerTail113_9
    (if 3 ≤ innerCandidates113_8.length then search edge 3 [641, 4787] innerCandidates113_8 5428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan113_9

def innerTail113_7 : List Nat := 4013 :: innerTail113_8
def innerCandidates113_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates113_7_checked : candidates edge 4 641 4013 innerTail113_8 best.bound = innerCandidates113_7 := by
  unfold candidates
  have row : edge 4013 = SearchData.row4013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan113_7 : scan edge (search edge 3) 4 [641] 641 innerTail113_7 best = best := by
  rw [innerTail113_7, scan, if_neg (by decide), innerCandidates113_7_checked]
  change scan edge (search edge 3) 4 [641] 641 innerTail113_8
    (if 3 ≤ innerCandidates113_7.length then search edge 3 [641, 4013] innerCandidates113_7 4654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan113_8

def innerTail113_6 : List Nat := 2957 :: innerTail113_7
def innerCandidates113_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates113_6_checked : candidates edge 4 641 2957 innerTail113_7 best.bound = innerCandidates113_6 := by
  unfold candidates
  have row : edge 2957 = SearchData.row2957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan113_6 : scan edge (search edge 3) 4 [641] 641 innerTail113_6 best = best := by
  rw [innerTail113_6, scan, if_neg (by decide), innerCandidates113_6_checked]
  change scan edge (search edge 3) 4 [641] 641 innerTail113_7
    (if 3 ≤ innerCandidates113_6.length then search edge 3 [641, 2957] innerCandidates113_6 3598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan113_7

def innerTail113_5 : List Nat := 2801 :: innerTail113_6
def innerCandidates113_5 : List Nat := [5693]
set_option maxRecDepth 4096 in
theorem innerCandidates113_5_checked : candidates edge 4 641 2801 innerTail113_6 best.bound = innerCandidates113_5 := by
  unfold candidates
  have row : edge 2801 = SearchData.row2801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan113_5 : scan edge (search edge 3) 4 [641] 641 innerTail113_5 best = best := by
  rw [innerTail113_5, scan, if_neg (by decide), innerCandidates113_5_checked]
  change scan edge (search edge 3) 4 [641] 641 innerTail113_6
    (if 3 ≤ innerCandidates113_5.length then search edge 3 [641, 2801] innerCandidates113_5 3442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan113_6

def innerTail113_4 : List Nat := 2543 :: innerTail113_5
def innerCandidates113_4 : List Nat := [5237, 5441]
set_option maxRecDepth 4096 in
theorem innerCandidates113_4_checked : candidates edge 4 641 2543 innerTail113_5 best.bound = innerCandidates113_4 := by
  unfold candidates
  have row : edge 2543 = PrimePairCache.smallRow2543.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan113_4 : scan edge (search edge 3) 4 [641] 641 innerTail113_4 best = best := by
  rw [innerTail113_4, scan, if_neg (by decide), innerCandidates113_4_checked]
  change scan edge (search edge 3) 4 [641] 641 innerTail113_5
    (if 3 ≤ innerCandidates113_4.length then search edge 3 [641, 2543] innerCandidates113_4 3184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan113_5

def innerTail113_3 : List Nat := 929 :: innerTail113_4
def innerCandidates113_3 : List Nat := [4013]
set_option maxRecDepth 4096 in
theorem innerCandidates113_3_checked : candidates edge 4 641 929 innerTail113_4 best.bound = innerCandidates113_3 := by
  unfold candidates
  have row : edge 929 = PrimePairCache.smallRow929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan113_3 : scan edge (search edge 3) 4 [641] 641 innerTail113_3 best = best := by
  rw [innerTail113_3, scan, if_neg (by decide), innerCandidates113_3_checked]
  change scan edge (search edge 3) 4 [641] 641 innerTail113_4
    (if 3 ≤ innerCandidates113_3.length then search edge 3 [641, 929] innerCandidates113_3 1570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan113_4

def innerTail113_2 : List Nat := 881 :: innerTail113_3
def innerCandidates113_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates113_2_checked : candidates edge 4 641 881 innerTail113_3 best.bound = innerCandidates113_2 := by
  unfold candidates
  have row : edge 881 = PrimePairCache.smallRow881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan113_2 : scan edge (search edge 3) 4 [641] 641 innerTail113_2 best = best := by
  rw [innerTail113_2, scan, if_neg (by decide), innerCandidates113_2_checked]
  change scan edge (search edge 3) 4 [641] 641 innerTail113_3
    (if 3 ≤ innerCandidates113_2.length then search edge 3 [641, 881] innerCandidates113_2 1522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan113_3

def innerTail113_1 : List Nat := 863 :: innerTail113_2
def innerCandidates113_1 : List Nat := [2801]
set_option maxRecDepth 4096 in
theorem innerCandidates113_1_checked : candidates edge 4 641 863 innerTail113_2 best.bound = innerCandidates113_1 := by
  unfold candidates
  have row : edge 863 = PrimePairCache.smallRow863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan113_1 : scan edge (search edge 3) 4 [641] 641 innerTail113_1 best = best := by
  rw [innerTail113_1, scan, if_neg (by decide), innerCandidates113_1_checked]
  change scan edge (search edge 3) 4 [641] 641 innerTail113_2
    (if 3 ≤ innerCandidates113_1.length then search edge 3 [641, 863] innerCandidates113_1 1504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan113_2

def innerTail113_0 : List Nat := 821 :: innerTail113_1
def innerCandidates113_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates113_0_checked : candidates edge 4 641 821 innerTail113_1 best.bound = innerCandidates113_0 := by
  unfold candidates
  have row : edge 821 = PrimePairCache.smallRow821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan113_0 : scan edge (search edge 3) 4 [641] 641 innerTail113_0 best = best := by
  rw [innerTail113_0, scan, if_neg (by decide), innerCandidates113_0_checked]
  change scan edge (search edge 3) 4 [641] 641 innerTail113_1
    (if 3 ≤ innerCandidates113_0.length then search edge 3 [641, 821] innerCandidates113_0 1462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan113_1

theorem rootBranch113 : search edge 4 [641] rootCandidates113 641 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [641] 641 innerTail113_0 best = best
  exact innerScan113_0

def rootCandidates112 : List Nat := [739, 751, 1297, 1483, 2221, 3253, 3457, 3709, 3889, 4051, 4057, 4801, 4933, 5479, 5527, 5557, 5737, 6199, 7243, 7681, 7873, 8161]
set_option maxRecDepth 4096 in
theorem rootCandidates112_checked : candidates edge 5 0 631 tail113 best.bound = rootCandidates112 := by
  unfold candidates
  have row : edge 631 = PrimePairCache.smallRow631.contains := by funext q; rfl
  rw [row]
  decide

def innerTail112_22 : List Nat := []
theorem innerScan112_22 : scan edge (search edge 3) 4 [631] 631 innerTail112_22 best = best := rfl

def innerTail112_21 : List Nat := 8161 :: innerTail112_22
def innerCandidates112_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates112_21_checked : candidates edge 4 631 8161 innerTail112_22 best.bound = innerCandidates112_21 := by
  unfold candidates
  have row : edge 8161 = SearchData.row8161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan112_21 : scan edge (search edge 3) 4 [631] 631 innerTail112_21 best = best := by
  rw [innerTail112_21, scan, if_neg (by decide), innerCandidates112_21_checked]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_22
    (if 3 ≤ innerCandidates112_21.length then search edge 3 [631, 8161] innerCandidates112_21 8792 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan112_22

def innerTail112_20 : List Nat := 7873 :: innerTail112_21
def innerCandidates112_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates112_20_checked : candidates edge 4 631 7873 innerTail112_21 best.bound = innerCandidates112_20 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan112_20 : scan edge (search edge 3) 4 [631] 631 innerTail112_20 best = best := by
  rw [innerTail112_20, scan, if_neg (by decide), innerCandidates112_20_checked]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_21
    (if 3 ≤ innerCandidates112_20.length then search edge 3 [631, 7873] innerCandidates112_20 8504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan112_21

def innerTail112_19 : List Nat := 7681 :: innerTail112_20
def innerCandidates112_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates112_19_checked : candidates edge 4 631 7681 innerTail112_20 best.bound = innerCandidates112_19 := by
  unfold candidates
  have row : edge 7681 = SearchData.row7681.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan112_19 : scan edge (search edge 3) 4 [631] 631 innerTail112_19 best = best := by
  rw [innerTail112_19, scan, if_neg (by decide), innerCandidates112_19_checked]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_20
    (if 3 ≤ innerCandidates112_19.length then search edge 3 [631, 7681] innerCandidates112_19 8312 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan112_20

def innerTail112_18 : List Nat := 7243 :: innerTail112_19
def innerCandidates112_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates112_18_checked : candidates edge 4 631 7243 innerTail112_19 best.bound = innerCandidates112_18 := by
  unfold candidates
  have row : edge 7243 = SearchData.row7243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan112_18 : scan edge (search edge 3) 4 [631] 631 innerTail112_18 best = best := by
  rw [innerTail112_18, scan, if_neg (by decide), innerCandidates112_18_checked]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_19
    (if 3 ≤ innerCandidates112_18.length then search edge 3 [631, 7243] innerCandidates112_18 7874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan112_19

def innerTail112_17 : List Nat := 6199 :: innerTail112_18
def innerCandidates112_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates112_17_checked : candidates edge 4 631 6199 innerTail112_18 best.bound = innerCandidates112_17 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan112_17 : scan edge (search edge 3) 4 [631] 631 innerTail112_17 best = best := by
  rw [innerTail112_17, scan, if_neg (by decide), innerCandidates112_17_checked]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_18
    (if 3 ≤ innerCandidates112_17.length then search edge 3 [631, 6199] innerCandidates112_17 6830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan112_18

def innerTail112_16 : List Nat := 5737 :: innerTail112_17
def innerCandidates112_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates112_16_checked : candidates edge 4 631 5737 innerTail112_17 best.bound = innerCandidates112_16 := by
  unfold candidates
  have row : edge 5737 = SearchData.row5737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan112_16 : scan edge (search edge 3) 4 [631] 631 innerTail112_16 best = best := by
  rw [innerTail112_16, scan, if_neg (by decide), innerCandidates112_16_checked]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_17
    (if 3 ≤ innerCandidates112_16.length then search edge 3 [631, 5737] innerCandidates112_16 6368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan112_17

def innerTail112_15 : List Nat := 5557 :: innerTail112_16
def innerCandidates112_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates112_15_checked : candidates edge 4 631 5557 innerTail112_16 best.bound = innerCandidates112_15 := by
  unfold candidates
  have row : edge 5557 = SearchData.row5557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan112_15 : scan edge (search edge 3) 4 [631] 631 innerTail112_15 best = best := by
  rw [innerTail112_15, scan, if_neg (by decide), innerCandidates112_15_checked]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_16
    (if 3 ≤ innerCandidates112_15.length then search edge 3 [631, 5557] innerCandidates112_15 6188 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan112_16

def innerTail112_14 : List Nat := 5527 :: innerTail112_15
def innerCandidates112_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates112_14_checked : candidates edge 4 631 5527 innerTail112_15 best.bound = innerCandidates112_14 := by
  unfold candidates
  have row : edge 5527 = SearchData.row5527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan112_14 : scan edge (search edge 3) 4 [631] 631 innerTail112_14 best = best := by
  rw [innerTail112_14, scan, if_neg (by decide), innerCandidates112_14_checked]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_15
    (if 3 ≤ innerCandidates112_14.length then search edge 3 [631, 5527] innerCandidates112_14 6158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan112_15

def innerTail112_13 : List Nat := 5479 :: innerTail112_14
def innerCandidates112_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates112_13_checked : candidates edge 4 631 5479 innerTail112_14 best.bound = innerCandidates112_13 := by
  unfold candidates
  have row : edge 5479 = SearchData.row5479.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan112_13 : scan edge (search edge 3) 4 [631] 631 innerTail112_13 best = best := by
  rw [innerTail112_13, scan, if_neg (by decide), innerCandidates112_13_checked]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_14
    (if 3 ≤ innerCandidates112_13.length then search edge 3 [631, 5479] innerCandidates112_13 6110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan112_14

def innerTail112_12 : List Nat := 4933 :: innerTail112_13
def innerCandidates112_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates112_12_checked : candidates edge 4 631 4933 innerTail112_13 best.bound = innerCandidates112_12 := by
  unfold candidates
  have row : edge 4933 = SearchData.row4933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan112_12 : scan edge (search edge 3) 4 [631] 631 innerTail112_12 best = best := by
  rw [innerTail112_12, scan, if_neg (by decide), innerCandidates112_12_checked]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_13
    (if 3 ≤ innerCandidates112_12.length then search edge 3 [631, 4933] innerCandidates112_12 5564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan112_13

def innerTail112_11 : List Nat := 4801 :: innerTail112_12
def innerCandidates112_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates112_11_checked : candidates edge 4 631 4801 innerTail112_12 best.bound = innerCandidates112_11 := by
  unfold candidates
  have row : edge 4801 = SearchData.row4801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan112_11 : scan edge (search edge 3) 4 [631] 631 innerTail112_11 best = best := by
  rw [innerTail112_11, scan, if_neg (by decide), innerCandidates112_11_checked]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_12
    (if 3 ≤ innerCandidates112_11.length then search edge 3 [631, 4801] innerCandidates112_11 5432 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan112_12

def innerTail112_10 : List Nat := 4057 :: innerTail112_11
def innerCandidates112_10 : List Nat := [6199]
set_option maxRecDepth 4096 in
theorem innerCandidates112_10_checked : candidates edge 4 631 4057 innerTail112_11 best.bound = innerCandidates112_10 := by
  unfold candidates
  have row : edge 4057 = SearchData.row4057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan112_10 : scan edge (search edge 3) 4 [631] 631 innerTail112_10 best = best := by
  rw [innerTail112_10, scan, if_neg (by decide), innerCandidates112_10_checked]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_11
    (if 3 ≤ innerCandidates112_10.length then search edge 3 [631, 4057] innerCandidates112_10 4688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan112_11

def innerTail112_9 : List Nat := 4051 :: innerTail112_10
def innerCandidates112_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates112_9_checked : candidates edge 4 631 4051 innerTail112_10 best.bound = innerCandidates112_9 := by
  unfold candidates
  have row : edge 4051 = SearchData.row4051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan112_9 : scan edge (search edge 3) 4 [631] 631 innerTail112_9 best = best := by
  rw [innerTail112_9, scan, if_neg (by decide), innerCandidates112_9_checked]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_10
    (if 3 ≤ innerCandidates112_9.length then search edge 3 [631, 4051] innerCandidates112_9 4682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan112_10

def innerTail112_8 : List Nat := 3889 :: innerTail112_9
def innerCandidates112_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates112_8_checked : candidates edge 4 631 3889 innerTail112_9 best.bound = innerCandidates112_8 := by
  unfold candidates
  have row : edge 3889 = SearchData.row3889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan112_8 : scan edge (search edge 3) 4 [631] 631 innerTail112_8 best = best := by
  rw [innerTail112_8, scan, if_neg (by decide), innerCandidates112_8_checked]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_9
    (if 3 ≤ innerCandidates112_8.length then search edge 3 [631, 3889] innerCandidates112_8 4520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan112_9

def innerTail112_7 : List Nat := 3709 :: innerTail112_8
def innerCandidates112_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates112_7_checked : candidates edge 4 631 3709 innerTail112_8 best.bound = innerCandidates112_7 := by
  unfold candidates
  have row : edge 3709 = SearchData.row3709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan112_7 : scan edge (search edge 3) 4 [631] 631 innerTail112_7 best = best := by
  rw [innerTail112_7, scan, if_neg (by decide), innerCandidates112_7_checked]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_8
    (if 3 ≤ innerCandidates112_7.length then search edge 3 [631, 3709] innerCandidates112_7 4340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan112_8

def innerTail112_6 : List Nat := 3457 :: innerTail112_7
def innerCandidates112_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates112_6_checked : candidates edge 4 631 3457 innerTail112_7 best.bound = innerCandidates112_6 := by
  unfold candidates
  have row : edge 3457 = SearchData.row3457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan112_6 : scan edge (search edge 3) 4 [631] 631 innerTail112_6 best = best := by
  rw [innerTail112_6, scan, if_neg (by decide), innerCandidates112_6_checked]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_7
    (if 3 ≤ innerCandidates112_6.length then search edge 3 [631, 3457] innerCandidates112_6 4088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan112_7

def innerTail112_5 : List Nat := 3253 :: innerTail112_6
def innerCandidates112_5 : List Nat := [5479, 5557, 7873]
set_option maxRecDepth 4096 in
theorem innerCandidates112_5_checked : candidates edge 4 631 3253 innerTail112_6 best.bound = innerCandidates112_5 := by
  unfold candidates
  have row : edge 3253 = SearchData.row3253.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild112_5 : search edge 3 [631, 3253] innerCandidates112_5 3884 best = best := by decide

theorem innerScan112_5 : scan edge (search edge 3) 4 [631] 631 innerTail112_5 best = best := by
  rw [innerTail112_5, scan, if_neg (by decide), innerCandidates112_5_checked]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_6
    (if 3 ≤ innerCandidates112_5.length then search edge 3 [631, 3253] innerCandidates112_5 3884 best else best) = best
  rw [if_pos (by decide), innerChild112_5]
  exact innerScan112_6

def innerTail112_4 : List Nat := 2221 :: innerTail112_5
def innerCandidates112_4 : List Nat := [5527, 5737, 7681, 7873]
set_option maxRecDepth 4096 in
theorem innerCandidates112_4_checked : candidates edge 4 631 2221 innerTail112_5 best.bound = innerCandidates112_4 := by
  unfold candidates
  have row : edge 2221 = PrimePairCache.smallRow2221.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild112_4 : search edge 3 [631, 2221] innerCandidates112_4 2852 best = best := by decide

theorem innerScan112_4 : scan edge (search edge 3) 4 [631] 631 innerTail112_4 best = best := by
  rw [innerTail112_4, scan, if_neg (by decide), innerCandidates112_4_checked]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_5
    (if 3 ≤ innerCandidates112_4.length then search edge 3 [631, 2221] innerCandidates112_4 2852 best else best) = best
  rw [if_pos (by decide), innerChild112_4]
  exact innerScan112_5

def innerTail112_3 : List Nat := 1483 :: innerTail112_4
def innerCandidates112_3 : List Nat := [3457, 7873]
set_option maxRecDepth 4096 in
theorem innerCandidates112_3_checked : candidates edge 4 631 1483 innerTail112_4 best.bound = innerCandidates112_3 := by
  unfold candidates
  have row : edge 1483 = PrimePairCache.smallRow1483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan112_3 : scan edge (search edge 3) 4 [631] 631 innerTail112_3 best = best := by
  rw [innerTail112_3, scan, if_neg (by decide), innerCandidates112_3_checked]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_4
    (if 3 ≤ innerCandidates112_3.length then search edge 3 [631, 1483] innerCandidates112_3 2114 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan112_4

def innerTail112_2 : List Nat := 1297 :: innerTail112_3
def innerCandidates112_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates112_2_checked : candidates edge 4 631 1297 innerTail112_3 best.bound = innerCandidates112_2 := by
  unfold candidates
  have row : edge 1297 = PrimePairCache.smallRow1297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan112_2 : scan edge (search edge 3) 4 [631] 631 innerTail112_2 best = best := by
  rw [innerTail112_2, scan, if_neg (by decide), innerCandidates112_2_checked]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_3
    (if 3 ≤ innerCandidates112_2.length then search edge 3 [631, 1297] innerCandidates112_2 1928 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan112_3

def innerTail112_1 : List Nat := 751 :: innerTail112_2
def innerCandidates112_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates112_1_checked : candidates edge 4 631 751 innerTail112_2 best.bound = innerCandidates112_1 := by
  unfold candidates
  have row : edge 751 = PrimePairCache.smallRow751.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan112_1 : scan edge (search edge 3) 4 [631] 631 innerTail112_1 best = best := by
  rw [innerTail112_1, scan, if_neg (by decide), innerCandidates112_1_checked]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_2
    (if 3 ≤ innerCandidates112_1.length then search edge 3 [631, 751] innerCandidates112_1 1382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan112_2

def innerTail112_0 : List Nat := 739 :: innerTail112_1
def innerCandidates112_0 : List Nat := [751]
set_option maxRecDepth 4096 in
theorem innerCandidates112_0_checked : candidates edge 4 631 739 innerTail112_1 best.bound = innerCandidates112_0 := by
  unfold candidates
  have row : edge 739 = PrimePairCache.smallRow739.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan112_0 : scan edge (search edge 3) 4 [631] 631 innerTail112_0 best = best := by
  rw [innerTail112_0, scan, if_neg (by decide), innerCandidates112_0_checked]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_1
    (if 3 ≤ innerCandidates112_0.length then search edge 3 [631, 739] innerCandidates112_0 1370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan112_1

theorem rootBranch112 : search edge 4 [631] rootCandidates112 631 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [631] 631 innerTail112_0 best = best
  exact innerScan112_0

def rootCandidates111 : List Nat := [811, 1051, 1579, 1597, 1747, 2671, 2707, 3571, 3793, 4621, 5419, 6991, 7207, 7309, 7459, 7687, 7753, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates111_checked : candidates edge 5 0 619 tail112 best.bound = rootCandidates111 := by
  unfold candidates
  have row : edge 619 = PrimePairCache.smallRow619.contains := by funext q; rfl
  rw [row]
  decide

def innerTail111_18 : List Nat := []
theorem innerScan111_18 : scan edge (search edge 3) 4 [619] 619 innerTail111_18 best = best := rfl

def innerTail111_17 : List Nat := 8377 :: innerTail111_18
def innerCandidates111_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates111_17_checked : candidates edge 4 619 8377 innerTail111_18 best.bound = innerCandidates111_17 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan111_17 : scan edge (search edge 3) 4 [619] 619 innerTail111_17 best = best := by
  rw [innerTail111_17, scan, if_neg (by decide), innerCandidates111_17_checked]
  change scan edge (search edge 3) 4 [619] 619 innerTail111_18
    (if 3 ≤ innerCandidates111_17.length then search edge 3 [619, 8377] innerCandidates111_17 8996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan111_18

def innerTail111_16 : List Nat := 7753 :: innerTail111_17
def innerCandidates111_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates111_16_checked : candidates edge 4 619 7753 innerTail111_17 best.bound = innerCandidates111_16 := by
  unfold candidates
  have row : edge 7753 = SearchData.row7753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan111_16 : scan edge (search edge 3) 4 [619] 619 innerTail111_16 best = best := by
  rw [innerTail111_16, scan, if_neg (by decide), innerCandidates111_16_checked]
  change scan edge (search edge 3) 4 [619] 619 innerTail111_17
    (if 3 ≤ innerCandidates111_16.length then search edge 3 [619, 7753] innerCandidates111_16 8372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan111_17

def innerTail111_15 : List Nat := 7687 :: innerTail111_16
def innerCandidates111_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates111_15_checked : candidates edge 4 619 7687 innerTail111_16 best.bound = innerCandidates111_15 := by
  unfold candidates
  have row : edge 7687 = SearchData.row7687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan111_15 : scan edge (search edge 3) 4 [619] 619 innerTail111_15 best = best := by
  rw [innerTail111_15, scan, if_neg (by decide), innerCandidates111_15_checked]
  change scan edge (search edge 3) 4 [619] 619 innerTail111_16
    (if 3 ≤ innerCandidates111_15.length then search edge 3 [619, 7687] innerCandidates111_15 8306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan111_16

def innerTail111_14 : List Nat := 7459 :: innerTail111_15
def innerCandidates111_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates111_14_checked : candidates edge 4 619 7459 innerTail111_15 best.bound = innerCandidates111_14 := by
  unfold candidates
  have row : edge 7459 = SearchData.row7459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan111_14 : scan edge (search edge 3) 4 [619] 619 innerTail111_14 best = best := by
  rw [innerTail111_14, scan, if_neg (by decide), innerCandidates111_14_checked]
  change scan edge (search edge 3) 4 [619] 619 innerTail111_15
    (if 3 ≤ innerCandidates111_14.length then search edge 3 [619, 7459] innerCandidates111_14 8078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan111_15

def innerTail111_13 : List Nat := 7309 :: innerTail111_14
def innerCandidates111_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates111_13_checked : candidates edge 4 619 7309 innerTail111_14 best.bound = innerCandidates111_13 := by
  unfold candidates
  have row : edge 7309 = SearchData.row7309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan111_13 : scan edge (search edge 3) 4 [619] 619 innerTail111_13 best = best := by
  rw [innerTail111_13, scan, if_neg (by decide), innerCandidates111_13_checked]
  change scan edge (search edge 3) 4 [619] 619 innerTail111_14
    (if 3 ≤ innerCandidates111_13.length then search edge 3 [619, 7309] innerCandidates111_13 7928 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan111_14

def innerTail111_12 : List Nat := 7207 :: innerTail111_13
def innerCandidates111_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates111_12_checked : candidates edge 4 619 7207 innerTail111_13 best.bound = innerCandidates111_12 := by
  unfold candidates
  have row : edge 7207 = SearchData.row7207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan111_12 : scan edge (search edge 3) 4 [619] 619 innerTail111_12 best = best := by
  rw [innerTail111_12, scan, if_neg (by decide), innerCandidates111_12_checked]
  change scan edge (search edge 3) 4 [619] 619 innerTail111_13
    (if 3 ≤ innerCandidates111_12.length then search edge 3 [619, 7207] innerCandidates111_12 7826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan111_13

def innerTail111_11 : List Nat := 6991 :: innerTail111_12
def innerCandidates111_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates111_11_checked : candidates edge 4 619 6991 innerTail111_12 best.bound = innerCandidates111_11 := by
  unfold candidates
  have row : edge 6991 = SearchData.row6991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan111_11 : scan edge (search edge 3) 4 [619] 619 innerTail111_11 best = best := by
  rw [innerTail111_11, scan, if_neg (by decide), innerCandidates111_11_checked]
  change scan edge (search edge 3) 4 [619] 619 innerTail111_12
    (if 3 ≤ innerCandidates111_11.length then search edge 3 [619, 6991] innerCandidates111_11 7610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan111_12

def innerTail111_10 : List Nat := 5419 :: innerTail111_11
def innerCandidates111_10 : List Nat := [7687]
set_option maxRecDepth 4096 in
theorem innerCandidates111_10_checked : candidates edge 4 619 5419 innerTail111_11 best.bound = innerCandidates111_10 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan111_10 : scan edge (search edge 3) 4 [619] 619 innerTail111_10 best = best := by
  rw [innerTail111_10, scan, if_neg (by decide), innerCandidates111_10_checked]
  change scan edge (search edge 3) 4 [619] 619 innerTail111_11
    (if 3 ≤ innerCandidates111_10.length then search edge 3 [619, 5419] innerCandidates111_10 6038 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan111_11

def innerTail111_9 : List Nat := 4621 :: innerTail111_10
def innerCandidates111_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates111_9_checked : candidates edge 4 619 4621 innerTail111_10 best.bound = innerCandidates111_9 := by
  unfold candidates
  have row : edge 4621 = SearchData.row4621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan111_9 : scan edge (search edge 3) 4 [619] 619 innerTail111_9 best = best := by
  rw [innerTail111_9, scan, if_neg (by decide), innerCandidates111_9_checked]
  change scan edge (search edge 3) 4 [619] 619 innerTail111_10
    (if 3 ≤ innerCandidates111_9.length then search edge 3 [619, 4621] innerCandidates111_9 5240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan111_10

def innerTail111_8 : List Nat := 3793 :: innerTail111_9
def innerCandidates111_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates111_8_checked : candidates edge 4 619 3793 innerTail111_9 best.bound = innerCandidates111_8 := by
  unfold candidates
  have row : edge 3793 = SearchData.row3793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan111_8 : scan edge (search edge 3) 4 [619] 619 innerTail111_8 best = best := by
  rw [innerTail111_8, scan, if_neg (by decide), innerCandidates111_8_checked]
  change scan edge (search edge 3) 4 [619] 619 innerTail111_9
    (if 3 ≤ innerCandidates111_8.length then search edge 3 [619, 3793] innerCandidates111_8 4412 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan111_9

def innerTail111_7 : List Nat := 3571 :: innerTail111_8
def innerCandidates111_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates111_7_checked : candidates edge 4 619 3571 innerTail111_8 best.bound = innerCandidates111_7 := by
  unfold candidates
  have row : edge 3571 = SearchData.row3571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan111_7 : scan edge (search edge 3) 4 [619] 619 innerTail111_7 best = best := by
  rw [innerTail111_7, scan, if_neg (by decide), innerCandidates111_7_checked]
  change scan edge (search edge 3) 4 [619] 619 innerTail111_8
    (if 3 ≤ innerCandidates111_7.length then search edge 3 [619, 3571] innerCandidates111_7 4190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan111_8

def innerTail111_6 : List Nat := 2707 :: innerTail111_7
def innerCandidates111_6 : List Nat := [3793]
set_option maxRecDepth 4096 in
theorem innerCandidates111_6_checked : candidates edge 4 619 2707 innerTail111_7 best.bound = innerCandidates111_6 := by
  unfold candidates
  have row : edge 2707 = SearchData.row2707.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan111_6 : scan edge (search edge 3) 4 [619] 619 innerTail111_6 best = best := by
  rw [innerTail111_6, scan, if_neg (by decide), innerCandidates111_6_checked]
  change scan edge (search edge 3) 4 [619] 619 innerTail111_7
    (if 3 ≤ innerCandidates111_6.length then search edge 3 [619, 2707] innerCandidates111_6 3326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan111_7

def innerTail111_5 : List Nat := 2671 :: innerTail111_6
def innerCandidates111_5 : List Nat := [3571]
set_option maxRecDepth 4096 in
theorem innerCandidates111_5_checked : candidates edge 4 619 2671 innerTail111_6 best.bound = innerCandidates111_5 := by
  unfold candidates
  have row : edge 2671 = SearchData.row2671.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan111_5 : scan edge (search edge 3) 4 [619] 619 innerTail111_5 best = best := by
  rw [innerTail111_5, scan, if_neg (by decide), innerCandidates111_5_checked]
  change scan edge (search edge 3) 4 [619] 619 innerTail111_6
    (if 3 ≤ innerCandidates111_5.length then search edge 3 [619, 2671] innerCandidates111_5 3290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan111_6

def innerTail111_4 : List Nat := 1747 :: innerTail111_5
def innerCandidates111_4 : List Nat := [2671]
set_option maxRecDepth 4096 in
theorem innerCandidates111_4_checked : candidates edge 4 619 1747 innerTail111_5 best.bound = innerCandidates111_4 := by
  unfold candidates
  have row : edge 1747 = PrimePairCache.smallRow1747.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan111_4 : scan edge (search edge 3) 4 [619] 619 innerTail111_4 best = best := by
  rw [innerTail111_4, scan, if_neg (by decide), innerCandidates111_4_checked]
  change scan edge (search edge 3) 4 [619] 619 innerTail111_5
    (if 3 ≤ innerCandidates111_4.length then search edge 3 [619, 1747] innerCandidates111_4 2366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan111_5

def innerTail111_3 : List Nat := 1597 :: innerTail111_4
def innerCandidates111_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates111_3_checked : candidates edge 4 619 1597 innerTail111_4 best.bound = innerCandidates111_3 := by
  unfold candidates
  have row : edge 1597 = PrimePairCache.smallRow1597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan111_3 : scan edge (search edge 3) 4 [619] 619 innerTail111_3 best = best := by
  rw [innerTail111_3, scan, if_neg (by decide), innerCandidates111_3_checked]
  change scan edge (search edge 3) 4 [619] 619 innerTail111_4
    (if 3 ≤ innerCandidates111_3.length then search edge 3 [619, 1597] innerCandidates111_3 2216 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan111_4

def innerTail111_2 : List Nat := 1579 :: innerTail111_3
def innerCandidates111_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates111_2_checked : candidates edge 4 619 1579 innerTail111_3 best.bound = innerCandidates111_2 := by
  unfold candidates
  have row : edge 1579 = PrimePairCache.smallRow1579.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan111_2 : scan edge (search edge 3) 4 [619] 619 innerTail111_2 best = best := by
  rw [innerTail111_2, scan, if_neg (by decide), innerCandidates111_2_checked]
  change scan edge (search edge 3) 4 [619] 619 innerTail111_3
    (if 3 ≤ innerCandidates111_2.length then search edge 3 [619, 1579] innerCandidates111_2 2198 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan111_3

def innerTail111_1 : List Nat := 1051 :: innerTail111_2
def innerCandidates111_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates111_1_checked : candidates edge 4 619 1051 innerTail111_2 best.bound = innerCandidates111_1 := by
  unfold candidates
  have row : edge 1051 = PrimePairCache.smallRow1051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan111_1 : scan edge (search edge 3) 4 [619] 619 innerTail111_1 best = best := by
  rw [innerTail111_1, scan, if_neg (by decide), innerCandidates111_1_checked]
  change scan edge (search edge 3) 4 [619] 619 innerTail111_2
    (if 3 ≤ innerCandidates111_1.length then search edge 3 [619, 1051] innerCandidates111_1 1670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan111_2

def innerTail111_0 : List Nat := 811 :: innerTail111_1
def innerCandidates111_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates111_0_checked : candidates edge 4 619 811 innerTail111_1 best.bound = innerCandidates111_0 := by
  unfold candidates
  have row : edge 811 = PrimePairCache.smallRow811.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan111_0 : scan edge (search edge 3) 4 [619] 619 innerTail111_0 best = best := by
  rw [innerTail111_0, scan, if_neg (by decide), innerCandidates111_0_checked]
  change scan edge (search edge 3) 4 [619] 619 innerTail111_1
    (if 3 ≤ innerCandidates111_0.length then search edge 3 [619, 811] innerCandidates111_0 1430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan111_1

theorem rootBranch111 : search edge 4 [619] rootCandidates111 619 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [619] 619 innerTail111_0 best = best
  exact innerScan111_0

end ProjectEulerSolutions.PrimePairUpper.Search
