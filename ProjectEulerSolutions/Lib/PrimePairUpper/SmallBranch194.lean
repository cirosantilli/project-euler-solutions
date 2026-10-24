import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates116 : List Nat := [659, 941, 1097, 1289, 1523, 2003, 2399, 2531, 2693, 3167, 3209, 3251, 3347, 3911, 4289, 5387, 5897, 6863, 6947, 7109, 7211, 7757]
set_option maxRecDepth 4096 in
theorem rootCandidates116_checked : candidates edge 5 0 653 tail117 best.bound = rootCandidates116 := by
  unfold candidates
  have row : edge 653 = PrimePairCache.smallRow653.contains := by funext q; rfl
  rw [row]
  decide

def innerTail116_22 : List Nat := []
theorem innerScan116_22 : scan edge (search edge 3) 4 [653] 653 innerTail116_22 best = best := rfl

def innerTail116_21 : List Nat := 7757 :: innerTail116_22
def innerCandidates116_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates116_21_checked : candidates edge 4 653 7757 innerTail116_22 best.bound = innerCandidates116_21 := by
  unfold candidates
  have row : edge 7757 = SearchData.row7757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan116_21 : scan edge (search edge 3) 4 [653] 653 innerTail116_21 best = best := by
  rw [innerTail116_21, scan, if_neg (by decide), innerCandidates116_21_checked]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_22
    (if 3 ≤ innerCandidates116_21.length then search edge 3 [653, 7757] innerCandidates116_21 8410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan116_22

def innerTail116_20 : List Nat := 7211 :: innerTail116_21
def innerCandidates116_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates116_20_checked : candidates edge 4 653 7211 innerTail116_21 best.bound = innerCandidates116_20 := by
  unfold candidates
  have row : edge 7211 = SearchData.row7211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan116_20 : scan edge (search edge 3) 4 [653] 653 innerTail116_20 best = best := by
  rw [innerTail116_20, scan, if_neg (by decide), innerCandidates116_20_checked]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_21
    (if 3 ≤ innerCandidates116_20.length then search edge 3 [653, 7211] innerCandidates116_20 7864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan116_21

def innerTail116_19 : List Nat := 7109 :: innerTail116_20
def innerCandidates116_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates116_19_checked : candidates edge 4 653 7109 innerTail116_20 best.bound = innerCandidates116_19 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan116_19 : scan edge (search edge 3) 4 [653] 653 innerTail116_19 best = best := by
  rw [innerTail116_19, scan, if_neg (by decide), innerCandidates116_19_checked]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_20
    (if 3 ≤ innerCandidates116_19.length then search edge 3 [653, 7109] innerCandidates116_19 7762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan116_20

def innerTail116_18 : List Nat := 6947 :: innerTail116_19
def innerCandidates116_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates116_18_checked : candidates edge 4 653 6947 innerTail116_19 best.bound = innerCandidates116_18 := by
  unfold candidates
  have row : edge 6947 = SearchData.row6947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan116_18 : scan edge (search edge 3) 4 [653] 653 innerTail116_18 best = best := by
  rw [innerTail116_18, scan, if_neg (by decide), innerCandidates116_18_checked]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_19
    (if 3 ≤ innerCandidates116_18.length then search edge 3 [653, 6947] innerCandidates116_18 7600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan116_19

def innerTail116_17 : List Nat := 6863 :: innerTail116_18
def innerCandidates116_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates116_17_checked : candidates edge 4 653 6863 innerTail116_18 best.bound = innerCandidates116_17 := by
  unfold candidates
  have row : edge 6863 = SearchData.row6863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan116_17 : scan edge (search edge 3) 4 [653] 653 innerTail116_17 best = best := by
  rw [innerTail116_17, scan, if_neg (by decide), innerCandidates116_17_checked]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_18
    (if 3 ≤ innerCandidates116_17.length then search edge 3 [653, 6863] innerCandidates116_17 7516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan116_18

def innerTail116_16 : List Nat := 5897 :: innerTail116_17
def innerCandidates116_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates116_16_checked : candidates edge 4 653 5897 innerTail116_17 best.bound = innerCandidates116_16 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan116_16 : scan edge (search edge 3) 4 [653] 653 innerTail116_16 best = best := by
  rw [innerTail116_16, scan, if_neg (by decide), innerCandidates116_16_checked]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_17
    (if 3 ≤ innerCandidates116_16.length then search edge 3 [653, 5897] innerCandidates116_16 6550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan116_17

def innerTail116_15 : List Nat := 5387 :: innerTail116_16
def innerCandidates116_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates116_15_checked : candidates edge 4 653 5387 innerTail116_16 best.bound = innerCandidates116_15 := by
  unfold candidates
  have row : edge 5387 = SearchData.row5387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan116_15 : scan edge (search edge 3) 4 [653] 653 innerTail116_15 best = best := by
  rw [innerTail116_15, scan, if_neg (by decide), innerCandidates116_15_checked]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_16
    (if 3 ≤ innerCandidates116_15.length then search edge 3 [653, 5387] innerCandidates116_15 6040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan116_16

def innerTail116_14 : List Nat := 4289 :: innerTail116_15
def innerCandidates116_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates116_14_checked : candidates edge 4 653 4289 innerTail116_15 best.bound = innerCandidates116_14 := by
  unfold candidates
  have row : edge 4289 = SearchData.row4289.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan116_14 : scan edge (search edge 3) 4 [653] 653 innerTail116_14 best = best := by
  rw [innerTail116_14, scan, if_neg (by decide), innerCandidates116_14_checked]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_15
    (if 3 ≤ innerCandidates116_14.length then search edge 3 [653, 4289] innerCandidates116_14 4942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan116_15

def innerTail116_13 : List Nat := 3911 :: innerTail116_14
def innerCandidates116_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates116_13_checked : candidates edge 4 653 3911 innerTail116_14 best.bound = innerCandidates116_13 := by
  unfold candidates
  have row : edge 3911 = SearchData.row3911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan116_13 : scan edge (search edge 3) 4 [653] 653 innerTail116_13 best = best := by
  rw [innerTail116_13, scan, if_neg (by decide), innerCandidates116_13_checked]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_14
    (if 3 ≤ innerCandidates116_13.length then search edge 3 [653, 3911] innerCandidates116_13 4564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan116_14

def innerTail116_12 : List Nat := 3347 :: innerTail116_13
def innerCandidates116_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates116_12_checked : candidates edge 4 653 3347 innerTail116_13 best.bound = innerCandidates116_12 := by
  unfold candidates
  have row : edge 3347 = SearchData.row3347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan116_12 : scan edge (search edge 3) 4 [653] 653 innerTail116_12 best = best := by
  rw [innerTail116_12, scan, if_neg (by decide), innerCandidates116_12_checked]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_13
    (if 3 ≤ innerCandidates116_12.length then search edge 3 [653, 3347] innerCandidates116_12 4000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan116_13

def innerTail116_11 : List Nat := 3251 :: innerTail116_12
def innerCandidates116_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates116_11_checked : candidates edge 4 653 3251 innerTail116_12 best.bound = innerCandidates116_11 := by
  unfold candidates
  have row : edge 3251 = SearchData.row3251.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan116_11 : scan edge (search edge 3) 4 [653] 653 innerTail116_11 best = best := by
  rw [innerTail116_11, scan, if_neg (by decide), innerCandidates116_11_checked]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_12
    (if 3 ≤ innerCandidates116_11.length then search edge 3 [653, 3251] innerCandidates116_11 3904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan116_12

def innerTail116_10 : List Nat := 3209 :: innerTail116_11
def innerCandidates116_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates116_10_checked : candidates edge 4 653 3209 innerTail116_11 best.bound = innerCandidates116_10 := by
  unfold candidates
  have row : edge 3209 = SearchData.row3209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan116_10 : scan edge (search edge 3) 4 [653] 653 innerTail116_10 best = best := by
  rw [innerTail116_10, scan, if_neg (by decide), innerCandidates116_10_checked]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_11
    (if 3 ≤ innerCandidates116_10.length then search edge 3 [653, 3209] innerCandidates116_10 3862 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan116_11

def innerTail116_9 : List Nat := 3167 :: innerTail116_10
def innerCandidates116_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates116_9_checked : candidates edge 4 653 3167 innerTail116_10 best.bound = innerCandidates116_9 := by
  unfold candidates
  have row : edge 3167 = SearchData.row3167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan116_9 : scan edge (search edge 3) 4 [653] 653 innerTail116_9 best = best := by
  rw [innerTail116_9, scan, if_neg (by decide), innerCandidates116_9_checked]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_10
    (if 3 ≤ innerCandidates116_9.length then search edge 3 [653, 3167] innerCandidates116_9 3820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan116_10

def innerTail116_8 : List Nat := 2693 :: innerTail116_9
def innerCandidates116_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates116_8_checked : candidates edge 4 653 2693 innerTail116_9 best.bound = innerCandidates116_8 := by
  unfold candidates
  have row : edge 2693 = SearchData.row2693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan116_8 : scan edge (search edge 3) 4 [653] 653 innerTail116_8 best = best := by
  rw [innerTail116_8, scan, if_neg (by decide), innerCandidates116_8_checked]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_9
    (if 3 ≤ innerCandidates116_8.length then search edge 3 [653, 2693] innerCandidates116_8 3346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan116_9

def innerTail116_7 : List Nat := 2531 :: innerTail116_8
def innerCandidates116_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates116_7_checked : candidates edge 4 653 2531 innerTail116_8 best.bound = innerCandidates116_7 := by
  unfold candidates
  have row : edge 2531 = PrimePairCache.smallRow2531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan116_7 : scan edge (search edge 3) 4 [653] 653 innerTail116_7 best = best := by
  rw [innerTail116_7, scan, if_neg (by decide), innerCandidates116_7_checked]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_8
    (if 3 ≤ innerCandidates116_7.length then search edge 3 [653, 2531] innerCandidates116_7 3184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan116_8

def innerTail116_6 : List Nat := 2399 :: innerTail116_7
def innerCandidates116_6 : List Nat := [3251, 3911]
set_option maxRecDepth 4096 in
theorem innerCandidates116_6_checked : candidates edge 4 653 2399 innerTail116_7 best.bound = innerCandidates116_6 := by
  unfold candidates
  have row : edge 2399 = PrimePairCache.smallRow2399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan116_6 : scan edge (search edge 3) 4 [653] 653 innerTail116_6 best = best := by
  rw [innerTail116_6, scan, if_neg (by decide), innerCandidates116_6_checked]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_7
    (if 3 ≤ innerCandidates116_6.length then search edge 3 [653, 2399] innerCandidates116_6 3052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan116_7

def innerTail116_5 : List Nat := 2003 :: innerTail116_6
def innerCandidates116_5 : List Nat := [7211]
set_option maxRecDepth 4096 in
theorem innerCandidates116_5_checked : candidates edge 4 653 2003 innerTail116_6 best.bound = innerCandidates116_5 := by
  unfold candidates
  have row : edge 2003 = PrimePairCache.smallRow2003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan116_5 : scan edge (search edge 3) 4 [653] 653 innerTail116_5 best = best := by
  rw [innerTail116_5, scan, if_neg (by decide), innerCandidates116_5_checked]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_6
    (if 3 ≤ innerCandidates116_5.length then search edge 3 [653, 2003] innerCandidates116_5 2656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan116_6

def innerTail116_4 : List Nat := 1523 :: innerTail116_5
def innerCandidates116_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates116_4_checked : candidates edge 4 653 1523 innerTail116_5 best.bound = innerCandidates116_4 := by
  unfold candidates
  have row : edge 1523 = PrimePairCache.smallRow1523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan116_4 : scan edge (search edge 3) 4 [653] 653 innerTail116_4 best = best := by
  rw [innerTail116_4, scan, if_neg (by decide), innerCandidates116_4_checked]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_5
    (if 3 ≤ innerCandidates116_4.length then search edge 3 [653, 1523] innerCandidates116_4 2176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan116_5

def innerTail116_3 : List Nat := 1289 :: innerTail116_4
def innerCandidates116_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates116_3_checked : candidates edge 4 653 1289 innerTail116_4 best.bound = innerCandidates116_3 := by
  unfold candidates
  have row : edge 1289 = PrimePairCache.smallRow1289.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan116_3 : scan edge (search edge 3) 4 [653] 653 innerTail116_3 best = best := by
  rw [innerTail116_3, scan, if_neg (by decide), innerCandidates116_3_checked]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_4
    (if 3 ≤ innerCandidates116_3.length then search edge 3 [653, 1289] innerCandidates116_3 1942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan116_4

def innerTail116_2 : List Nat := 1097 :: innerTail116_3
def innerCandidates116_2 : List Nat := [3251]
set_option maxRecDepth 4096 in
theorem innerCandidates116_2_checked : candidates edge 4 653 1097 innerTail116_3 best.bound = innerCandidates116_2 := by
  unfold candidates
  have row : edge 1097 = PrimePairCache.smallRow1097.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan116_2 : scan edge (search edge 3) 4 [653] 653 innerTail116_2 best = best := by
  rw [innerTail116_2, scan, if_neg (by decide), innerCandidates116_2_checked]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_3
    (if 3 ≤ innerCandidates116_2.length then search edge 3 [653, 1097] innerCandidates116_2 1750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan116_3

def innerTail116_1 : List Nat := 941 :: innerTail116_2
def innerCandidates116_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates116_1_checked : candidates edge 4 653 941 innerTail116_2 best.bound = innerCandidates116_1 := by
  unfold candidates
  have row : edge 941 = PrimePairCache.smallRow941.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan116_1 : scan edge (search edge 3) 4 [653] 653 innerTail116_1 best = best := by
  rw [innerTail116_1, scan, if_neg (by decide), innerCandidates116_1_checked]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_2
    (if 3 ≤ innerCandidates116_1.length then search edge 3 [653, 941] innerCandidates116_1 1594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan116_2

def innerTail116_0 : List Nat := 659 :: innerTail116_1
def innerCandidates116_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates116_0_checked : candidates edge 4 653 659 innerTail116_1 best.bound = innerCandidates116_0 := by
  unfold candidates
  have row : edge 659 = PrimePairCache.smallRow659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan116_0 : scan edge (search edge 3) 4 [653] 653 innerTail116_0 best = best := by
  rw [innerTail116_0, scan, if_neg (by decide), innerCandidates116_0_checked]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_1
    (if 3 ≤ innerCandidates116_0.length then search edge 3 [653, 659] innerCandidates116_0 1312 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan116_1

theorem rootBranch116 : search edge 4 [653] rootCandidates116 653 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [653] 653 innerTail116_0 best = best
  exact innerScan116_0

def rootCandidates115 : List Nat := [821, 953, 1019, 1511, 1667, 1733, 1811, 1889, 3011, 3347, 3407, 3911, 4019, 4241, 4493, 5153, 5387, 5471, 5879, 6329, 7253, 7949, 8273]
set_option maxRecDepth 4096 in
theorem rootCandidates115_checked : candidates edge 5 0 647 tail116 best.bound = rootCandidates115 := by
  unfold candidates
  have row : edge 647 = PrimePairCache.smallRow647.contains := by funext q; rfl
  rw [row]
  decide

def innerTail115_23 : List Nat := []
theorem innerScan115_23 : scan edge (search edge 3) 4 [647] 647 innerTail115_23 best = best := rfl

def innerTail115_22 : List Nat := 8273 :: innerTail115_23
def innerCandidates115_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates115_22_checked : candidates edge 4 647 8273 innerTail115_23 best.bound = innerCandidates115_22 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_22 : scan edge (search edge 3) 4 [647] 647 innerTail115_22 best = best := by
  rw [innerTail115_22, scan, if_neg (by decide), innerCandidates115_22_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_23
    (if 3 ≤ innerCandidates115_22.length then search edge 3 [647, 8273] innerCandidates115_22 8920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_23

def innerTail115_21 : List Nat := 7949 :: innerTail115_22
def innerCandidates115_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates115_21_checked : candidates edge 4 647 7949 innerTail115_22 best.bound = innerCandidates115_21 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_21 : scan edge (search edge 3) 4 [647] 647 innerTail115_21 best = best := by
  rw [innerTail115_21, scan, if_neg (by decide), innerCandidates115_21_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_22
    (if 3 ≤ innerCandidates115_21.length then search edge 3 [647, 7949] innerCandidates115_21 8596 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_22

def innerTail115_20 : List Nat := 7253 :: innerTail115_21
def innerCandidates115_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates115_20_checked : candidates edge 4 647 7253 innerTail115_21 best.bound = innerCandidates115_20 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_20 : scan edge (search edge 3) 4 [647] 647 innerTail115_20 best = best := by
  rw [innerTail115_20, scan, if_neg (by decide), innerCandidates115_20_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_21
    (if 3 ≤ innerCandidates115_20.length then search edge 3 [647, 7253] innerCandidates115_20 7900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_21

def innerTail115_19 : List Nat := 6329 :: innerTail115_20
def innerCandidates115_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates115_19_checked : candidates edge 4 647 6329 innerTail115_20 best.bound = innerCandidates115_19 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_19 : scan edge (search edge 3) 4 [647] 647 innerTail115_19 best = best := by
  rw [innerTail115_19, scan, if_neg (by decide), innerCandidates115_19_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_20
    (if 3 ≤ innerCandidates115_19.length then search edge 3 [647, 6329] innerCandidates115_19 6976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_20

def innerTail115_18 : List Nat := 5879 :: innerTail115_19
def innerCandidates115_18 : List Nat := [8273]
set_option maxRecDepth 4096 in
theorem innerCandidates115_18_checked : candidates edge 4 647 5879 innerTail115_19 best.bound = innerCandidates115_18 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_18 : scan edge (search edge 3) 4 [647] 647 innerTail115_18 best = best := by
  rw [innerTail115_18, scan, if_neg (by decide), innerCandidates115_18_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_19
    (if 3 ≤ innerCandidates115_18.length then search edge 3 [647, 5879] innerCandidates115_18 6526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_19

def innerTail115_17 : List Nat := 5471 :: innerTail115_18
def innerCandidates115_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates115_17_checked : candidates edge 4 647 5471 innerTail115_18 best.bound = innerCandidates115_17 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_17 : scan edge (search edge 3) 4 [647] 647 innerTail115_17 best = best := by
  rw [innerTail115_17, scan, if_neg (by decide), innerCandidates115_17_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_18
    (if 3 ≤ innerCandidates115_17.length then search edge 3 [647, 5471] innerCandidates115_17 6118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_18

def innerTail115_16 : List Nat := 5387 :: innerTail115_17
def innerCandidates115_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates115_16_checked : candidates edge 4 647 5387 innerTail115_17 best.bound = innerCandidates115_16 := by
  unfold candidates
  have row : edge 5387 = SearchData.row5387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_16 : scan edge (search edge 3) 4 [647] 647 innerTail115_16 best = best := by
  rw [innerTail115_16, scan, if_neg (by decide), innerCandidates115_16_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_17
    (if 3 ≤ innerCandidates115_16.length then search edge 3 [647, 5387] innerCandidates115_16 6034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_17

def innerTail115_15 : List Nat := 5153 :: innerTail115_16
def innerCandidates115_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates115_15_checked : candidates edge 4 647 5153 innerTail115_16 best.bound = innerCandidates115_15 := by
  unfold candidates
  have row : edge 5153 = SearchData.row5153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_15 : scan edge (search edge 3) 4 [647] 647 innerTail115_15 best = best := by
  rw [innerTail115_15, scan, if_neg (by decide), innerCandidates115_15_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_16
    (if 3 ≤ innerCandidates115_15.length then search edge 3 [647, 5153] innerCandidates115_15 5800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_16

def innerTail115_14 : List Nat := 4493 :: innerTail115_15
def innerCandidates115_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates115_14_checked : candidates edge 4 647 4493 innerTail115_15 best.bound = innerCandidates115_14 := by
  unfold candidates
  have row : edge 4493 = SearchData.row4493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_14 : scan edge (search edge 3) 4 [647] 647 innerTail115_14 best = best := by
  rw [innerTail115_14, scan, if_neg (by decide), innerCandidates115_14_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_15
    (if 3 ≤ innerCandidates115_14.length then search edge 3 [647, 4493] innerCandidates115_14 5140 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_15

def innerTail115_13 : List Nat := 4241 :: innerTail115_14
def innerCandidates115_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates115_13_checked : candidates edge 4 647 4241 innerTail115_14 best.bound = innerCandidates115_13 := by
  unfold candidates
  have row : edge 4241 = SearchData.row4241.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_13 : scan edge (search edge 3) 4 [647] 647 innerTail115_13 best = best := by
  rw [innerTail115_13, scan, if_neg (by decide), innerCandidates115_13_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_14
    (if 3 ≤ innerCandidates115_13.length then search edge 3 [647, 4241] innerCandidates115_13 4888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_14

def innerTail115_12 : List Nat := 4019 :: innerTail115_13
def innerCandidates115_12 : List Nat := [5471]
set_option maxRecDepth 4096 in
theorem innerCandidates115_12_checked : candidates edge 4 647 4019 innerTail115_13 best.bound = innerCandidates115_12 := by
  unfold candidates
  have row : edge 4019 = SearchData.row4019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_12 : scan edge (search edge 3) 4 [647] 647 innerTail115_12 best = best := by
  rw [innerTail115_12, scan, if_neg (by decide), innerCandidates115_12_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_13
    (if 3 ≤ innerCandidates115_12.length then search edge 3 [647, 4019] innerCandidates115_12 4666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_13

def innerTail115_11 : List Nat := 3911 :: innerTail115_12
def innerCandidates115_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates115_11_checked : candidates edge 4 647 3911 innerTail115_12 best.bound = innerCandidates115_11 := by
  unfold candidates
  have row : edge 3911 = SearchData.row3911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_11 : scan edge (search edge 3) 4 [647] 647 innerTail115_11 best = best := by
  rw [innerTail115_11, scan, if_neg (by decide), innerCandidates115_11_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_12
    (if 3 ≤ innerCandidates115_11.length then search edge 3 [647, 3911] innerCandidates115_11 4558 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_12

def innerTail115_10 : List Nat := 3407 :: innerTail115_11
def innerCandidates115_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates115_10_checked : candidates edge 4 647 3407 innerTail115_11 best.bound = innerCandidates115_10 := by
  unfold candidates
  have row : edge 3407 = SearchData.row3407.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_10 : scan edge (search edge 3) 4 [647] 647 innerTail115_10 best = best := by
  rw [innerTail115_10, scan, if_neg (by decide), innerCandidates115_10_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_11
    (if 3 ≤ innerCandidates115_10.length then search edge 3 [647, 3407] innerCandidates115_10 4054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_11

def innerTail115_9 : List Nat := 3347 :: innerTail115_10
def innerCandidates115_9 : List Nat := [8273]
set_option maxRecDepth 4096 in
theorem innerCandidates115_9_checked : candidates edge 4 647 3347 innerTail115_10 best.bound = innerCandidates115_9 := by
  unfold candidates
  have row : edge 3347 = SearchData.row3347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_9 : scan edge (search edge 3) 4 [647] 647 innerTail115_9 best = best := by
  rw [innerTail115_9, scan, if_neg (by decide), innerCandidates115_9_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_10
    (if 3 ≤ innerCandidates115_9.length then search edge 3 [647, 3347] innerCandidates115_9 3994 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_10

def innerTail115_8 : List Nat := 3011 :: innerTail115_9
def innerCandidates115_8 : List Nat := [4493, 6329]
set_option maxRecDepth 4096 in
theorem innerCandidates115_8_checked : candidates edge 4 647 3011 innerTail115_9 best.bound = innerCandidates115_8 := by
  unfold candidates
  have row : edge 3011 = SearchData.row3011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_8 : scan edge (search edge 3) 4 [647] 647 innerTail115_8 best = best := by
  rw [innerTail115_8, scan, if_neg (by decide), innerCandidates115_8_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_9
    (if 3 ≤ innerCandidates115_8.length then search edge 3 [647, 3011] innerCandidates115_8 3658 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_9

def innerTail115_7 : List Nat := 1889 :: innerTail115_8
def innerCandidates115_7 : List Nat := [4019]
set_option maxRecDepth 4096 in
theorem innerCandidates115_7_checked : candidates edge 4 647 1889 innerTail115_8 best.bound = innerCandidates115_7 := by
  unfold candidates
  have row : edge 1889 = PrimePairCache.smallRow1889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_7 : scan edge (search edge 3) 4 [647] 647 innerTail115_7 best = best := by
  rw [innerTail115_7, scan, if_neg (by decide), innerCandidates115_7_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_8
    (if 3 ≤ innerCandidates115_7.length then search edge 3 [647, 1889] innerCandidates115_7 2536 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_8

def innerTail115_6 : List Nat := 1811 :: innerTail115_7
def innerCandidates115_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates115_6_checked : candidates edge 4 647 1811 innerTail115_7 best.bound = innerCandidates115_6 := by
  unfold candidates
  have row : edge 1811 = PrimePairCache.smallRow1811.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_6 : scan edge (search edge 3) 4 [647] 647 innerTail115_6 best = best := by
  rw [innerTail115_6, scan, if_neg (by decide), innerCandidates115_6_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_7
    (if 3 ≤ innerCandidates115_6.length then search edge 3 [647, 1811] innerCandidates115_6 2458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_7

def innerTail115_5 : List Nat := 1733 :: innerTail115_6
def innerCandidates115_5 : List Nat := [3347, 7949]
set_option maxRecDepth 4096 in
theorem innerCandidates115_5_checked : candidates edge 4 647 1733 innerTail115_6 best.bound = innerCandidates115_5 := by
  unfold candidates
  have row : edge 1733 = PrimePairCache.smallRow1733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_5 : scan edge (search edge 3) 4 [647] 647 innerTail115_5 best = best := by
  rw [innerTail115_5, scan, if_neg (by decide), innerCandidates115_5_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_6
    (if 3 ≤ innerCandidates115_5.length then search edge 3 [647, 1733] innerCandidates115_5 2380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_6

def innerTail115_4 : List Nat := 1667 :: innerTail115_5
def innerCandidates115_4 : List Nat := [6329]
set_option maxRecDepth 4096 in
theorem innerCandidates115_4_checked : candidates edge 4 647 1667 innerTail115_5 best.bound = innerCandidates115_4 := by
  unfold candidates
  have row : edge 1667 = PrimePairCache.smallRow1667.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_4 : scan edge (search edge 3) 4 [647] 647 innerTail115_4 best = best := by
  rw [innerTail115_4, scan, if_neg (by decide), innerCandidates115_4_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_5
    (if 3 ≤ innerCandidates115_4.length then search edge 3 [647, 1667] innerCandidates115_4 2314 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_5

def innerTail115_3 : List Nat := 1511 :: innerTail115_4
def innerCandidates115_3 : List Nat := [3011]
set_option maxRecDepth 4096 in
theorem innerCandidates115_3_checked : candidates edge 4 647 1511 innerTail115_4 best.bound = innerCandidates115_3 := by
  unfold candidates
  have row : edge 1511 = PrimePairCache.smallRow1511.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_3 : scan edge (search edge 3) 4 [647] 647 innerTail115_3 best = best := by
  rw [innerTail115_3, scan, if_neg (by decide), innerCandidates115_3_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_4
    (if 3 ≤ innerCandidates115_3.length then search edge 3 [647, 1511] innerCandidates115_3 2158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_4

def innerTail115_2 : List Nat := 1019 :: innerTail115_3
def innerCandidates115_2 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates115_2_checked : candidates edge 4 647 1019 innerTail115_3 best.bound = innerCandidates115_2 := by
  unfold candidates
  have row : edge 1019 = PrimePairCache.smallRow1019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_2 : scan edge (search edge 3) 4 [647] 647 innerTail115_2 best = best := by
  rw [innerTail115_2, scan, if_neg (by decide), innerCandidates115_2_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_3
    (if 3 ≤ innerCandidates115_2.length then search edge 3 [647, 1019] innerCandidates115_2 1666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_3

def innerTail115_1 : List Nat := 953 :: innerTail115_2
def innerCandidates115_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates115_1_checked : candidates edge 4 647 953 innerTail115_2 best.bound = innerCandidates115_1 := by
  unfold candidates
  have row : edge 953 = PrimePairCache.smallRow953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_1 : scan edge (search edge 3) 4 [647] 647 innerTail115_1 best = best := by
  rw [innerTail115_1, scan, if_neg (by decide), innerCandidates115_1_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_2
    (if 3 ≤ innerCandidates115_1.length then search edge 3 [647, 953] innerCandidates115_1 1600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_2

def innerTail115_0 : List Nat := 821 :: innerTail115_1
def innerCandidates115_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates115_0_checked : candidates edge 4 647 821 innerTail115_1 best.bound = innerCandidates115_0 := by
  unfold candidates
  have row : edge 821 = PrimePairCache.smallRow821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan115_0 : scan edge (search edge 3) 4 [647] 647 innerTail115_0 best = best := by
  rw [innerTail115_0, scan, if_neg (by decide), innerCandidates115_0_checked]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_1
    (if 3 ≤ innerCandidates115_0.length then search edge 3 [647, 821] innerCandidates115_0 1468 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan115_1

theorem rootBranch115 : search edge 4 [647] rootCandidates115 647 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [647] 647 innerTail115_0 best = best
  exact innerScan115_0

def rootCandidates114 : List Nat := [751, 991, 1213, 1993, 2017, 2053, 2383, 2917, 4111, 5119, 5413, 5563, 6007, 6037, 7393, 7477, 8059, 8191]
set_option maxRecDepth 4096 in
theorem rootCandidates114_checked : candidates edge 5 0 643 tail115 best.bound = rootCandidates114 := by
  unfold candidates
  have row : edge 643 = PrimePairCache.smallRow643.contains := by funext q; rfl
  rw [row]
  decide

def innerTail114_18 : List Nat := []
theorem innerScan114_18 : scan edge (search edge 3) 4 [643] 643 innerTail114_18 best = best := rfl

def innerTail114_17 : List Nat := 8191 :: innerTail114_18
def innerCandidates114_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates114_17_checked : candidates edge 4 643 8191 innerTail114_18 best.bound = innerCandidates114_17 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan114_17 : scan edge (search edge 3) 4 [643] 643 innerTail114_17 best = best := by
  rw [innerTail114_17, scan, if_neg (by decide), innerCandidates114_17_checked]
  change scan edge (search edge 3) 4 [643] 643 innerTail114_18
    (if 3 ≤ innerCandidates114_17.length then search edge 3 [643, 8191] innerCandidates114_17 8834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan114_18

def innerTail114_16 : List Nat := 8059 :: innerTail114_17
def innerCandidates114_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates114_16_checked : candidates edge 4 643 8059 innerTail114_17 best.bound = innerCandidates114_16 := by
  unfold candidates
  have row : edge 8059 = SearchData.row8059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan114_16 : scan edge (search edge 3) 4 [643] 643 innerTail114_16 best = best := by
  rw [innerTail114_16, scan, if_neg (by decide), innerCandidates114_16_checked]
  change scan edge (search edge 3) 4 [643] 643 innerTail114_17
    (if 3 ≤ innerCandidates114_16.length then search edge 3 [643, 8059] innerCandidates114_16 8702 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan114_17

def innerTail114_15 : List Nat := 7477 :: innerTail114_16
def innerCandidates114_15 : List Nat := [8191]
set_option maxRecDepth 4096 in
theorem innerCandidates114_15_checked : candidates edge 4 643 7477 innerTail114_16 best.bound = innerCandidates114_15 := by
  unfold candidates
  have row : edge 7477 = SearchData.row7477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan114_15 : scan edge (search edge 3) 4 [643] 643 innerTail114_15 best = best := by
  rw [innerTail114_15, scan, if_neg (by decide), innerCandidates114_15_checked]
  change scan edge (search edge 3) 4 [643] 643 innerTail114_16
    (if 3 ≤ innerCandidates114_15.length then search edge 3 [643, 7477] innerCandidates114_15 8120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan114_16

def innerTail114_14 : List Nat := 7393 :: innerTail114_15
def innerCandidates114_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates114_14_checked : candidates edge 4 643 7393 innerTail114_15 best.bound = innerCandidates114_14 := by
  unfold candidates
  have row : edge 7393 = SearchData.row7393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan114_14 : scan edge (search edge 3) 4 [643] 643 innerTail114_14 best = best := by
  rw [innerTail114_14, scan, if_neg (by decide), innerCandidates114_14_checked]
  change scan edge (search edge 3) 4 [643] 643 innerTail114_15
    (if 3 ≤ innerCandidates114_14.length then search edge 3 [643, 7393] innerCandidates114_14 8036 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan114_15

def innerTail114_13 : List Nat := 6037 :: innerTail114_14
def innerCandidates114_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates114_13_checked : candidates edge 4 643 6037 innerTail114_14 best.bound = innerCandidates114_13 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan114_13 : scan edge (search edge 3) 4 [643] 643 innerTail114_13 best = best := by
  rw [innerTail114_13, scan, if_neg (by decide), innerCandidates114_13_checked]
  change scan edge (search edge 3) 4 [643] 643 innerTail114_14
    (if 3 ≤ innerCandidates114_13.length then search edge 3 [643, 6037] innerCandidates114_13 6680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan114_14

def innerTail114_12 : List Nat := 6007 :: innerTail114_13
def innerCandidates114_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates114_12_checked : candidates edge 4 643 6007 innerTail114_13 best.bound = innerCandidates114_12 := by
  unfold candidates
  have row : edge 6007 = SearchData.row6007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan114_12 : scan edge (search edge 3) 4 [643] 643 innerTail114_12 best = best := by
  rw [innerTail114_12, scan, if_neg (by decide), innerCandidates114_12_checked]
  change scan edge (search edge 3) 4 [643] 643 innerTail114_13
    (if 3 ≤ innerCandidates114_12.length then search edge 3 [643, 6007] innerCandidates114_12 6650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan114_13

def innerTail114_11 : List Nat := 5563 :: innerTail114_12
def innerCandidates114_11 : List Nat := [6037]
set_option maxRecDepth 4096 in
theorem innerCandidates114_11_checked : candidates edge 4 643 5563 innerTail114_12 best.bound = innerCandidates114_11 := by
  unfold candidates
  have row : edge 5563 = SearchData.row5563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan114_11 : scan edge (search edge 3) 4 [643] 643 innerTail114_11 best = best := by
  rw [innerTail114_11, scan, if_neg (by decide), innerCandidates114_11_checked]
  change scan edge (search edge 3) 4 [643] 643 innerTail114_12
    (if 3 ≤ innerCandidates114_11.length then search edge 3 [643, 5563] innerCandidates114_11 6206 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan114_12

def innerTail114_10 : List Nat := 5413 :: innerTail114_11
def innerCandidates114_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates114_10_checked : candidates edge 4 643 5413 innerTail114_11 best.bound = innerCandidates114_10 := by
  unfold candidates
  have row : edge 5413 = SearchData.row5413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan114_10 : scan edge (search edge 3) 4 [643] 643 innerTail114_10 best = best := by
  rw [innerTail114_10, scan, if_neg (by decide), innerCandidates114_10_checked]
  change scan edge (search edge 3) 4 [643] 643 innerTail114_11
    (if 3 ≤ innerCandidates114_10.length then search edge 3 [643, 5413] innerCandidates114_10 6056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan114_11

def innerTail114_9 : List Nat := 5119 :: innerTail114_10
def innerCandidates114_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates114_9_checked : candidates edge 4 643 5119 innerTail114_10 best.bound = innerCandidates114_9 := by
  unfold candidates
  have row : edge 5119 = SearchData.row5119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan114_9 : scan edge (search edge 3) 4 [643] 643 innerTail114_9 best = best := by
  rw [innerTail114_9, scan, if_neg (by decide), innerCandidates114_9_checked]
  change scan edge (search edge 3) 4 [643] 643 innerTail114_10
    (if 3 ≤ innerCandidates114_9.length then search edge 3 [643, 5119] innerCandidates114_9 5762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan114_10

def innerTail114_8 : List Nat := 4111 :: innerTail114_9
def innerCandidates114_8 : List Nat := [5119]
set_option maxRecDepth 4096 in
theorem innerCandidates114_8_checked : candidates edge 4 643 4111 innerTail114_9 best.bound = innerCandidates114_8 := by
  unfold candidates
  have row : edge 4111 = SearchData.row4111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan114_8 : scan edge (search edge 3) 4 [643] 643 innerTail114_8 best = best := by
  rw [innerTail114_8, scan, if_neg (by decide), innerCandidates114_8_checked]
  change scan edge (search edge 3) 4 [643] 643 innerTail114_9
    (if 3 ≤ innerCandidates114_8.length then search edge 3 [643, 4111] innerCandidates114_8 4754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan114_9

def innerTail114_7 : List Nat := 2917 :: innerTail114_8
def innerCandidates114_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates114_7_checked : candidates edge 4 643 2917 innerTail114_8 best.bound = innerCandidates114_7 := by
  unfold candidates
  have row : edge 2917 = SearchData.row2917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan114_7 : scan edge (search edge 3) 4 [643] 643 innerTail114_7 best = best := by
  rw [innerTail114_7, scan, if_neg (by decide), innerCandidates114_7_checked]
  change scan edge (search edge 3) 4 [643] 643 innerTail114_8
    (if 3 ≤ innerCandidates114_7.length then search edge 3 [643, 2917] innerCandidates114_7 3560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan114_8

def innerTail114_6 : List Nat := 2383 :: innerTail114_7
def innerCandidates114_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates114_6_checked : candidates edge 4 643 2383 innerTail114_7 best.bound = innerCandidates114_6 := by
  unfold candidates
  have row : edge 2383 = PrimePairCache.smallRow2383.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan114_6 : scan edge (search edge 3) 4 [643] 643 innerTail114_6 best = best := by
  rw [innerTail114_6, scan, if_neg (by decide), innerCandidates114_6_checked]
  change scan edge (search edge 3) 4 [643] 643 innerTail114_7
    (if 3 ≤ innerCandidates114_6.length then search edge 3 [643, 2383] innerCandidates114_6 3026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan114_7

def innerTail114_5 : List Nat := 2053 :: innerTail114_6
def innerCandidates114_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates114_5_checked : candidates edge 4 643 2053 innerTail114_6 best.bound = innerCandidates114_5 := by
  unfold candidates
  have row : edge 2053 = PrimePairCache.smallRow2053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan114_5 : scan edge (search edge 3) 4 [643] 643 innerTail114_5 best = best := by
  rw [innerTail114_5, scan, if_neg (by decide), innerCandidates114_5_checked]
  change scan edge (search edge 3) 4 [643] 643 innerTail114_6
    (if 3 ≤ innerCandidates114_5.length then search edge 3 [643, 2053] innerCandidates114_5 2696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan114_6

def innerTail114_4 : List Nat := 2017 :: innerTail114_5
def innerCandidates114_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates114_4_checked : candidates edge 4 643 2017 innerTail114_5 best.bound = innerCandidates114_4 := by
  unfold candidates
  have row : edge 2017 = PrimePairCache.smallRow2017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan114_4 : scan edge (search edge 3) 4 [643] 643 innerTail114_4 best = best := by
  rw [innerTail114_4, scan, if_neg (by decide), innerCandidates114_4_checked]
  change scan edge (search edge 3) 4 [643] 643 innerTail114_5
    (if 3 ≤ innerCandidates114_4.length then search edge 3 [643, 2017] innerCandidates114_4 2660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan114_5

def innerTail114_3 : List Nat := 1993 :: innerTail114_4
def innerCandidates114_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates114_3_checked : candidates edge 4 643 1993 innerTail114_4 best.bound = innerCandidates114_3 := by
  unfold candidates
  have row : edge 1993 = PrimePairCache.smallRow1993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan114_3 : scan edge (search edge 3) 4 [643] 643 innerTail114_3 best = best := by
  rw [innerTail114_3, scan, if_neg (by decide), innerCandidates114_3_checked]
  change scan edge (search edge 3) 4 [643] 643 innerTail114_4
    (if 3 ≤ innerCandidates114_3.length then search edge 3 [643, 1993] innerCandidates114_3 2636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan114_4

def innerTail114_2 : List Nat := 1213 :: innerTail114_3
def innerCandidates114_2 : List Nat := [2053]
set_option maxRecDepth 4096 in
theorem innerCandidates114_2_checked : candidates edge 4 643 1213 innerTail114_3 best.bound = innerCandidates114_2 := by
  unfold candidates
  have row : edge 1213 = PrimePairCache.smallRow1213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan114_2 : scan edge (search edge 3) 4 [643] 643 innerTail114_2 best = best := by
  rw [innerTail114_2, scan, if_neg (by decide), innerCandidates114_2_checked]
  change scan edge (search edge 3) 4 [643] 643 innerTail114_3
    (if 3 ≤ innerCandidates114_2.length then search edge 3 [643, 1213] innerCandidates114_2 1856 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan114_3

def innerTail114_1 : List Nat := 991 :: innerTail114_2
def innerCandidates114_1 : List Nat := [2017]
set_option maxRecDepth 4096 in
theorem innerCandidates114_1_checked : candidates edge 4 643 991 innerTail114_2 best.bound = innerCandidates114_1 := by
  unfold candidates
  have row : edge 991 = PrimePairCache.smallRow991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan114_1 : scan edge (search edge 3) 4 [643] 643 innerTail114_1 best = best := by
  rw [innerTail114_1, scan, if_neg (by decide), innerCandidates114_1_checked]
  change scan edge (search edge 3) 4 [643] 643 innerTail114_2
    (if 3 ≤ innerCandidates114_1.length then search edge 3 [643, 991] innerCandidates114_1 1634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan114_2

def innerTail114_0 : List Nat := 751 :: innerTail114_1
def innerCandidates114_0 : List Nat := [2383]
set_option maxRecDepth 4096 in
theorem innerCandidates114_0_checked : candidates edge 4 643 751 innerTail114_1 best.bound = innerCandidates114_0 := by
  unfold candidates
  have row : edge 751 = PrimePairCache.smallRow751.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan114_0 : scan edge (search edge 3) 4 [643] 643 innerTail114_0 best = best := by
  rw [innerTail114_0, scan, if_neg (by decide), innerCandidates114_0_checked]
  change scan edge (search edge 3) 4 [643] 643 innerTail114_1
    (if 3 ≤ innerCandidates114_0.length then search edge 3 [643, 751] innerCandidates114_0 1394 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan114_1

theorem rootBranch114 : search edge 4 [643] rootCandidates114 643 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [643] 643 innerTail114_0 best = best
  exact innerScan114_0

end ProjectEulerSolutions.PrimePairUpper.Search
