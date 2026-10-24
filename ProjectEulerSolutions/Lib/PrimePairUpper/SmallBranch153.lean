import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates217 : List Nat := [1439, 1451, 1499, 1511, 1637, 1901, 2423, 2531, 2591, 3023, 3539, 4493, 5333, 6143, 6491, 6803, 7541, 7727, 7907, 7937]
set_option maxRecDepth 4096 in
theorem rootCandidates217_checked : candidates edge 5 0 1373 tail218 best.bound = rootCandidates217 := by
  unfold candidates
  have row : edge 1373 = PrimePairCache.smallRow1373.contains := by funext q; rfl
  rw [row]
  decide

def innerTail217_20 : List Nat := []
theorem innerScan217_20 : scan edge (search edge 3) 4 [1373] 1373 innerTail217_20 best = best := rfl

def innerTail217_19 : List Nat := 7937 :: innerTail217_20
def innerCandidates217_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates217_19_checked : candidates edge 4 1373 7937 innerTail217_20 best.bound = innerCandidates217_19 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan217_19 : scan edge (search edge 3) 4 [1373] 1373 innerTail217_19 best = best := by
  rw [innerTail217_19, scan, if_neg (by decide), innerCandidates217_19_checked]
  change scan edge (search edge 3) 4 [1373] 1373 innerTail217_20
    (if 3 ≤ innerCandidates217_19.length then search edge 3 [1373, 7937] innerCandidates217_19 9310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan217_20

def innerTail217_18 : List Nat := 7907 :: innerTail217_19
def innerCandidates217_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates217_18_checked : candidates edge 4 1373 7907 innerTail217_19 best.bound = innerCandidates217_18 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan217_18 : scan edge (search edge 3) 4 [1373] 1373 innerTail217_18 best = best := by
  rw [innerTail217_18, scan, if_neg (by decide), innerCandidates217_18_checked]
  change scan edge (search edge 3) 4 [1373] 1373 innerTail217_19
    (if 3 ≤ innerCandidates217_18.length then search edge 3 [1373, 7907] innerCandidates217_18 9280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan217_19

def innerTail217_17 : List Nat := 7727 :: innerTail217_18
def innerCandidates217_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates217_17_checked : candidates edge 4 1373 7727 innerTail217_18 best.bound = innerCandidates217_17 := by
  unfold candidates
  have row : edge 7727 = SearchData.row7727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan217_17 : scan edge (search edge 3) 4 [1373] 1373 innerTail217_17 best = best := by
  rw [innerTail217_17, scan, if_neg (by decide), innerCandidates217_17_checked]
  change scan edge (search edge 3) 4 [1373] 1373 innerTail217_18
    (if 3 ≤ innerCandidates217_17.length then search edge 3 [1373, 7727] innerCandidates217_17 9100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan217_18

def innerTail217_16 : List Nat := 7541 :: innerTail217_17
def innerCandidates217_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates217_16_checked : candidates edge 4 1373 7541 innerTail217_17 best.bound = innerCandidates217_16 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan217_16 : scan edge (search edge 3) 4 [1373] 1373 innerTail217_16 best = best := by
  rw [innerTail217_16, scan, if_neg (by decide), innerCandidates217_16_checked]
  change scan edge (search edge 3) 4 [1373] 1373 innerTail217_17
    (if 3 ≤ innerCandidates217_16.length then search edge 3 [1373, 7541] innerCandidates217_16 8914 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan217_17

def innerTail217_15 : List Nat := 6803 :: innerTail217_16
def innerCandidates217_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates217_15_checked : candidates edge 4 1373 6803 innerTail217_16 best.bound = innerCandidates217_15 := by
  unfold candidates
  have row : edge 6803 = SearchData.row6803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan217_15 : scan edge (search edge 3) 4 [1373] 1373 innerTail217_15 best = best := by
  rw [innerTail217_15, scan, if_neg (by decide), innerCandidates217_15_checked]
  change scan edge (search edge 3) 4 [1373] 1373 innerTail217_16
    (if 3 ≤ innerCandidates217_15.length then search edge 3 [1373, 6803] innerCandidates217_15 8176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan217_16

def innerTail217_14 : List Nat := 6491 :: innerTail217_15
def innerCandidates217_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates217_14_checked : candidates edge 4 1373 6491 innerTail217_15 best.bound = innerCandidates217_14 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan217_14 : scan edge (search edge 3) 4 [1373] 1373 innerTail217_14 best = best := by
  rw [innerTail217_14, scan, if_neg (by decide), innerCandidates217_14_checked]
  change scan edge (search edge 3) 4 [1373] 1373 innerTail217_15
    (if 3 ≤ innerCandidates217_14.length then search edge 3 [1373, 6491] innerCandidates217_14 7864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan217_15

def innerTail217_13 : List Nat := 6143 :: innerTail217_14
def innerCandidates217_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates217_13_checked : candidates edge 4 1373 6143 innerTail217_14 best.bound = innerCandidates217_13 := by
  unfold candidates
  have row : edge 6143 = SearchData.row6143.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan217_13 : scan edge (search edge 3) 4 [1373] 1373 innerTail217_13 best = best := by
  rw [innerTail217_13, scan, if_neg (by decide), innerCandidates217_13_checked]
  change scan edge (search edge 3) 4 [1373] 1373 innerTail217_14
    (if 3 ≤ innerCandidates217_13.length then search edge 3 [1373, 6143] innerCandidates217_13 7516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan217_14

def innerTail217_12 : List Nat := 5333 :: innerTail217_13
def innerCandidates217_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates217_12_checked : candidates edge 4 1373 5333 innerTail217_13 best.bound = innerCandidates217_12 := by
  unfold candidates
  have row : edge 5333 = SearchData.row5333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan217_12 : scan edge (search edge 3) 4 [1373] 1373 innerTail217_12 best = best := by
  rw [innerTail217_12, scan, if_neg (by decide), innerCandidates217_12_checked]
  change scan edge (search edge 3) 4 [1373] 1373 innerTail217_13
    (if 3 ≤ innerCandidates217_12.length then search edge 3 [1373, 5333] innerCandidates217_12 6706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan217_13

def innerTail217_11 : List Nat := 4493 :: innerTail217_12
def innerCandidates217_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates217_11_checked : candidates edge 4 1373 4493 innerTail217_12 best.bound = innerCandidates217_11 := by
  unfold candidates
  have row : edge 4493 = SearchData.row4493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan217_11 : scan edge (search edge 3) 4 [1373] 1373 innerTail217_11 best = best := by
  rw [innerTail217_11, scan, if_neg (by decide), innerCandidates217_11_checked]
  change scan edge (search edge 3) 4 [1373] 1373 innerTail217_12
    (if 3 ≤ innerCandidates217_11.length then search edge 3 [1373, 4493] innerCandidates217_11 5866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan217_12

def innerTail217_10 : List Nat := 3539 :: innerTail217_11
def innerCandidates217_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates217_10_checked : candidates edge 4 1373 3539 innerTail217_11 best.bound = innerCandidates217_10 := by
  unfold candidates
  have row : edge 3539 = SearchData.row3539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan217_10 : scan edge (search edge 3) 4 [1373] 1373 innerTail217_10 best = best := by
  rw [innerTail217_10, scan, if_neg (by decide), innerCandidates217_10_checked]
  change scan edge (search edge 3) 4 [1373] 1373 innerTail217_11
    (if 3 ≤ innerCandidates217_10.length then search edge 3 [1373, 3539] innerCandidates217_10 4912 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan217_11

def innerTail217_9 : List Nat := 3023 :: innerTail217_10
def innerCandidates217_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates217_9_checked : candidates edge 4 1373 3023 innerTail217_10 best.bound = innerCandidates217_9 := by
  unfold candidates
  have row : edge 3023 = SearchData.row3023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan217_9 : scan edge (search edge 3) 4 [1373] 1373 innerTail217_9 best = best := by
  rw [innerTail217_9, scan, if_neg (by decide), innerCandidates217_9_checked]
  change scan edge (search edge 3) 4 [1373] 1373 innerTail217_10
    (if 3 ≤ innerCandidates217_9.length then search edge 3 [1373, 3023] innerCandidates217_9 4396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan217_10

def innerTail217_8 : List Nat := 2591 :: innerTail217_9
def innerCandidates217_8 : List Nat := [7937]
set_option maxRecDepth 4096 in
theorem innerCandidates217_8_checked : candidates edge 4 1373 2591 innerTail217_9 best.bound = innerCandidates217_8 := by
  unfold candidates
  have row : edge 2591 = PrimePairCache.smallRow2591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan217_8 : scan edge (search edge 3) 4 [1373] 1373 innerTail217_8 best = best := by
  rw [innerTail217_8, scan, if_neg (by decide), innerCandidates217_8_checked]
  change scan edge (search edge 3) 4 [1373] 1373 innerTail217_9
    (if 3 ≤ innerCandidates217_8.length then search edge 3 [1373, 2591] innerCandidates217_8 3964 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan217_9

def innerTail217_7 : List Nat := 2531 :: innerTail217_8
def innerCandidates217_7 : List Nat := [3539]
set_option maxRecDepth 4096 in
theorem innerCandidates217_7_checked : candidates edge 4 1373 2531 innerTail217_8 best.bound = innerCandidates217_7 := by
  unfold candidates
  have row : edge 2531 = PrimePairCache.smallRow2531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan217_7 : scan edge (search edge 3) 4 [1373] 1373 innerTail217_7 best = best := by
  rw [innerTail217_7, scan, if_neg (by decide), innerCandidates217_7_checked]
  change scan edge (search edge 3) 4 [1373] 1373 innerTail217_8
    (if 3 ≤ innerCandidates217_7.length then search edge 3 [1373, 2531] innerCandidates217_7 3904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan217_8

def innerTail217_6 : List Nat := 2423 :: innerTail217_7
def innerCandidates217_6 : List Nat := [7727]
set_option maxRecDepth 4096 in
theorem innerCandidates217_6_checked : candidates edge 4 1373 2423 innerTail217_7 best.bound = innerCandidates217_6 := by
  unfold candidates
  have row : edge 2423 = PrimePairCache.smallRow2423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan217_6 : scan edge (search edge 3) 4 [1373] 1373 innerTail217_6 best = best := by
  rw [innerTail217_6, scan, if_neg (by decide), innerCandidates217_6_checked]
  change scan edge (search edge 3) 4 [1373] 1373 innerTail217_7
    (if 3 ≤ innerCandidates217_6.length then search edge 3 [1373, 2423] innerCandidates217_6 3796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan217_7

def innerTail217_5 : List Nat := 1901 :: innerTail217_6
def innerCandidates217_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates217_5_checked : candidates edge 4 1373 1901 innerTail217_6 best.bound = innerCandidates217_5 := by
  unfold candidates
  have row : edge 1901 = PrimePairCache.smallRow1901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan217_5 : scan edge (search edge 3) 4 [1373] 1373 innerTail217_5 best = best := by
  rw [innerTail217_5, scan, if_neg (by decide), innerCandidates217_5_checked]
  change scan edge (search edge 3) 4 [1373] 1373 innerTail217_6
    (if 3 ≤ innerCandidates217_5.length then search edge 3 [1373, 1901] innerCandidates217_5 3274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan217_6

def innerTail217_4 : List Nat := 1637 :: innerTail217_5
def innerCandidates217_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates217_4_checked : candidates edge 4 1373 1637 innerTail217_5 best.bound = innerCandidates217_4 := by
  unfold candidates
  have row : edge 1637 = PrimePairCache.smallRow1637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan217_4 : scan edge (search edge 3) 4 [1373] 1373 innerTail217_4 best = best := by
  rw [innerTail217_4, scan, if_neg (by decide), innerCandidates217_4_checked]
  change scan edge (search edge 3) 4 [1373] 1373 innerTail217_5
    (if 3 ≤ innerCandidates217_4.length then search edge 3 [1373, 1637] innerCandidates217_4 3010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan217_5

def innerTail217_3 : List Nat := 1511 :: innerTail217_4
def innerCandidates217_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates217_3_checked : candidates edge 4 1373 1511 innerTail217_4 best.bound = innerCandidates217_3 := by
  unfold candidates
  have row : edge 1511 = PrimePairCache.smallRow1511.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan217_3 : scan edge (search edge 3) 4 [1373] 1373 innerTail217_3 best = best := by
  rw [innerTail217_3, scan, if_neg (by decide), innerCandidates217_3_checked]
  change scan edge (search edge 3) 4 [1373] 1373 innerTail217_4
    (if 3 ≤ innerCandidates217_3.length then search edge 3 [1373, 1511] innerCandidates217_3 2884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan217_4

def innerTail217_2 : List Nat := 1499 :: innerTail217_3
def innerCandidates217_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates217_2_checked : candidates edge 4 1373 1499 innerTail217_3 best.bound = innerCandidates217_2 := by
  unfold candidates
  have row : edge 1499 = PrimePairCache.smallRow1499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan217_2 : scan edge (search edge 3) 4 [1373] 1373 innerTail217_2 best = best := by
  rw [innerTail217_2, scan, if_neg (by decide), innerCandidates217_2_checked]
  change scan edge (search edge 3) 4 [1373] 1373 innerTail217_3
    (if 3 ≤ innerCandidates217_2.length then search edge 3 [1373, 1499] innerCandidates217_2 2872 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan217_3

def innerTail217_1 : List Nat := 1451 :: innerTail217_2
def innerCandidates217_1 : List Nat := [7907]
set_option maxRecDepth 4096 in
theorem innerCandidates217_1_checked : candidates edge 4 1373 1451 innerTail217_2 best.bound = innerCandidates217_1 := by
  unfold candidates
  have row : edge 1451 = PrimePairCache.smallRow1451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan217_1 : scan edge (search edge 3) 4 [1373] 1373 innerTail217_1 best = best := by
  rw [innerTail217_1, scan, if_neg (by decide), innerCandidates217_1_checked]
  change scan edge (search edge 3) 4 [1373] 1373 innerTail217_2
    (if 3 ≤ innerCandidates217_1.length then search edge 3 [1373, 1451] innerCandidates217_1 2824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan217_2

def innerTail217_0 : List Nat := 1439 :: innerTail217_1
def innerCandidates217_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates217_0_checked : candidates edge 4 1373 1439 innerTail217_1 best.bound = innerCandidates217_0 := by
  unfold candidates
  have row : edge 1439 = PrimePairCache.smallRow1439.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan217_0 : scan edge (search edge 3) 4 [1373] 1373 innerTail217_0 best = best := by
  rw [innerTail217_0, scan, if_neg (by decide), innerCandidates217_0_checked]
  change scan edge (search edge 3) 4 [1373] 1373 innerTail217_1
    (if 3 ≤ innerCandidates217_0.length then search edge 3 [1373, 1439] innerCandidates217_0 2812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan217_1

theorem rootBranch217 : search edge 4 [1373] rootCandidates217 1373 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1373] 1373 innerTail217_0 best = best
  exact innerScan217_0

def rootCandidates216 : List Nat := [1733, 2027, 2297, 2591, 2843, 2963, 3221, 3467, 3929, 4079, 4373, 4691, 4733, 4877, 5471, 6323, 6353, 6563, 7043, 7229, 7919, 7949, 8219]
set_option maxRecDepth 4096 in
theorem rootCandidates216_checked : candidates edge 5 0 1367 tail217 best.bound = rootCandidates216 := by
  unfold candidates
  have row : edge 1367 = PrimePairCache.smallRow1367.contains := by funext q; rfl
  rw [row]
  decide

def innerTail216_23 : List Nat := []
theorem innerScan216_23 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_23 best = best := rfl

def innerTail216_22 : List Nat := 8219 :: innerTail216_23
def innerCandidates216_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates216_22_checked : candidates edge 4 1367 8219 innerTail216_23 best.bound = innerCandidates216_22 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan216_22 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_22 best = best := by
  rw [innerTail216_22, scan, if_neg (by decide), innerCandidates216_22_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_23
    (if 3 ≤ innerCandidates216_22.length then search edge 3 [1367, 8219] innerCandidates216_22 9586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan216_23

def innerTail216_21 : List Nat := 7949 :: innerTail216_22
def innerCandidates216_21 : List Nat := [8219]
set_option maxRecDepth 4096 in
theorem innerCandidates216_21_checked : candidates edge 4 1367 7949 innerTail216_22 best.bound = innerCandidates216_21 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan216_21 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_21 best = best := by
  rw [innerTail216_21, scan, if_neg (by decide), innerCandidates216_21_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_22
    (if 3 ≤ innerCandidates216_21.length then search edge 3 [1367, 7949] innerCandidates216_21 9316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan216_22

def innerTail216_20 : List Nat := 7919 :: innerTail216_21
def innerCandidates216_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates216_20_checked : candidates edge 4 1367 7919 innerTail216_21 best.bound = innerCandidates216_20 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan216_20 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_20 best = best := by
  rw [innerTail216_20, scan, if_neg (by decide), innerCandidates216_20_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_21
    (if 3 ≤ innerCandidates216_20.length then search edge 3 [1367, 7919] innerCandidates216_20 9286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan216_21

def innerTail216_19 : List Nat := 7229 :: innerTail216_20
def innerCandidates216_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates216_19_checked : candidates edge 4 1367 7229 innerTail216_20 best.bound = innerCandidates216_19 := by
  unfold candidates
  have row : edge 7229 = SearchData.row7229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan216_19 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_19 best = best := by
  rw [innerTail216_19, scan, if_neg (by decide), innerCandidates216_19_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_20
    (if 3 ≤ innerCandidates216_19.length then search edge 3 [1367, 7229] innerCandidates216_19 8596 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan216_20

def innerTail216_18 : List Nat := 7043 :: innerTail216_19
def innerCandidates216_18 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates216_18_checked : candidates edge 4 1367 7043 innerTail216_19 best.bound = innerCandidates216_18 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan216_18 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_18 best = best := by
  rw [innerTail216_18, scan, if_neg (by decide), innerCandidates216_18_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_19
    (if 3 ≤ innerCandidates216_18.length then search edge 3 [1367, 7043] innerCandidates216_18 8410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan216_19

def innerTail216_17 : List Nat := 6563 :: innerTail216_18
def innerCandidates216_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates216_17_checked : candidates edge 4 1367 6563 innerTail216_18 best.bound = innerCandidates216_17 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan216_17 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_17 best = best := by
  rw [innerTail216_17, scan, if_neg (by decide), innerCandidates216_17_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_18
    (if 3 ≤ innerCandidates216_17.length then search edge 3 [1367, 6563] innerCandidates216_17 7930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan216_18

def innerTail216_16 : List Nat := 6353 :: innerTail216_17
def innerCandidates216_16 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates216_16_checked : candidates edge 4 1367 6353 innerTail216_17 best.bound = innerCandidates216_16 := by
  unfold candidates
  have row : edge 6353 = SearchData.row6353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan216_16 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_16 best = best := by
  rw [innerTail216_16, scan, if_neg (by decide), innerCandidates216_16_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_17
    (if 3 ≤ innerCandidates216_16.length then search edge 3 [1367, 6353] innerCandidates216_16 7720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan216_17

def innerTail216_15 : List Nat := 6323 :: innerTail216_16
def innerCandidates216_15 : List Nat := [7919]
set_option maxRecDepth 4096 in
theorem innerCandidates216_15_checked : candidates edge 4 1367 6323 innerTail216_16 best.bound = innerCandidates216_15 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan216_15 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_15 best = best := by
  rw [innerTail216_15, scan, if_neg (by decide), innerCandidates216_15_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_16
    (if 3 ≤ innerCandidates216_15.length then search edge 3 [1367, 6323] innerCandidates216_15 7690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan216_16

def innerTail216_14 : List Nat := 5471 :: innerTail216_15
def innerCandidates216_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates216_14_checked : candidates edge 4 1367 5471 innerTail216_15 best.bound = innerCandidates216_14 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan216_14 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_14 best = best := by
  rw [innerTail216_14, scan, if_neg (by decide), innerCandidates216_14_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_15
    (if 3 ≤ innerCandidates216_14.length then search edge 3 [1367, 5471] innerCandidates216_14 6838 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan216_15

def innerTail216_13 : List Nat := 4877 :: innerTail216_14
def innerCandidates216_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates216_13_checked : candidates edge 4 1367 4877 innerTail216_14 best.bound = innerCandidates216_13 := by
  unfold candidates
  have row : edge 4877 = SearchData.row4877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan216_13 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_13 best = best := by
  rw [innerTail216_13, scan, if_neg (by decide), innerCandidates216_13_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_14
    (if 3 ≤ innerCandidates216_13.length then search edge 3 [1367, 4877] innerCandidates216_13 6244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan216_14

def innerTail216_12 : List Nat := 4733 :: innerTail216_13
def innerCandidates216_12 : List Nat := [7919]
set_option maxRecDepth 4096 in
theorem innerCandidates216_12_checked : candidates edge 4 1367 4733 innerTail216_13 best.bound = innerCandidates216_12 := by
  unfold candidates
  have row : edge 4733 = SearchData.row4733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan216_12 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_12 best = best := by
  rw [innerTail216_12, scan, if_neg (by decide), innerCandidates216_12_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_13
    (if 3 ≤ innerCandidates216_12.length then search edge 3 [1367, 4733] innerCandidates216_12 6100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan216_13

def innerTail216_11 : List Nat := 4691 :: innerTail216_12
def innerCandidates216_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates216_11_checked : candidates edge 4 1367 4691 innerTail216_12 best.bound = innerCandidates216_11 := by
  unfold candidates
  have row : edge 4691 = SearchData.row4691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan216_11 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_11 best = best := by
  rw [innerTail216_11, scan, if_neg (by decide), innerCandidates216_11_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_12
    (if 3 ≤ innerCandidates216_11.length then search edge 3 [1367, 4691] innerCandidates216_11 6058 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan216_12

def innerTail216_10 : List Nat := 4373 :: innerTail216_11
def innerCandidates216_10 : List Nat := [7919]
set_option maxRecDepth 4096 in
theorem innerCandidates216_10_checked : candidates edge 4 1367 4373 innerTail216_11 best.bound = innerCandidates216_10 := by
  unfold candidates
  have row : edge 4373 = SearchData.row4373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan216_10 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_10 best = best := by
  rw [innerTail216_10, scan, if_neg (by decide), innerCandidates216_10_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_11
    (if 3 ≤ innerCandidates216_10.length then search edge 3 [1367, 4373] innerCandidates216_10 5740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan216_11

def innerTail216_9 : List Nat := 4079 :: innerTail216_10
def innerCandidates216_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates216_9_checked : candidates edge 4 1367 4079 innerTail216_10 best.bound = innerCandidates216_9 := by
  unfold candidates
  have row : edge 4079 = SearchData.row4079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan216_9 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_9 best = best := by
  rw [innerTail216_9, scan, if_neg (by decide), innerCandidates216_9_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_10
    (if 3 ≤ innerCandidates216_9.length then search edge 3 [1367, 4079] innerCandidates216_9 5446 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan216_10

def innerTail216_8 : List Nat := 3929 :: innerTail216_9
def innerCandidates216_8 : List Nat := [7043]
set_option maxRecDepth 4096 in
theorem innerCandidates216_8_checked : candidates edge 4 1367 3929 innerTail216_9 best.bound = innerCandidates216_8 := by
  unfold candidates
  have row : edge 3929 = SearchData.row3929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan216_8 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_8 best = best := by
  rw [innerTail216_8, scan, if_neg (by decide), innerCandidates216_8_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_9
    (if 3 ≤ innerCandidates216_8.length then search edge 3 [1367, 3929] innerCandidates216_8 5296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan216_9

def innerTail216_7 : List Nat := 3467 :: innerTail216_8
def innerCandidates216_7 : List Nat := [5471, 7949]
set_option maxRecDepth 4096 in
theorem innerCandidates216_7_checked : candidates edge 4 1367 3467 innerTail216_8 best.bound = innerCandidates216_7 := by
  unfold candidates
  have row : edge 3467 = SearchData.row3467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan216_7 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_7 best = best := by
  rw [innerTail216_7, scan, if_neg (by decide), innerCandidates216_7_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_8
    (if 3 ≤ innerCandidates216_7.length then search edge 3 [1367, 3467] innerCandidates216_7 4834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan216_8

def innerTail216_6 : List Nat := 3221 :: innerTail216_7
def innerCandidates216_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates216_6_checked : candidates edge 4 1367 3221 innerTail216_7 best.bound = innerCandidates216_6 := by
  unfold candidates
  have row : edge 3221 = SearchData.row3221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan216_6 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_6 best = best := by
  rw [innerTail216_6, scan, if_neg (by decide), innerCandidates216_6_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_7
    (if 3 ≤ innerCandidates216_6.length then search edge 3 [1367, 3221] innerCandidates216_6 4588 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan216_7

def innerTail216_5 : List Nat := 2963 :: innerTail216_6
def innerCandidates216_5 : List Nat := [4079, 4691]
set_option maxRecDepth 4096 in
theorem innerCandidates216_5_checked : candidates edge 4 1367 2963 innerTail216_6 best.bound = innerCandidates216_5 := by
  unfold candidates
  have row : edge 2963 = SearchData.row2963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan216_5 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_5 best = best := by
  rw [innerTail216_5, scan, if_neg (by decide), innerCandidates216_5_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_6
    (if 3 ≤ innerCandidates216_5.length then search edge 3 [1367, 2963] innerCandidates216_5 4330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan216_6

def innerTail216_4 : List Nat := 2843 :: innerTail216_5
def innerCandidates216_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates216_4_checked : candidates edge 4 1367 2843 innerTail216_5 best.bound = innerCandidates216_4 := by
  unfold candidates
  have row : edge 2843 = SearchData.row2843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan216_4 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_4 best = best := by
  rw [innerTail216_4, scan, if_neg (by decide), innerCandidates216_4_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_5
    (if 3 ≤ innerCandidates216_4.length then search edge 3 [1367, 2843] innerCandidates216_4 4210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan216_5

def innerTail216_3 : List Nat := 2591 :: innerTail216_4
def innerCandidates216_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates216_3_checked : candidates edge 4 1367 2591 innerTail216_4 best.bound = innerCandidates216_3 := by
  unfold candidates
  have row : edge 2591 = PrimePairCache.smallRow2591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan216_3 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_3 best = best := by
  rw [innerTail216_3, scan, if_neg (by decide), innerCandidates216_3_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_4
    (if 3 ≤ innerCandidates216_3.length then search edge 3 [1367, 2591] innerCandidates216_3 3958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan216_4

def innerTail216_2 : List Nat := 2297 :: innerTail216_3
def innerCandidates216_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates216_2_checked : candidates edge 4 1367 2297 innerTail216_3 best.bound = innerCandidates216_2 := by
  unfold candidates
  have row : edge 2297 = PrimePairCache.smallRow2297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan216_2 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_2 best = best := by
  rw [innerTail216_2, scan, if_neg (by decide), innerCandidates216_2_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_3
    (if 3 ≤ innerCandidates216_2.length then search edge 3 [1367, 2297] innerCandidates216_2 3664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan216_3

def innerTail216_1 : List Nat := 2027 :: innerTail216_2
def innerCandidates216_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates216_1_checked : candidates edge 4 1367 2027 innerTail216_2 best.bound = innerCandidates216_1 := by
  unfold candidates
  have row : edge 2027 = PrimePairCache.smallRow2027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan216_1 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_1 best = best := by
  rw [innerTail216_1, scan, if_neg (by decide), innerCandidates216_1_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_2
    (if 3 ≤ innerCandidates216_1.length then search edge 3 [1367, 2027] innerCandidates216_1 3394 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan216_2

def innerTail216_0 : List Nat := 1733 :: innerTail216_1
def innerCandidates216_0 : List Nat := [6323, 7043, 7949]
set_option maxRecDepth 4096 in
theorem innerCandidates216_0_checked : candidates edge 4 1367 1733 innerTail216_1 best.bound = innerCandidates216_0 := by
  unfold candidates
  have row : edge 1733 = PrimePairCache.smallRow1733.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild216_0 : search edge 3 [1367, 1733] innerCandidates216_0 3100 best = best := by decide

theorem innerScan216_0 : scan edge (search edge 3) 4 [1367] 1367 innerTail216_0 best = best := by
  rw [innerTail216_0, scan, if_neg (by decide), innerCandidates216_0_checked]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_1
    (if 3 ≤ innerCandidates216_0.length then search edge 3 [1367, 1733] innerCandidates216_0 3100 best else best) = best
  rw [if_pos (by decide), innerChild216_0]
  exact innerScan216_1

theorem rootBranch216 : search edge 4 [1367] rootCandidates216 1367 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1367] 1367 innerTail216_0 best = best
  exact innerScan216_0

def rootCandidates215 : List Nat := [2141, 3461, 3581, 3701, 3917, 3947, 4793, 5099, 5669, 6047, 6599, 6737, 6899, 6947, 7103, 7607]
set_option maxRecDepth 4096 in
theorem rootCandidates215_checked : candidates edge 5 0 1361 tail216 best.bound = rootCandidates215 := by
  unfold candidates
  have row : edge 1361 = PrimePairCache.smallRow1361.contains := by funext q; rfl
  rw [row]
  decide

def innerTail215_16 : List Nat := []
theorem innerScan215_16 : scan edge (search edge 3) 4 [1361] 1361 innerTail215_16 best = best := rfl

def innerTail215_15 : List Nat := 7607 :: innerTail215_16
def innerCandidates215_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates215_15_checked : candidates edge 4 1361 7607 innerTail215_16 best.bound = innerCandidates215_15 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan215_15 : scan edge (search edge 3) 4 [1361] 1361 innerTail215_15 best = best := by
  rw [innerTail215_15, scan, if_neg (by decide), innerCandidates215_15_checked]
  change scan edge (search edge 3) 4 [1361] 1361 innerTail215_16
    (if 3 ≤ innerCandidates215_15.length then search edge 3 [1361, 7607] innerCandidates215_15 8968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan215_16

def innerTail215_14 : List Nat := 7103 :: innerTail215_15
def innerCandidates215_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates215_14_checked : candidates edge 4 1361 7103 innerTail215_15 best.bound = innerCandidates215_14 := by
  unfold candidates
  have row : edge 7103 = SearchData.row7103.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan215_14 : scan edge (search edge 3) 4 [1361] 1361 innerTail215_14 best = best := by
  rw [innerTail215_14, scan, if_neg (by decide), innerCandidates215_14_checked]
  change scan edge (search edge 3) 4 [1361] 1361 innerTail215_15
    (if 3 ≤ innerCandidates215_14.length then search edge 3 [1361, 7103] innerCandidates215_14 8464 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan215_15

def innerTail215_13 : List Nat := 6947 :: innerTail215_14
def innerCandidates215_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates215_13_checked : candidates edge 4 1361 6947 innerTail215_14 best.bound = innerCandidates215_13 := by
  unfold candidates
  have row : edge 6947 = SearchData.row6947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan215_13 : scan edge (search edge 3) 4 [1361] 1361 innerTail215_13 best = best := by
  rw [innerTail215_13, scan, if_neg (by decide), innerCandidates215_13_checked]
  change scan edge (search edge 3) 4 [1361] 1361 innerTail215_14
    (if 3 ≤ innerCandidates215_13.length then search edge 3 [1361, 6947] innerCandidates215_13 8308 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan215_14

def innerTail215_12 : List Nat := 6899 :: innerTail215_13
def innerCandidates215_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates215_12_checked : candidates edge 4 1361 6899 innerTail215_13 best.bound = innerCandidates215_12 := by
  unfold candidates
  have row : edge 6899 = SearchData.row6899.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan215_12 : scan edge (search edge 3) 4 [1361] 1361 innerTail215_12 best = best := by
  rw [innerTail215_12, scan, if_neg (by decide), innerCandidates215_12_checked]
  change scan edge (search edge 3) 4 [1361] 1361 innerTail215_13
    (if 3 ≤ innerCandidates215_12.length then search edge 3 [1361, 6899] innerCandidates215_12 8260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan215_13

def innerTail215_11 : List Nat := 6737 :: innerTail215_12
def innerCandidates215_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates215_11_checked : candidates edge 4 1361 6737 innerTail215_12 best.bound = innerCandidates215_11 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan215_11 : scan edge (search edge 3) 4 [1361] 1361 innerTail215_11 best = best := by
  rw [innerTail215_11, scan, if_neg (by decide), innerCandidates215_11_checked]
  change scan edge (search edge 3) 4 [1361] 1361 innerTail215_12
    (if 3 ≤ innerCandidates215_11.length then search edge 3 [1361, 6737] innerCandidates215_11 8098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan215_12

def innerTail215_10 : List Nat := 6599 :: innerTail215_11
def innerCandidates215_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates215_10_checked : candidates edge 4 1361 6599 innerTail215_11 best.bound = innerCandidates215_10 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan215_10 : scan edge (search edge 3) 4 [1361] 1361 innerTail215_10 best = best := by
  rw [innerTail215_10, scan, if_neg (by decide), innerCandidates215_10_checked]
  change scan edge (search edge 3) 4 [1361] 1361 innerTail215_11
    (if 3 ≤ innerCandidates215_10.length then search edge 3 [1361, 6599] innerCandidates215_10 7960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan215_11

def innerTail215_9 : List Nat := 6047 :: innerTail215_10
def innerCandidates215_9 : List Nat := [6737]
set_option maxRecDepth 4096 in
theorem innerCandidates215_9_checked : candidates edge 4 1361 6047 innerTail215_10 best.bound = innerCandidates215_9 := by
  unfold candidates
  have row : edge 6047 = SearchData.row6047.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan215_9 : scan edge (search edge 3) 4 [1361] 1361 innerTail215_9 best = best := by
  rw [innerTail215_9, scan, if_neg (by decide), innerCandidates215_9_checked]
  change scan edge (search edge 3) 4 [1361] 1361 innerTail215_10
    (if 3 ≤ innerCandidates215_9.length then search edge 3 [1361, 6047] innerCandidates215_9 7408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan215_10

def innerTail215_8 : List Nat := 5669 :: innerTail215_9
def innerCandidates215_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates215_8_checked : candidates edge 4 1361 5669 innerTail215_9 best.bound = innerCandidates215_8 := by
  unfold candidates
  have row : edge 5669 = SearchData.row5669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan215_8 : scan edge (search edge 3) 4 [1361] 1361 innerTail215_8 best = best := by
  rw [innerTail215_8, scan, if_neg (by decide), innerCandidates215_8_checked]
  change scan edge (search edge 3) 4 [1361] 1361 innerTail215_9
    (if 3 ≤ innerCandidates215_8.length then search edge 3 [1361, 5669] innerCandidates215_8 7030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan215_9

def innerTail215_7 : List Nat := 5099 :: innerTail215_8
def innerCandidates215_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates215_7_checked : candidates edge 4 1361 5099 innerTail215_8 best.bound = innerCandidates215_7 := by
  unfold candidates
  have row : edge 5099 = SearchData.row5099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan215_7 : scan edge (search edge 3) 4 [1361] 1361 innerTail215_7 best = best := by
  rw [innerTail215_7, scan, if_neg (by decide), innerCandidates215_7_checked]
  change scan edge (search edge 3) 4 [1361] 1361 innerTail215_8
    (if 3 ≤ innerCandidates215_7.length then search edge 3 [1361, 5099] innerCandidates215_7 6460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan215_8

def innerTail215_6 : List Nat := 4793 :: innerTail215_7
def innerCandidates215_6 : List Nat := [5099]
set_option maxRecDepth 4096 in
theorem innerCandidates215_6_checked : candidates edge 4 1361 4793 innerTail215_7 best.bound = innerCandidates215_6 := by
  unfold candidates
  have row : edge 4793 = SearchData.row4793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan215_6 : scan edge (search edge 3) 4 [1361] 1361 innerTail215_6 best = best := by
  rw [innerTail215_6, scan, if_neg (by decide), innerCandidates215_6_checked]
  change scan edge (search edge 3) 4 [1361] 1361 innerTail215_7
    (if 3 ≤ innerCandidates215_6.length then search edge 3 [1361, 4793] innerCandidates215_6 6154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan215_7

def innerTail215_5 : List Nat := 3947 :: innerTail215_6
def innerCandidates215_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates215_5_checked : candidates edge 4 1361 3947 innerTail215_6 best.bound = innerCandidates215_5 := by
  unfold candidates
  have row : edge 3947 = SearchData.row3947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan215_5 : scan edge (search edge 3) 4 [1361] 1361 innerTail215_5 best = best := by
  rw [innerTail215_5, scan, if_neg (by decide), innerCandidates215_5_checked]
  change scan edge (search edge 3) 4 [1361] 1361 innerTail215_6
    (if 3 ≤ innerCandidates215_5.length then search edge 3 [1361, 3947] innerCandidates215_5 5308 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan215_6

def innerTail215_4 : List Nat := 3917 :: innerTail215_5
def innerCandidates215_4 : List Nat := [6899]
set_option maxRecDepth 4096 in
theorem innerCandidates215_4_checked : candidates edge 4 1361 3917 innerTail215_5 best.bound = innerCandidates215_4 := by
  unfold candidates
  have row : edge 3917 = SearchData.row3917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan215_4 : scan edge (search edge 3) 4 [1361] 1361 innerTail215_4 best = best := by
  rw [innerTail215_4, scan, if_neg (by decide), innerCandidates215_4_checked]
  change scan edge (search edge 3) 4 [1361] 1361 innerTail215_5
    (if 3 ≤ innerCandidates215_4.length then search edge 3 [1361, 3917] innerCandidates215_4 5278 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan215_5

def innerTail215_3 : List Nat := 3701 :: innerTail215_4
def innerCandidates215_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates215_3_checked : candidates edge 4 1361 3701 innerTail215_4 best.bound = innerCandidates215_3 := by
  unfold candidates
  have row : edge 3701 = SearchData.row3701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan215_3 : scan edge (search edge 3) 4 [1361] 1361 innerTail215_3 best = best := by
  rw [innerTail215_3, scan, if_neg (by decide), innerCandidates215_3_checked]
  change scan edge (search edge 3) 4 [1361] 1361 innerTail215_4
    (if 3 ≤ innerCandidates215_3.length then search edge 3 [1361, 3701] innerCandidates215_3 5062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan215_4

def innerTail215_2 : List Nat := 3581 :: innerTail215_3
def innerCandidates215_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates215_2_checked : candidates edge 4 1361 3581 innerTail215_3 best.bound = innerCandidates215_2 := by
  unfold candidates
  have row : edge 3581 = SearchData.row3581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan215_2 : scan edge (search edge 3) 4 [1361] 1361 innerTail215_2 best = best := by
  rw [innerTail215_2, scan, if_neg (by decide), innerCandidates215_2_checked]
  change scan edge (search edge 3) 4 [1361] 1361 innerTail215_3
    (if 3 ≤ innerCandidates215_2.length then search edge 3 [1361, 3581] innerCandidates215_2 4942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan215_3

def innerTail215_1 : List Nat := 3461 :: innerTail215_2
def innerCandidates215_1 : List Nat := [6737]
set_option maxRecDepth 4096 in
theorem innerCandidates215_1_checked : candidates edge 4 1361 3461 innerTail215_2 best.bound = innerCandidates215_1 := by
  unfold candidates
  have row : edge 3461 = SearchData.row3461.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan215_1 : scan edge (search edge 3) 4 [1361] 1361 innerTail215_1 best = best := by
  rw [innerTail215_1, scan, if_neg (by decide), innerCandidates215_1_checked]
  change scan edge (search edge 3) 4 [1361] 1361 innerTail215_2
    (if 3 ≤ innerCandidates215_1.length then search edge 3 [1361, 3461] innerCandidates215_1 4822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan215_2

def innerTail215_0 : List Nat := 2141 :: innerTail215_1
def innerCandidates215_0 : List Nat := [3947, 6737]
set_option maxRecDepth 4096 in
theorem innerCandidates215_0_checked : candidates edge 4 1361 2141 innerTail215_1 best.bound = innerCandidates215_0 := by
  unfold candidates
  have row : edge 2141 = PrimePairCache.smallRow2141.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan215_0 : scan edge (search edge 3) 4 [1361] 1361 innerTail215_0 best = best := by
  rw [innerTail215_0, scan, if_neg (by decide), innerCandidates215_0_checked]
  change scan edge (search edge 3) 4 [1361] 1361 innerTail215_1
    (if 3 ≤ innerCandidates215_0.length then search edge 3 [1361, 2141] innerCandidates215_0 3502 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan215_1

theorem rootBranch215 : search edge 4 [1361] rootCandidates215 1361 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1361] 1361 innerTail215_0 best = best
  exact innerScan215_0

end ProjectEulerSolutions.PrimePairUpper.Search
