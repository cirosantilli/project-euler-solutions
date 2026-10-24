import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates88 : List Nat := [587, 617, 641, 941, 1301, 1571, 1913, 1973, 2039, 2099, 2141, 2333, 2621, 2687, 3917, 3989, 4049, 4127, 4253, 4421, 4463, 5477, 5861, 6029, 6101, 6323, 6779, 6827, 8231]
set_option maxRecDepth 4096 in
theorem rootCandidates88_checked : candidates edge 5 0 467 tail89 best.bound = rootCandidates88 := by
  unfold candidates
  have row : edge 467 = PrimePairCache.smallRow467.contains := by funext q; rfl
  rw [row]
  decide

def innerTail88_29 : List Nat := []
theorem innerScan88_29 : scan edge (search edge 3) 4 [467] 467 innerTail88_29 best = best := rfl

def innerTail88_28 : List Nat := 8231 :: innerTail88_29
def innerCandidates88_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates88_28_checked : candidates edge 4 467 8231 innerTail88_29 best.bound = innerCandidates88_28 := by
  unfold candidates
  have row : edge 8231 = SearchData.row8231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_28 : scan edge (search edge 3) 4 [467] 467 innerTail88_28 best = best := by
  rw [innerTail88_28, scan, if_neg (by decide), innerCandidates88_28_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_29
    (if 3 ≤ innerCandidates88_28.length then search edge 3 [467, 8231] innerCandidates88_28 8698 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_29

def innerTail88_27 : List Nat := 6827 :: innerTail88_28
def innerCandidates88_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates88_27_checked : candidates edge 4 467 6827 innerTail88_28 best.bound = innerCandidates88_27 := by
  unfold candidates
  have row : edge 6827 = SearchData.row6827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_27 : scan edge (search edge 3) 4 [467] 467 innerTail88_27 best = best := by
  rw [innerTail88_27, scan, if_neg (by decide), innerCandidates88_27_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_28
    (if 3 ≤ innerCandidates88_27.length then search edge 3 [467, 6827] innerCandidates88_27 7294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_28

def innerTail88_26 : List Nat := 6779 :: innerTail88_27
def innerCandidates88_26 : List Nat := [8231]
set_option maxRecDepth 4096 in
theorem innerCandidates88_26_checked : candidates edge 4 467 6779 innerTail88_27 best.bound = innerCandidates88_26 := by
  unfold candidates
  have row : edge 6779 = SearchData.row6779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_26 : scan edge (search edge 3) 4 [467] 467 innerTail88_26 best = best := by
  rw [innerTail88_26, scan, if_neg (by decide), innerCandidates88_26_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_27
    (if 3 ≤ innerCandidates88_26.length then search edge 3 [467, 6779] innerCandidates88_26 7246 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_27

def innerTail88_25 : List Nat := 6323 :: innerTail88_26
def innerCandidates88_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates88_25_checked : candidates edge 4 467 6323 innerTail88_26 best.bound = innerCandidates88_25 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_25 : scan edge (search edge 3) 4 [467] 467 innerTail88_25 best = best := by
  rw [innerTail88_25, scan, if_neg (by decide), innerCandidates88_25_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_26
    (if 3 ≤ innerCandidates88_25.length then search edge 3 [467, 6323] innerCandidates88_25 6790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_26

def innerTail88_24 : List Nat := 6101 :: innerTail88_25
def innerCandidates88_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates88_24_checked : candidates edge 4 467 6101 innerTail88_25 best.bound = innerCandidates88_24 := by
  unfold candidates
  have row : edge 6101 = SearchData.row6101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_24 : scan edge (search edge 3) 4 [467] 467 innerTail88_24 best = best := by
  rw [innerTail88_24, scan, if_neg (by decide), innerCandidates88_24_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_25
    (if 3 ≤ innerCandidates88_24.length then search edge 3 [467, 6101] innerCandidates88_24 6568 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_25

def innerTail88_23 : List Nat := 6029 :: innerTail88_24
def innerCandidates88_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates88_23_checked : candidates edge 4 467 6029 innerTail88_24 best.bound = innerCandidates88_23 := by
  unfold candidates
  have row : edge 6029 = SearchData.row6029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_23 : scan edge (search edge 3) 4 [467] 467 innerTail88_23 best = best := by
  rw [innerTail88_23, scan, if_neg (by decide), innerCandidates88_23_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_24
    (if 3 ≤ innerCandidates88_23.length then search edge 3 [467, 6029] innerCandidates88_23 6496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_24

def innerTail88_22 : List Nat := 5861 :: innerTail88_23
def innerCandidates88_22 : List Nat := [6827]
set_option maxRecDepth 4096 in
theorem innerCandidates88_22_checked : candidates edge 4 467 5861 innerTail88_23 best.bound = innerCandidates88_22 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_22 : scan edge (search edge 3) 4 [467] 467 innerTail88_22 best = best := by
  rw [innerTail88_22, scan, if_neg (by decide), innerCandidates88_22_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_23
    (if 3 ≤ innerCandidates88_22.length then search edge 3 [467, 5861] innerCandidates88_22 6328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_23

def innerTail88_21 : List Nat := 5477 :: innerTail88_22
def innerCandidates88_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates88_21_checked : candidates edge 4 467 5477 innerTail88_22 best.bound = innerCandidates88_21 := by
  unfold candidates
  have row : edge 5477 = SearchData.row5477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_21 : scan edge (search edge 3) 4 [467] 467 innerTail88_21 best = best := by
  rw [innerTail88_21, scan, if_neg (by decide), innerCandidates88_21_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_22
    (if 3 ≤ innerCandidates88_21.length then search edge 3 [467, 5477] innerCandidates88_21 5944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_22

def innerTail88_20 : List Nat := 4463 :: innerTail88_21
def innerCandidates88_20 : List Nat := [5861]
set_option maxRecDepth 4096 in
theorem innerCandidates88_20_checked : candidates edge 4 467 4463 innerTail88_21 best.bound = innerCandidates88_20 := by
  unfold candidates
  have row : edge 4463 = SearchData.row4463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_20 : scan edge (search edge 3) 4 [467] 467 innerTail88_20 best = best := by
  rw [innerTail88_20, scan, if_neg (by decide), innerCandidates88_20_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_21
    (if 3 ≤ innerCandidates88_20.length then search edge 3 [467, 4463] innerCandidates88_20 4930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_21

def innerTail88_19 : List Nat := 4421 :: innerTail88_20
def innerCandidates88_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates88_19_checked : candidates edge 4 467 4421 innerTail88_20 best.bound = innerCandidates88_19 := by
  unfold candidates
  have row : edge 4421 = SearchData.row4421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_19 : scan edge (search edge 3) 4 [467] 467 innerTail88_19 best = best := by
  rw [innerTail88_19, scan, if_neg (by decide), innerCandidates88_19_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_20
    (if 3 ≤ innerCandidates88_19.length then search edge 3 [467, 4421] innerCandidates88_19 4888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_20

def innerTail88_18 : List Nat := 4253 :: innerTail88_19
def innerCandidates88_18 : List Nat := [4463]
set_option maxRecDepth 4096 in
theorem innerCandidates88_18_checked : candidates edge 4 467 4253 innerTail88_19 best.bound = innerCandidates88_18 := by
  unfold candidates
  have row : edge 4253 = SearchData.row4253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_18 : scan edge (search edge 3) 4 [467] 467 innerTail88_18 best = best := by
  rw [innerTail88_18, scan, if_neg (by decide), innerCandidates88_18_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_19
    (if 3 ≤ innerCandidates88_18.length then search edge 3 [467, 4253] innerCandidates88_18 4720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_19

def innerTail88_17 : List Nat := 4127 :: innerTail88_18
def innerCandidates88_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates88_17_checked : candidates edge 4 467 4127 innerTail88_18 best.bound = innerCandidates88_17 := by
  unfold candidates
  have row : edge 4127 = SearchData.row4127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_17 : scan edge (search edge 3) 4 [467] 467 innerTail88_17 best = best := by
  rw [innerTail88_17, scan, if_neg (by decide), innerCandidates88_17_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_18
    (if 3 ≤ innerCandidates88_17.length then search edge 3 [467, 4127] innerCandidates88_17 4594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_18

def innerTail88_16 : List Nat := 4049 :: innerTail88_17
def innerCandidates88_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates88_16_checked : candidates edge 4 467 4049 innerTail88_17 best.bound = innerCandidates88_16 := by
  unfold candidates
  have row : edge 4049 = SearchData.row4049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_16 : scan edge (search edge 3) 4 [467] 467 innerTail88_16 best = best := by
  rw [innerTail88_16, scan, if_neg (by decide), innerCandidates88_16_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_17
    (if 3 ≤ innerCandidates88_16.length then search edge 3 [467, 4049] innerCandidates88_16 4516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_17

def innerTail88_15 : List Nat := 3989 :: innerTail88_16
def innerCandidates88_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates88_15_checked : candidates edge 4 467 3989 innerTail88_16 best.bound = innerCandidates88_15 := by
  unfold candidates
  have row : edge 3989 = SearchData.row3989.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_15 : scan edge (search edge 3) 4 [467] 467 innerTail88_15 best = best := by
  rw [innerTail88_15, scan, if_neg (by decide), innerCandidates88_15_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_16
    (if 3 ≤ innerCandidates88_15.length then search edge 3 [467, 3989] innerCandidates88_15 4456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_16

def innerTail88_14 : List Nat := 3917 :: innerTail88_15
def innerCandidates88_14 : List Nat := [6323, 6779]
set_option maxRecDepth 4096 in
theorem innerCandidates88_14_checked : candidates edge 4 467 3917 innerTail88_15 best.bound = innerCandidates88_14 := by
  unfold candidates
  have row : edge 3917 = SearchData.row3917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_14 : scan edge (search edge 3) 4 [467] 467 innerTail88_14 best = best := by
  rw [innerTail88_14, scan, if_neg (by decide), innerCandidates88_14_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_15
    (if 3 ≤ innerCandidates88_14.length then search edge 3 [467, 3917] innerCandidates88_14 4384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_15

def innerTail88_13 : List Nat := 2687 :: innerTail88_14
def innerCandidates88_13 : List Nat := [8231]
set_option maxRecDepth 4096 in
theorem innerCandidates88_13_checked : candidates edge 4 467 2687 innerTail88_14 best.bound = innerCandidates88_13 := by
  unfold candidates
  have row : edge 2687 = SearchData.row2687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_13 : scan edge (search edge 3) 4 [467] 467 innerTail88_13 best = best := by
  rw [innerTail88_13, scan, if_neg (by decide), innerCandidates88_13_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_14
    (if 3 ≤ innerCandidates88_13.length then search edge 3 [467, 2687] innerCandidates88_13 3154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_14

def innerTail88_12 : List Nat := 2621 :: innerTail88_13
def innerCandidates88_12 : List Nat := [5477, 6101]
set_option maxRecDepth 4096 in
theorem innerCandidates88_12_checked : candidates edge 4 467 2621 innerTail88_13 best.bound = innerCandidates88_12 := by
  unfold candidates
  have row : edge 2621 = PrimePairCache.smallRow2621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_12 : scan edge (search edge 3) 4 [467] 467 innerTail88_12 best = best := by
  rw [innerTail88_12, scan, if_neg (by decide), innerCandidates88_12_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_13
    (if 3 ≤ innerCandidates88_12.length then search edge 3 [467, 2621] innerCandidates88_12 3088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_13

def innerTail88_11 : List Nat := 2333 :: innerTail88_12
def innerCandidates88_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates88_11_checked : candidates edge 4 467 2333 innerTail88_12 best.bound = innerCandidates88_11 := by
  unfold candidates
  have row : edge 2333 = PrimePairCache.smallRow2333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_11 : scan edge (search edge 3) 4 [467] 467 innerTail88_11 best = best := by
  rw [innerTail88_11, scan, if_neg (by decide), innerCandidates88_11_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_12
    (if 3 ≤ innerCandidates88_11.length then search edge 3 [467, 2333] innerCandidates88_11 2800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_12

def innerTail88_10 : List Nat := 2141 :: innerTail88_11
def innerCandidates88_10 : List Nat := [2621, 4049]
set_option maxRecDepth 4096 in
theorem innerCandidates88_10_checked : candidates edge 4 467 2141 innerTail88_11 best.bound = innerCandidates88_10 := by
  unfold candidates
  have row : edge 2141 = PrimePairCache.smallRow2141.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_10 : scan edge (search edge 3) 4 [467] 467 innerTail88_10 best = best := by
  rw [innerTail88_10, scan, if_neg (by decide), innerCandidates88_10_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_11
    (if 3 ≤ innerCandidates88_10.length then search edge 3 [467, 2141] innerCandidates88_10 2608 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_11

def innerTail88_9 : List Nat := 2099 :: innerTail88_10
def innerCandidates88_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates88_9_checked : candidates edge 4 467 2099 innerTail88_10 best.bound = innerCandidates88_9 := by
  unfold candidates
  have row : edge 2099 = PrimePairCache.smallRow2099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_9 : scan edge (search edge 3) 4 [467] 467 innerTail88_9 best = best := by
  rw [innerTail88_9, scan, if_neg (by decide), innerCandidates88_9_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_10
    (if 3 ≤ innerCandidates88_9.length then search edge 3 [467, 2099] innerCandidates88_9 2566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_10

def innerTail88_8 : List Nat := 2039 :: innerTail88_9
def innerCandidates88_8 : List Nat := [3917]
set_option maxRecDepth 4096 in
theorem innerCandidates88_8_checked : candidates edge 4 467 2039 innerTail88_9 best.bound = innerCandidates88_8 := by
  unfold candidates
  have row : edge 2039 = PrimePairCache.smallRow2039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_8 : scan edge (search edge 3) 4 [467] 467 innerTail88_8 best = best := by
  rw [innerTail88_8, scan, if_neg (by decide), innerCandidates88_8_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_9
    (if 3 ≤ innerCandidates88_8.length then search edge 3 [467, 2039] innerCandidates88_8 2506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_9

def innerTail88_7 : List Nat := 1973 :: innerTail88_8
def innerCandidates88_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates88_7_checked : candidates edge 4 467 1973 innerTail88_8 best.bound = innerCandidates88_7 := by
  unfold candidates
  have row : edge 1973 = PrimePairCache.smallRow1973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_7 : scan edge (search edge 3) 4 [467] 467 innerTail88_7 best = best := by
  rw [innerTail88_7, scan, if_neg (by decide), innerCandidates88_7_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_8
    (if 3 ≤ innerCandidates88_7.length then search edge 3 [467, 1973] innerCandidates88_7 2440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_8

def innerTail88_6 : List Nat := 1913 :: innerTail88_7
def innerCandidates88_6 : List Nat := [5861]
set_option maxRecDepth 4096 in
theorem innerCandidates88_6_checked : candidates edge 4 467 1913 innerTail88_7 best.bound = innerCandidates88_6 := by
  unfold candidates
  have row : edge 1913 = PrimePairCache.smallRow1913.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_6 : scan edge (search edge 3) 4 [467] 467 innerTail88_6 best = best := by
  rw [innerTail88_6, scan, if_neg (by decide), innerCandidates88_6_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_7
    (if 3 ≤ innerCandidates88_6.length then search edge 3 [467, 1913] innerCandidates88_6 2380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_7

def innerTail88_5 : List Nat := 1571 :: innerTail88_6
def innerCandidates88_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates88_5_checked : candidates edge 4 467 1571 innerTail88_6 best.bound = innerCandidates88_5 := by
  unfold candidates
  have row : edge 1571 = PrimePairCache.smallRow1571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_5 : scan edge (search edge 3) 4 [467] 467 innerTail88_5 best = best := by
  rw [innerTail88_5, scan, if_neg (by decide), innerCandidates88_5_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_6
    (if 3 ≤ innerCandidates88_5.length then search edge 3 [467, 1571] innerCandidates88_5 2038 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_6

def innerTail88_4 : List Nat := 1301 :: innerTail88_5
def innerCandidates88_4 : List Nat := [2141]
set_option maxRecDepth 4096 in
theorem innerCandidates88_4_checked : candidates edge 4 467 1301 innerTail88_5 best.bound = innerCandidates88_4 := by
  unfold candidates
  have row : edge 1301 = PrimePairCache.smallRow1301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_4 : scan edge (search edge 3) 4 [467] 467 innerTail88_4 best = best := by
  rw [innerTail88_4, scan, if_neg (by decide), innerCandidates88_4_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_5
    (if 3 ≤ innerCandidates88_4.length then search edge 3 [467, 1301] innerCandidates88_4 1768 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_5

def innerTail88_3 : List Nat := 941 :: innerTail88_4
def innerCandidates88_3 : List Nat := [2099]
set_option maxRecDepth 4096 in
theorem innerCandidates88_3_checked : candidates edge 4 467 941 innerTail88_4 best.bound = innerCandidates88_3 := by
  unfold candidates
  have row : edge 941 = PrimePairCache.smallRow941.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_3 : scan edge (search edge 3) 4 [467] 467 innerTail88_3 best = best := by
  rw [innerTail88_3, scan, if_neg (by decide), innerCandidates88_3_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_4
    (if 3 ≤ innerCandidates88_3.length then search edge 3 [467, 941] innerCandidates88_3 1408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_4

def innerTail88_2 : List Nat := 641 :: innerTail88_3
def innerCandidates88_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates88_2_checked : candidates edge 4 467 641 innerTail88_3 best.bound = innerCandidates88_2 := by
  unfold candidates
  have row : edge 641 = PrimePairCache.smallRow641.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan88_2 : scan edge (search edge 3) 4 [467] 467 innerTail88_2 best = best := by
  rw [innerTail88_2, scan, if_neg (by decide), innerCandidates88_2_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_3
    (if 3 ≤ innerCandidates88_2.length then search edge 3 [467, 641] innerCandidates88_2 1108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan88_3

def innerTail88_1 : List Nat := 617 :: innerTail88_2
def innerCandidates88_1 : List Nat := [1301, 4253, 6323]
set_option maxRecDepth 4096 in
theorem innerCandidates88_1_checked : candidates edge 4 467 617 innerTail88_2 best.bound = innerCandidates88_1 := by
  unfold candidates
  have row : edge 617 = PrimePairCache.smallRow617.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild88_1 : search edge 3 [467, 617] innerCandidates88_1 1084 best = best := by decide

theorem innerScan88_1 : scan edge (search edge 3) 4 [467] 467 innerTail88_1 best = best := by
  rw [innerTail88_1, scan, if_neg (by decide), innerCandidates88_1_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_2
    (if 3 ≤ innerCandidates88_1.length then search edge 3 [467, 617] innerCandidates88_1 1084 best else best) = best
  rw [if_pos (by decide), innerChild88_1]
  exact innerScan88_2

def innerTail88_0 : List Nat := 587 :: innerTail88_1
def innerCandidates88_0 : List Nat := [617, 1571, 6323]
set_option maxRecDepth 4096 in
theorem innerCandidates88_0_checked : candidates edge 4 467 587 innerTail88_1 best.bound = innerCandidates88_0 := by
  unfold candidates
  have row : edge 587 = PrimePairCache.smallRow587.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild88_0 : search edge 3 [467, 587] innerCandidates88_0 1054 best = best := by decide

theorem innerScan88_0 : scan edge (search edge 3) 4 [467] 467 innerTail88_0 best = best := by
  rw [innerTail88_0, scan, if_neg (by decide), innerCandidates88_0_checked]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_1
    (if 3 ≤ innerCandidates88_0.length then search edge 3 [467, 587] innerCandidates88_0 1054 best else best) = best
  rw [if_pos (by decide), innerChild88_0]
  exact innerScan88_1

theorem rootBranch88 : search edge 4 [467] rootCandidates88 467 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [467] 467 innerTail88_0 best = best
  exact innerScan88_0

def rootCandidates87 : List Nat := [523, 613, 643, 829, 1171, 1489, 1663, 2473, 2917, 3001, 3259, 3331, 3457, 3793, 3931, 4111, 4723, 4933, 5437, 5569, 5623, 6451, 6967, 7333, 7489, 7669]
set_option maxRecDepth 4096 in
theorem rootCandidates87_checked : candidates edge 5 0 463 tail88 best.bound = rootCandidates87 := by
  unfold candidates
  have row : edge 463 = PrimePairCache.smallRow463.contains := by funext q; rfl
  rw [row]
  decide

def innerTail87_26 : List Nat := []
theorem innerScan87_26 : scan edge (search edge 3) 4 [463] 463 innerTail87_26 best = best := rfl

def innerTail87_25 : List Nat := 7669 :: innerTail87_26
def innerCandidates87_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates87_25_checked : candidates edge 4 463 7669 innerTail87_26 best.bound = innerCandidates87_25 := by
  unfold candidates
  have row : edge 7669 = SearchData.row7669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_25 : scan edge (search edge 3) 4 [463] 463 innerTail87_25 best = best := by
  rw [innerTail87_25, scan, if_neg (by decide), innerCandidates87_25_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_26
    (if 3 ≤ innerCandidates87_25.length then search edge 3 [463, 7669] innerCandidates87_25 8132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_26

def innerTail87_24 : List Nat := 7489 :: innerTail87_25
def innerCandidates87_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates87_24_checked : candidates edge 4 463 7489 innerTail87_25 best.bound = innerCandidates87_24 := by
  unfold candidates
  have row : edge 7489 = SearchData.row7489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_24 : scan edge (search edge 3) 4 [463] 463 innerTail87_24 best = best := by
  rw [innerTail87_24, scan, if_neg (by decide), innerCandidates87_24_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_25
    (if 3 ≤ innerCandidates87_24.length then search edge 3 [463, 7489] innerCandidates87_24 7952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_25

def innerTail87_23 : List Nat := 7333 :: innerTail87_24
def innerCandidates87_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates87_23_checked : candidates edge 4 463 7333 innerTail87_24 best.bound = innerCandidates87_23 := by
  unfold candidates
  have row : edge 7333 = SearchData.row7333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_23 : scan edge (search edge 3) 4 [463] 463 innerTail87_23 best = best := by
  rw [innerTail87_23, scan, if_neg (by decide), innerCandidates87_23_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_24
    (if 3 ≤ innerCandidates87_23.length then search edge 3 [463, 7333] innerCandidates87_23 7796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_24

def innerTail87_22 : List Nat := 6967 :: innerTail87_23
def innerCandidates87_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates87_22_checked : candidates edge 4 463 6967 innerTail87_23 best.bound = innerCandidates87_22 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_22 : scan edge (search edge 3) 4 [463] 463 innerTail87_22 best = best := by
  rw [innerTail87_22, scan, if_neg (by decide), innerCandidates87_22_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_23
    (if 3 ≤ innerCandidates87_22.length then search edge 3 [463, 6967] innerCandidates87_22 7430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_23

def innerTail87_21 : List Nat := 6451 :: innerTail87_22
def innerCandidates87_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates87_21_checked : candidates edge 4 463 6451 innerTail87_22 best.bound = innerCandidates87_21 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_21 : scan edge (search edge 3) 4 [463] 463 innerTail87_21 best = best := by
  rw [innerTail87_21, scan, if_neg (by decide), innerCandidates87_21_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_22
    (if 3 ≤ innerCandidates87_21.length then search edge 3 [463, 6451] innerCandidates87_21 6914 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_22

def innerTail87_20 : List Nat := 5623 :: innerTail87_21
def innerCandidates87_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates87_20_checked : candidates edge 4 463 5623 innerTail87_21 best.bound = innerCandidates87_20 := by
  unfold candidates
  have row : edge 5623 = SearchData.row5623.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_20 : scan edge (search edge 3) 4 [463] 463 innerTail87_20 best = best := by
  rw [innerTail87_20, scan, if_neg (by decide), innerCandidates87_20_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_21
    (if 3 ≤ innerCandidates87_20.length then search edge 3 [463, 5623] innerCandidates87_20 6086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_21

def innerTail87_19 : List Nat := 5569 :: innerTail87_20
def innerCandidates87_19 : List Nat := [6451]
set_option maxRecDepth 4096 in
theorem innerCandidates87_19_checked : candidates edge 4 463 5569 innerTail87_20 best.bound = innerCandidates87_19 := by
  unfold candidates
  have row : edge 5569 = SearchData.row5569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_19 : scan edge (search edge 3) 4 [463] 463 innerTail87_19 best = best := by
  rw [innerTail87_19, scan, if_neg (by decide), innerCandidates87_19_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_20
    (if 3 ≤ innerCandidates87_19.length then search edge 3 [463, 5569] innerCandidates87_19 6032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_20

def innerTail87_18 : List Nat := 5437 :: innerTail87_19
def innerCandidates87_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates87_18_checked : candidates edge 4 463 5437 innerTail87_19 best.bound = innerCandidates87_18 := by
  unfold candidates
  have row : edge 5437 = SearchData.row5437.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_18 : scan edge (search edge 3) 4 [463] 463 innerTail87_18 best = best := by
  rw [innerTail87_18, scan, if_neg (by decide), innerCandidates87_18_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_19
    (if 3 ≤ innerCandidates87_18.length then search edge 3 [463, 5437] innerCandidates87_18 5900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_19

def innerTail87_17 : List Nat := 4933 :: innerTail87_18
def innerCandidates87_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates87_17_checked : candidates edge 4 463 4933 innerTail87_18 best.bound = innerCandidates87_17 := by
  unfold candidates
  have row : edge 4933 = SearchData.row4933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_17 : scan edge (search edge 3) 4 [463] 463 innerTail87_17 best = best := by
  rw [innerTail87_17, scan, if_neg (by decide), innerCandidates87_17_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_18
    (if 3 ≤ innerCandidates87_17.length then search edge 3 [463, 4933] innerCandidates87_17 5396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_18

def innerTail87_16 : List Nat := 4723 :: innerTail87_17
def innerCandidates87_16 : List Nat := [5623]
set_option maxRecDepth 4096 in
theorem innerCandidates87_16_checked : candidates edge 4 463 4723 innerTail87_17 best.bound = innerCandidates87_16 := by
  unfold candidates
  have row : edge 4723 = SearchData.row4723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_16 : scan edge (search edge 3) 4 [463] 463 innerTail87_16 best = best := by
  rw [innerTail87_16, scan, if_neg (by decide), innerCandidates87_16_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_17
    (if 3 ≤ innerCandidates87_16.length then search edge 3 [463, 4723] innerCandidates87_16 5186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_17

def innerTail87_15 : List Nat := 4111 :: innerTail87_16
def innerCandidates87_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates87_15_checked : candidates edge 4 463 4111 innerTail87_16 best.bound = innerCandidates87_15 := by
  unfold candidates
  have row : edge 4111 = SearchData.row4111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_15 : scan edge (search edge 3) 4 [463] 463 innerTail87_15 best = best := by
  rw [innerTail87_15, scan, if_neg (by decide), innerCandidates87_15_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_16
    (if 3 ≤ innerCandidates87_15.length then search edge 3 [463, 4111] innerCandidates87_15 4574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_16

def innerTail87_14 : List Nat := 3931 :: innerTail87_15
def innerCandidates87_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates87_14_checked : candidates edge 4 463 3931 innerTail87_15 best.bound = innerCandidates87_14 := by
  unfold candidates
  have row : edge 3931 = SearchData.row3931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_14 : scan edge (search edge 3) 4 [463] 463 innerTail87_14 best = best := by
  rw [innerTail87_14, scan, if_neg (by decide), innerCandidates87_14_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_15
    (if 3 ≤ innerCandidates87_14.length then search edge 3 [463, 3931] innerCandidates87_14 4394 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_15

def innerTail87_13 : List Nat := 3793 :: innerTail87_14
def innerCandidates87_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates87_13_checked : candidates edge 4 463 3793 innerTail87_14 best.bound = innerCandidates87_13 := by
  unfold candidates
  have row : edge 3793 = SearchData.row3793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_13 : scan edge (search edge 3) 4 [463] 463 innerTail87_13 best = best := by
  rw [innerTail87_13, scan, if_neg (by decide), innerCandidates87_13_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_14
    (if 3 ≤ innerCandidates87_13.length then search edge 3 [463, 3793] innerCandidates87_13 4256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_14

def innerTail87_12 : List Nat := 3457 :: innerTail87_13
def innerCandidates87_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates87_12_checked : candidates edge 4 463 3457 innerTail87_13 best.bound = innerCandidates87_12 := by
  unfold candidates
  have row : edge 3457 = SearchData.row3457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_12 : scan edge (search edge 3) 4 [463] 463 innerTail87_12 best = best := by
  rw [innerTail87_12, scan, if_neg (by decide), innerCandidates87_12_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_13
    (if 3 ≤ innerCandidates87_12.length then search edge 3 [463, 3457] innerCandidates87_12 3920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_13

def innerTail87_11 : List Nat := 3331 :: innerTail87_12
def innerCandidates87_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates87_11_checked : candidates edge 4 463 3331 innerTail87_12 best.bound = innerCandidates87_11 := by
  unfold candidates
  have row : edge 3331 = SearchData.row3331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_11 : scan edge (search edge 3) 4 [463] 463 innerTail87_11 best = best := by
  rw [innerTail87_11, scan, if_neg (by decide), innerCandidates87_11_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_12
    (if 3 ≤ innerCandidates87_11.length then search edge 3 [463, 3331] innerCandidates87_11 3794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_12

def innerTail87_10 : List Nat := 3259 :: innerTail87_11
def innerCandidates87_10 : List Nat := [5569]
set_option maxRecDepth 4096 in
theorem innerCandidates87_10_checked : candidates edge 4 463 3259 innerTail87_11 best.bound = innerCandidates87_10 := by
  unfold candidates
  have row : edge 3259 = SearchData.row3259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_10 : scan edge (search edge 3) 4 [463] 463 innerTail87_10 best = best := by
  rw [innerTail87_10, scan, if_neg (by decide), innerCandidates87_10_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_11
    (if 3 ≤ innerCandidates87_10.length then search edge 3 [463, 3259] innerCandidates87_10 3722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_11

def innerTail87_9 : List Nat := 3001 :: innerTail87_10
def innerCandidates87_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates87_9_checked : candidates edge 4 463 3001 innerTail87_10 best.bound = innerCandidates87_9 := by
  unfold candidates
  have row : edge 3001 = SearchData.row3001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_9 : scan edge (search edge 3) 4 [463] 463 innerTail87_9 best = best := by
  rw [innerTail87_9, scan, if_neg (by decide), innerCandidates87_9_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_10
    (if 3 ≤ innerCandidates87_9.length then search edge 3 [463, 3001] innerCandidates87_9 3464 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_10

def innerTail87_8 : List Nat := 2917 :: innerTail87_9
def innerCandidates87_8 : List Nat := [5437]
set_option maxRecDepth 4096 in
theorem innerCandidates87_8_checked : candidates edge 4 463 2917 innerTail87_9 best.bound = innerCandidates87_8 := by
  unfold candidates
  have row : edge 2917 = SearchData.row2917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_8 : scan edge (search edge 3) 4 [463] 463 innerTail87_8 best = best := by
  rw [innerTail87_8, scan, if_neg (by decide), innerCandidates87_8_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_9
    (if 3 ≤ innerCandidates87_8.length then search edge 3 [463, 2917] innerCandidates87_8 3380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_9

def innerTail87_7 : List Nat := 2473 :: innerTail87_8
def innerCandidates87_7 : List Nat := [3001, 5437]
set_option maxRecDepth 4096 in
theorem innerCandidates87_7_checked : candidates edge 4 463 2473 innerTail87_8 best.bound = innerCandidates87_7 := by
  unfold candidates
  have row : edge 2473 = PrimePairCache.smallRow2473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_7 : scan edge (search edge 3) 4 [463] 463 innerTail87_7 best = best := by
  rw [innerTail87_7, scan, if_neg (by decide), innerCandidates87_7_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_8
    (if 3 ≤ innerCandidates87_7.length then search edge 3 [463, 2473] innerCandidates87_7 2936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_8

def innerTail87_6 : List Nat := 1663 :: innerTail87_7
def innerCandidates87_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates87_6_checked : candidates edge 4 463 1663 innerTail87_7 best.bound = innerCandidates87_6 := by
  unfold candidates
  have row : edge 1663 = PrimePairCache.smallRow1663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_6 : scan edge (search edge 3) 4 [463] 463 innerTail87_6 best = best := by
  rw [innerTail87_6, scan, if_neg (by decide), innerCandidates87_6_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_7
    (if 3 ≤ innerCandidates87_6.length then search edge 3 [463, 1663] innerCandidates87_6 2126 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_7

def innerTail87_5 : List Nat := 1489 :: innerTail87_6
def innerCandidates87_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates87_5_checked : candidates edge 4 463 1489 innerTail87_6 best.bound = innerCandidates87_5 := by
  unfold candidates
  have row : edge 1489 = PrimePairCache.smallRow1489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_5 : scan edge (search edge 3) 4 [463] 463 innerTail87_5 best = best := by
  rw [innerTail87_5, scan, if_neg (by decide), innerCandidates87_5_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_6
    (if 3 ≤ innerCandidates87_5.length then search edge 3 [463, 1489] innerCandidates87_5 1952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_6

def innerTail87_4 : List Nat := 1171 :: innerTail87_5
def innerCandidates87_4 : List Nat := [4933]
set_option maxRecDepth 4096 in
theorem innerCandidates87_4_checked : candidates edge 4 463 1171 innerTail87_5 best.bound = innerCandidates87_4 := by
  unfold candidates
  have row : edge 1171 = PrimePairCache.smallRow1171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_4 : scan edge (search edge 3) 4 [463] 463 innerTail87_4 best = best := by
  rw [innerTail87_4, scan, if_neg (by decide), innerCandidates87_4_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_5
    (if 3 ≤ innerCandidates87_4.length then search edge 3 [463, 1171] innerCandidates87_4 1634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_5

def innerTail87_3 : List Nat := 829 :: innerTail87_4
def innerCandidates87_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates87_3_checked : candidates edge 4 463 829 innerTail87_4 best.bound = innerCandidates87_3 := by
  unfold candidates
  have row : edge 829 = PrimePairCache.smallRow829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_3 : scan edge (search edge 3) 4 [463] 463 innerTail87_3 best = best := by
  rw [innerTail87_3, scan, if_neg (by decide), innerCandidates87_3_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_4
    (if 3 ≤ innerCandidates87_3.length then search edge 3 [463, 829] innerCandidates87_3 1292 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_4

def innerTail87_2 : List Nat := 643 :: innerTail87_3
def innerCandidates87_2 : List Nat := [2917, 4111]
set_option maxRecDepth 4096 in
theorem innerCandidates87_2_checked : candidates edge 4 463 643 innerTail87_3 best.bound = innerCandidates87_2 := by
  unfold candidates
  have row : edge 643 = PrimePairCache.smallRow643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_2 : scan edge (search edge 3) 4 [463] 463 innerTail87_2 best = best := by
  rw [innerTail87_2, scan, if_neg (by decide), innerCandidates87_2_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_3
    (if 3 ≤ innerCandidates87_2.length then search edge 3 [463, 643] innerCandidates87_2 1106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_3

def innerTail87_1 : List Nat := 613 :: innerTail87_2
def innerCandidates87_1 : List Nat := [829, 6967]
set_option maxRecDepth 4096 in
theorem innerCandidates87_1_checked : candidates edge 4 463 613 innerTail87_2 best.bound = innerCandidates87_1 := by
  unfold candidates
  have row : edge 613 = PrimePairCache.smallRow613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_1 : scan edge (search edge 3) 4 [463] 463 innerTail87_1 best = best := by
  rw [innerTail87_1, scan, if_neg (by decide), innerCandidates87_1_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_2
    (if 3 ≤ innerCandidates87_1.length then search edge 3 [463, 613] innerCandidates87_1 1076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_2

def innerTail87_0 : List Nat := 523 :: innerTail87_1
def innerCandidates87_0 : List Nat := [1663, 6967]
set_option maxRecDepth 4096 in
theorem innerCandidates87_0_checked : candidates edge 4 463 523 innerTail87_1 best.bound = innerCandidates87_0 := by
  unfold candidates
  have row : edge 523 = PrimePairCache.smallRow523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan87_0 : scan edge (search edge 3) 4 [463] 463 innerTail87_0 best = best := by
  rw [innerTail87_0, scan, if_neg (by decide), innerCandidates87_0_checked]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_1
    (if 3 ≤ innerCandidates87_0.length then search edge 3 [463, 523] innerCandidates87_0 986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan87_1

theorem rootBranch87 : search edge 4 [463] rootCandidates87 463 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [463] 463 innerTail87_0 best = best
  exact innerScan87_0

end ProjectEulerSolutions.PrimePairUpper.Search
