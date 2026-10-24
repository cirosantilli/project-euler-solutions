import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates236 : List Nat := [1619, 2129, 2309, 2543, 2633, 2801, 2897, 3359, 3947, 4013, 4253, 6323, 6521, 6563, 7121, 7331, 7589, 8171, 8273]
set_option maxRecDepth 4096 in
theorem rootCandidates236_checked : candidates edge 5 0 1499 tail237 best.bound = rootCandidates236 := by
  unfold candidates
  have row : edge 1499 = PrimePairCache.smallRow1499.contains := by funext q; rfl
  rw [row]
  decide

def innerTail236_19 : List Nat := []
theorem innerScan236_19 : scan edge (search edge 3) 4 [1499] 1499 innerTail236_19 best = best := rfl

def innerTail236_18 : List Nat := 8273 :: innerTail236_19
def innerCandidates236_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates236_18_checked : candidates edge 4 1499 8273 innerTail236_19 best.bound = innerCandidates236_18 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan236_18 : scan edge (search edge 3) 4 [1499] 1499 innerTail236_18 best = best := by
  rw [innerTail236_18, scan, if_neg (by decide), innerCandidates236_18_checked]
  change scan edge (search edge 3) 4 [1499] 1499 innerTail236_19
    (if 3 ≤ innerCandidates236_18.length then search edge 3 [1499, 8273] innerCandidates236_18 9772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan236_19

def innerTail236_17 : List Nat := 8171 :: innerTail236_18
def innerCandidates236_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates236_17_checked : candidates edge 4 1499 8171 innerTail236_18 best.bound = innerCandidates236_17 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan236_17 : scan edge (search edge 3) 4 [1499] 1499 innerTail236_17 best = best := by
  rw [innerTail236_17, scan, if_neg (by decide), innerCandidates236_17_checked]
  change scan edge (search edge 3) 4 [1499] 1499 innerTail236_18
    (if 3 ≤ innerCandidates236_17.length then search edge 3 [1499, 8171] innerCandidates236_17 9670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan236_18

def innerTail236_16 : List Nat := 7589 :: innerTail236_17
def innerCandidates236_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates236_16_checked : candidates edge 4 1499 7589 innerTail236_17 best.bound = innerCandidates236_16 := by
  unfold candidates
  have row : edge 7589 = SearchData.row7589.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan236_16 : scan edge (search edge 3) 4 [1499] 1499 innerTail236_16 best = best := by
  rw [innerTail236_16, scan, if_neg (by decide), innerCandidates236_16_checked]
  change scan edge (search edge 3) 4 [1499] 1499 innerTail236_17
    (if 3 ≤ innerCandidates236_16.length then search edge 3 [1499, 7589] innerCandidates236_16 9088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan236_17

def innerTail236_15 : List Nat := 7331 :: innerTail236_16
def innerCandidates236_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates236_15_checked : candidates edge 4 1499 7331 innerTail236_16 best.bound = innerCandidates236_15 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan236_15 : scan edge (search edge 3) 4 [1499] 1499 innerTail236_15 best = best := by
  rw [innerTail236_15, scan, if_neg (by decide), innerCandidates236_15_checked]
  change scan edge (search edge 3) 4 [1499] 1499 innerTail236_16
    (if 3 ≤ innerCandidates236_15.length then search edge 3 [1499, 7331] innerCandidates236_15 8830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan236_16

def innerTail236_14 : List Nat := 7121 :: innerTail236_15
def innerCandidates236_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates236_14_checked : candidates edge 4 1499 7121 innerTail236_15 best.bound = innerCandidates236_14 := by
  unfold candidates
  have row : edge 7121 = SearchData.row7121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan236_14 : scan edge (search edge 3) 4 [1499] 1499 innerTail236_14 best = best := by
  rw [innerTail236_14, scan, if_neg (by decide), innerCandidates236_14_checked]
  change scan edge (search edge 3) 4 [1499] 1499 innerTail236_15
    (if 3 ≤ innerCandidates236_14.length then search edge 3 [1499, 7121] innerCandidates236_14 8620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan236_15

def innerTail236_13 : List Nat := 6563 :: innerTail236_14
def innerCandidates236_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates236_13_checked : candidates edge 4 1499 6563 innerTail236_14 best.bound = innerCandidates236_13 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan236_13 : scan edge (search edge 3) 4 [1499] 1499 innerTail236_13 best = best := by
  rw [innerTail236_13, scan, if_neg (by decide), innerCandidates236_13_checked]
  change scan edge (search edge 3) 4 [1499] 1499 innerTail236_14
    (if 3 ≤ innerCandidates236_13.length then search edge 3 [1499, 6563] innerCandidates236_13 8062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan236_14

def innerTail236_12 : List Nat := 6521 :: innerTail236_13
def innerCandidates236_12 : List Nat := [7589]
set_option maxRecDepth 4096 in
theorem innerCandidates236_12_checked : candidates edge 4 1499 6521 innerTail236_13 best.bound = innerCandidates236_12 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan236_12 : scan edge (search edge 3) 4 [1499] 1499 innerTail236_12 best = best := by
  rw [innerTail236_12, scan, if_neg (by decide), innerCandidates236_12_checked]
  change scan edge (search edge 3) 4 [1499] 1499 innerTail236_13
    (if 3 ≤ innerCandidates236_12.length then search edge 3 [1499, 6521] innerCandidates236_12 8020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan236_13

def innerTail236_11 : List Nat := 6323 :: innerTail236_12
def innerCandidates236_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates236_11_checked : candidates edge 4 1499 6323 innerTail236_12 best.bound = innerCandidates236_11 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan236_11 : scan edge (search edge 3) 4 [1499] 1499 innerTail236_11 best = best := by
  rw [innerTail236_11, scan, if_neg (by decide), innerCandidates236_11_checked]
  change scan edge (search edge 3) 4 [1499] 1499 innerTail236_12
    (if 3 ≤ innerCandidates236_11.length then search edge 3 [1499, 6323] innerCandidates236_11 7822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan236_12

def innerTail236_10 : List Nat := 4253 :: innerTail236_11
def innerCandidates236_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates236_10_checked : candidates edge 4 1499 4253 innerTail236_11 best.bound = innerCandidates236_10 := by
  unfold candidates
  have row : edge 4253 = SearchData.row4253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan236_10 : scan edge (search edge 3) 4 [1499] 1499 innerTail236_10 best = best := by
  rw [innerTail236_10, scan, if_neg (by decide), innerCandidates236_10_checked]
  change scan edge (search edge 3) 4 [1499] 1499 innerTail236_11
    (if 3 ≤ innerCandidates236_10.length then search edge 3 [1499, 4253] innerCandidates236_10 5752 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan236_11

def innerTail236_9 : List Nat := 4013 :: innerTail236_10
def innerCandidates236_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates236_9_checked : candidates edge 4 1499 4013 innerTail236_10 best.bound = innerCandidates236_9 := by
  unfold candidates
  have row : edge 4013 = SearchData.row4013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan236_9 : scan edge (search edge 3) 4 [1499] 1499 innerTail236_9 best = best := by
  rw [innerTail236_9, scan, if_neg (by decide), innerCandidates236_9_checked]
  change scan edge (search edge 3) 4 [1499] 1499 innerTail236_10
    (if 3 ≤ innerCandidates236_9.length then search edge 3 [1499, 4013] innerCandidates236_9 5512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan236_10

def innerTail236_8 : List Nat := 3947 :: innerTail236_9
def innerCandidates236_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates236_8_checked : candidates edge 4 1499 3947 innerTail236_9 best.bound = innerCandidates236_8 := by
  unfold candidates
  have row : edge 3947 = SearchData.row3947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan236_8 : scan edge (search edge 3) 4 [1499] 1499 innerTail236_8 best = best := by
  rw [innerTail236_8, scan, if_neg (by decide), innerCandidates236_8_checked]
  change scan edge (search edge 3) 4 [1499] 1499 innerTail236_9
    (if 3 ≤ innerCandidates236_8.length then search edge 3 [1499, 3947] innerCandidates236_8 5446 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan236_9

def innerTail236_7 : List Nat := 3359 :: innerTail236_8
def innerCandidates236_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates236_7_checked : candidates edge 4 1499 3359 innerTail236_8 best.bound = innerCandidates236_7 := by
  unfold candidates
  have row : edge 3359 = SearchData.row3359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan236_7 : scan edge (search edge 3) 4 [1499] 1499 innerTail236_7 best = best := by
  rw [innerTail236_7, scan, if_neg (by decide), innerCandidates236_7_checked]
  change scan edge (search edge 3) 4 [1499] 1499 innerTail236_8
    (if 3 ≤ innerCandidates236_7.length then search edge 3 [1499, 3359] innerCandidates236_7 4858 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan236_8

def innerTail236_6 : List Nat := 2897 :: innerTail236_7
def innerCandidates236_6 : List Nat := [3947]
set_option maxRecDepth 4096 in
theorem innerCandidates236_6_checked : candidates edge 4 1499 2897 innerTail236_7 best.bound = innerCandidates236_6 := by
  unfold candidates
  have row : edge 2897 = SearchData.row2897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan236_6 : scan edge (search edge 3) 4 [1499] 1499 innerTail236_6 best = best := by
  rw [innerTail236_6, scan, if_neg (by decide), innerCandidates236_6_checked]
  change scan edge (search edge 3) 4 [1499] 1499 innerTail236_7
    (if 3 ≤ innerCandidates236_6.length then search edge 3 [1499, 2897] innerCandidates236_6 4396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan236_7

def innerTail236_5 : List Nat := 2801 :: innerTail236_6
def innerCandidates236_5 : List Nat := [7331, 8171]
set_option maxRecDepth 4096 in
theorem innerCandidates236_5_checked : candidates edge 4 1499 2801 innerTail236_6 best.bound = innerCandidates236_5 := by
  unfold candidates
  have row : edge 2801 = SearchData.row2801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan236_5 : scan edge (search edge 3) 4 [1499] 1499 innerTail236_5 best = best := by
  rw [innerTail236_5, scan, if_neg (by decide), innerCandidates236_5_checked]
  change scan edge (search edge 3) 4 [1499] 1499 innerTail236_6
    (if 3 ≤ innerCandidates236_5.length then search edge 3 [1499, 2801] innerCandidates236_5 4300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan236_6

def innerTail236_4 : List Nat := 2633 :: innerTail236_5
def innerCandidates236_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates236_4_checked : candidates edge 4 1499 2633 innerTail236_5 best.bound = innerCandidates236_4 := by
  unfold candidates
  have row : edge 2633 = PrimePairCache.smallRow2633.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan236_4 : scan edge (search edge 3) 4 [1499] 1499 innerTail236_4 best = best := by
  rw [innerTail236_4, scan, if_neg (by decide), innerCandidates236_4_checked]
  change scan edge (search edge 3) 4 [1499] 1499 innerTail236_5
    (if 3 ≤ innerCandidates236_4.length then search edge 3 [1499, 2633] innerCandidates236_4 4132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan236_5

def innerTail236_3 : List Nat := 2543 :: innerTail236_4
def innerCandidates236_3 : List Nat := [7331]
set_option maxRecDepth 4096 in
theorem innerCandidates236_3_checked : candidates edge 4 1499 2543 innerTail236_4 best.bound = innerCandidates236_3 := by
  unfold candidates
  have row : edge 2543 = PrimePairCache.smallRow2543.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan236_3 : scan edge (search edge 3) 4 [1499] 1499 innerTail236_3 best = best := by
  rw [innerTail236_3, scan, if_neg (by decide), innerCandidates236_3_checked]
  change scan edge (search edge 3) 4 [1499] 1499 innerTail236_4
    (if 3 ≤ innerCandidates236_3.length then search edge 3 [1499, 2543] innerCandidates236_3 4042 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan236_4

def innerTail236_2 : List Nat := 2309 :: innerTail236_3
def innerCandidates236_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates236_2_checked : candidates edge 4 1499 2309 innerTail236_3 best.bound = innerCandidates236_2 := by
  unfold candidates
  have row : edge 2309 = PrimePairCache.smallRow2309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan236_2 : scan edge (search edge 3) 4 [1499] 1499 innerTail236_2 best = best := by
  rw [innerTail236_2, scan, if_neg (by decide), innerCandidates236_2_checked]
  change scan edge (search edge 3) 4 [1499] 1499 innerTail236_3
    (if 3 ≤ innerCandidates236_2.length then search edge 3 [1499, 2309] innerCandidates236_2 3808 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan236_3

def innerTail236_1 : List Nat := 2129 :: innerTail236_2
def innerCandidates236_1 : List Nat := [3947, 7121]
set_option maxRecDepth 4096 in
theorem innerCandidates236_1_checked : candidates edge 4 1499 2129 innerTail236_2 best.bound = innerCandidates236_1 := by
  unfold candidates
  have row : edge 2129 = PrimePairCache.smallRow2129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan236_1 : scan edge (search edge 3) 4 [1499] 1499 innerTail236_1 best = best := by
  rw [innerTail236_1, scan, if_neg (by decide), innerCandidates236_1_checked]
  change scan edge (search edge 3) 4 [1499] 1499 innerTail236_2
    (if 3 ≤ innerCandidates236_1.length then search edge 3 [1499, 2129] innerCandidates236_1 3628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan236_2

def innerTail236_0 : List Nat := 1619 :: innerTail236_1
def innerCandidates236_0 : List Nat := [7331]
set_option maxRecDepth 4096 in
theorem innerCandidates236_0_checked : candidates edge 4 1499 1619 innerTail236_1 best.bound = innerCandidates236_0 := by
  unfold candidates
  have row : edge 1619 = PrimePairCache.smallRow1619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan236_0 : scan edge (search edge 3) 4 [1499] 1499 innerTail236_0 best = best := by
  rw [innerTail236_0, scan, if_neg (by decide), innerCandidates236_0_checked]
  change scan edge (search edge 3) 4 [1499] 1499 innerTail236_1
    (if 3 ≤ innerCandidates236_0.length then search edge 3 [1499, 1619] innerCandidates236_0 3118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan236_1

theorem rootBranch236 : search edge 4 [1499] rootCandidates236 1499 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1499] 1499 innerTail236_0 best = best
  exact innerScan236_0

def rootCandidates235 : List Nat := [1523, 2339, 2441, 2609, 3329, 3623, 4649, 4943, 5021, 5189, 5279, 5651, 5867, 6599, 7229, 7457, 7541, 7607]
set_option maxRecDepth 4096 in
theorem rootCandidates235_checked : candidates edge 5 0 1493 tail236 best.bound = rootCandidates235 := by
  unfold candidates
  have row : edge 1493 = PrimePairCache.smallRow1493.contains := by funext q; rfl
  rw [row]
  decide

def innerTail235_18 : List Nat := []
theorem innerScan235_18 : scan edge (search edge 3) 4 [1493] 1493 innerTail235_18 best = best := rfl

def innerTail235_17 : List Nat := 7607 :: innerTail235_18
def innerCandidates235_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates235_17_checked : candidates edge 4 1493 7607 innerTail235_18 best.bound = innerCandidates235_17 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan235_17 : scan edge (search edge 3) 4 [1493] 1493 innerTail235_17 best = best := by
  rw [innerTail235_17, scan, if_neg (by decide), innerCandidates235_17_checked]
  change scan edge (search edge 3) 4 [1493] 1493 innerTail235_18
    (if 3 ≤ innerCandidates235_17.length then search edge 3 [1493, 7607] innerCandidates235_17 9100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan235_18

def innerTail235_16 : List Nat := 7541 :: innerTail235_17
def innerCandidates235_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates235_16_checked : candidates edge 4 1493 7541 innerTail235_17 best.bound = innerCandidates235_16 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan235_16 : scan edge (search edge 3) 4 [1493] 1493 innerTail235_16 best = best := by
  rw [innerTail235_16, scan, if_neg (by decide), innerCandidates235_16_checked]
  change scan edge (search edge 3) 4 [1493] 1493 innerTail235_17
    (if 3 ≤ innerCandidates235_16.length then search edge 3 [1493, 7541] innerCandidates235_16 9034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan235_17

def innerTail235_15 : List Nat := 7457 :: innerTail235_16
def innerCandidates235_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates235_15_checked : candidates edge 4 1493 7457 innerTail235_16 best.bound = innerCandidates235_15 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan235_15 : scan edge (search edge 3) 4 [1493] 1493 innerTail235_15 best = best := by
  rw [innerTail235_15, scan, if_neg (by decide), innerCandidates235_15_checked]
  change scan edge (search edge 3) 4 [1493] 1493 innerTail235_16
    (if 3 ≤ innerCandidates235_15.length then search edge 3 [1493, 7457] innerCandidates235_15 8950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan235_16

def innerTail235_14 : List Nat := 7229 :: innerTail235_15
def innerCandidates235_14 : List Nat := [7541]
set_option maxRecDepth 4096 in
theorem innerCandidates235_14_checked : candidates edge 4 1493 7229 innerTail235_15 best.bound = innerCandidates235_14 := by
  unfold candidates
  have row : edge 7229 = SearchData.row7229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan235_14 : scan edge (search edge 3) 4 [1493] 1493 innerTail235_14 best = best := by
  rw [innerTail235_14, scan, if_neg (by decide), innerCandidates235_14_checked]
  change scan edge (search edge 3) 4 [1493] 1493 innerTail235_15
    (if 3 ≤ innerCandidates235_14.length then search edge 3 [1493, 7229] innerCandidates235_14 8722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan235_15

def innerTail235_13 : List Nat := 6599 :: innerTail235_14
def innerCandidates235_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates235_13_checked : candidates edge 4 1493 6599 innerTail235_14 best.bound = innerCandidates235_13 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan235_13 : scan edge (search edge 3) 4 [1493] 1493 innerTail235_13 best = best := by
  rw [innerTail235_13, scan, if_neg (by decide), innerCandidates235_13_checked]
  change scan edge (search edge 3) 4 [1493] 1493 innerTail235_14
    (if 3 ≤ innerCandidates235_13.length then search edge 3 [1493, 6599] innerCandidates235_13 8092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan235_14

def innerTail235_12 : List Nat := 5867 :: innerTail235_13
def innerCandidates235_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates235_12_checked : candidates edge 4 1493 5867 innerTail235_13 best.bound = innerCandidates235_12 := by
  unfold candidates
  have row : edge 5867 = SearchData.row5867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan235_12 : scan edge (search edge 3) 4 [1493] 1493 innerTail235_12 best = best := by
  rw [innerTail235_12, scan, if_neg (by decide), innerCandidates235_12_checked]
  change scan edge (search edge 3) 4 [1493] 1493 innerTail235_13
    (if 3 ≤ innerCandidates235_12.length then search edge 3 [1493, 5867] innerCandidates235_12 7360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan235_13

def innerTail235_11 : List Nat := 5651 :: innerTail235_12
def innerCandidates235_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates235_11_checked : candidates edge 4 1493 5651 innerTail235_12 best.bound = innerCandidates235_11 := by
  unfold candidates
  have row : edge 5651 = SearchData.row5651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan235_11 : scan edge (search edge 3) 4 [1493] 1493 innerTail235_11 best = best := by
  rw [innerTail235_11, scan, if_neg (by decide), innerCandidates235_11_checked]
  change scan edge (search edge 3) 4 [1493] 1493 innerTail235_12
    (if 3 ≤ innerCandidates235_11.length then search edge 3 [1493, 5651] innerCandidates235_11 7144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan235_12

def innerTail235_10 : List Nat := 5279 :: innerTail235_11
def innerCandidates235_10 : List Nat := [5651]
set_option maxRecDepth 4096 in
theorem innerCandidates235_10_checked : candidates edge 4 1493 5279 innerTail235_11 best.bound = innerCandidates235_10 := by
  unfold candidates
  have row : edge 5279 = SearchData.row5279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan235_10 : scan edge (search edge 3) 4 [1493] 1493 innerTail235_10 best = best := by
  rw [innerTail235_10, scan, if_neg (by decide), innerCandidates235_10_checked]
  change scan edge (search edge 3) 4 [1493] 1493 innerTail235_11
    (if 3 ≤ innerCandidates235_10.length then search edge 3 [1493, 5279] innerCandidates235_10 6772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan235_11

def innerTail235_9 : List Nat := 5189 :: innerTail235_10
def innerCandidates235_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates235_9_checked : candidates edge 4 1493 5189 innerTail235_10 best.bound = innerCandidates235_9 := by
  unfold candidates
  have row : edge 5189 = SearchData.row5189.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan235_9 : scan edge (search edge 3) 4 [1493] 1493 innerTail235_9 best = best := by
  rw [innerTail235_9, scan, if_neg (by decide), innerCandidates235_9_checked]
  change scan edge (search edge 3) 4 [1493] 1493 innerTail235_10
    (if 3 ≤ innerCandidates235_9.length then search edge 3 [1493, 5189] innerCandidates235_9 6682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan235_10

def innerTail235_8 : List Nat := 5021 :: innerTail235_9
def innerCandidates235_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates235_8_checked : candidates edge 4 1493 5021 innerTail235_9 best.bound = innerCandidates235_8 := by
  unfold candidates
  have row : edge 5021 = SearchData.row5021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan235_8 : scan edge (search edge 3) 4 [1493] 1493 innerTail235_8 best = best := by
  rw [innerTail235_8, scan, if_neg (by decide), innerCandidates235_8_checked]
  change scan edge (search edge 3) 4 [1493] 1493 innerTail235_9
    (if 3 ≤ innerCandidates235_8.length then search edge 3 [1493, 5021] innerCandidates235_8 6514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan235_9

def innerTail235_7 : List Nat := 4943 :: innerTail235_8
def innerCandidates235_7 : List Nat := [5021, 5189]
set_option maxRecDepth 4096 in
theorem innerCandidates235_7_checked : candidates edge 4 1493 4943 innerTail235_8 best.bound = innerCandidates235_7 := by
  unfold candidates
  have row : edge 4943 = SearchData.row4943.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan235_7 : scan edge (search edge 3) 4 [1493] 1493 innerTail235_7 best = best := by
  rw [innerTail235_7, scan, if_neg (by decide), innerCandidates235_7_checked]
  change scan edge (search edge 3) 4 [1493] 1493 innerTail235_8
    (if 3 ≤ innerCandidates235_7.length then search edge 3 [1493, 4943] innerCandidates235_7 6436 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan235_8

def innerTail235_6 : List Nat := 4649 :: innerTail235_7
def innerCandidates235_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates235_6_checked : candidates edge 4 1493 4649 innerTail235_7 best.bound = innerCandidates235_6 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan235_6 : scan edge (search edge 3) 4 [1493] 1493 innerTail235_6 best = best := by
  rw [innerTail235_6, scan, if_neg (by decide), innerCandidates235_6_checked]
  change scan edge (search edge 3) 4 [1493] 1493 innerTail235_7
    (if 3 ≤ innerCandidates235_6.length then search edge 3 [1493, 4649] innerCandidates235_6 6142 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan235_7

def innerTail235_5 : List Nat := 3623 :: innerTail235_6
def innerCandidates235_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates235_5_checked : candidates edge 4 1493 3623 innerTail235_6 best.bound = innerCandidates235_5 := by
  unfold candidates
  have row : edge 3623 = SearchData.row3623.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan235_5 : scan edge (search edge 3) 4 [1493] 1493 innerTail235_5 best = best := by
  rw [innerTail235_5, scan, if_neg (by decide), innerCandidates235_5_checked]
  change scan edge (search edge 3) 4 [1493] 1493 innerTail235_6
    (if 3 ≤ innerCandidates235_5.length then search edge 3 [1493, 3623] innerCandidates235_5 5116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan235_6

def innerTail235_4 : List Nat := 3329 :: innerTail235_5
def innerCandidates235_4 : List Nat := [4649]
set_option maxRecDepth 4096 in
theorem innerCandidates235_4_checked : candidates edge 4 1493 3329 innerTail235_5 best.bound = innerCandidates235_4 := by
  unfold candidates
  have row : edge 3329 = SearchData.row3329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan235_4 : scan edge (search edge 3) 4 [1493] 1493 innerTail235_4 best = best := by
  rw [innerTail235_4, scan, if_neg (by decide), innerCandidates235_4_checked]
  change scan edge (search edge 3) 4 [1493] 1493 innerTail235_5
    (if 3 ≤ innerCandidates235_4.length then search edge 3 [1493, 3329] innerCandidates235_4 4822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan235_5

def innerTail235_3 : List Nat := 2609 :: innerTail235_4
def innerCandidates235_3 : List Nat := [5651, 6599]
set_option maxRecDepth 4096 in
theorem innerCandidates235_3_checked : candidates edge 4 1493 2609 innerTail235_4 best.bound = innerCandidates235_3 := by
  unfold candidates
  have row : edge 2609 = PrimePairCache.smallRow2609.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan235_3 : scan edge (search edge 3) 4 [1493] 1493 innerTail235_3 best = best := by
  rw [innerTail235_3, scan, if_neg (by decide), innerCandidates235_3_checked]
  change scan edge (search edge 3) 4 [1493] 1493 innerTail235_4
    (if 3 ≤ innerCandidates235_3.length then search edge 3 [1493, 2609] innerCandidates235_3 4102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan235_4

def innerTail235_2 : List Nat := 2441 :: innerTail235_3
def innerCandidates235_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates235_2_checked : candidates edge 4 1493 2441 innerTail235_3 best.bound = innerCandidates235_2 := by
  unfold candidates
  have row : edge 2441 = PrimePairCache.smallRow2441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan235_2 : scan edge (search edge 3) 4 [1493] 1493 innerTail235_2 best = best := by
  rw [innerTail235_2, scan, if_neg (by decide), innerCandidates235_2_checked]
  change scan edge (search edge 3) 4 [1493] 1493 innerTail235_3
    (if 3 ≤ innerCandidates235_2.length then search edge 3 [1493, 2441] innerCandidates235_2 3934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan235_3

def innerTail235_1 : List Nat := 2339 :: innerTail235_2
def innerCandidates235_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates235_1_checked : candidates edge 4 1493 2339 innerTail235_2 best.bound = innerCandidates235_1 := by
  unfold candidates
  have row : edge 2339 = PrimePairCache.smallRow2339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan235_1 : scan edge (search edge 3) 4 [1493] 1493 innerTail235_1 best = best := by
  rw [innerTail235_1, scan, if_neg (by decide), innerCandidates235_1_checked]
  change scan edge (search edge 3) 4 [1493] 1493 innerTail235_2
    (if 3 ≤ innerCandidates235_1.length then search edge 3 [1493, 2339] innerCandidates235_1 3832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan235_2

def innerTail235_0 : List Nat := 1523 :: innerTail235_1
def innerCandidates235_0 : List Nat := [5279]
set_option maxRecDepth 4096 in
theorem innerCandidates235_0_checked : candidates edge 4 1493 1523 innerTail235_1 best.bound = innerCandidates235_0 := by
  unfold candidates
  have row : edge 1523 = PrimePairCache.smallRow1523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan235_0 : scan edge (search edge 3) 4 [1493] 1493 innerTail235_0 best = best := by
  rw [innerTail235_0, scan, if_neg (by decide), innerCandidates235_0_checked]
  change scan edge (search edge 3) 4 [1493] 1493 innerTail235_1
    (if 3 ≤ innerCandidates235_0.length then search edge 3 [1493, 1523] innerCandidates235_0 3016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan235_1

theorem rootBranch235 : search edge 4 [1493] rootCandidates235 1493 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1493] 1493 innerTail235_0 best = best
  exact innerScan235_0

def rootCandidates234 : List Nat := [3691, 4597, 4801, 5119, 5557, 5857, 6547, 6883, 8263, 8329]
set_option maxRecDepth 4096 in
theorem rootCandidates234_checked : candidates edge 5 0 1489 tail235 best.bound = rootCandidates234 := by
  unfold candidates
  have row : edge 1489 = PrimePairCache.smallRow1489.contains := by funext q; rfl
  rw [row]
  decide

def innerTail234_10 : List Nat := []
theorem innerScan234_10 : scan edge (search edge 3) 4 [1489] 1489 innerTail234_10 best = best := rfl

def innerTail234_9 : List Nat := 8329 :: innerTail234_10
def innerCandidates234_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates234_9_checked : candidates edge 4 1489 8329 innerTail234_10 best.bound = innerCandidates234_9 := by
  unfold candidates
  have row : edge 8329 = SearchData.row8329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan234_9 : scan edge (search edge 3) 4 [1489] 1489 innerTail234_9 best = best := by
  rw [innerTail234_9, scan, if_neg (by decide), innerCandidates234_9_checked]
  change scan edge (search edge 3) 4 [1489] 1489 innerTail234_10
    (if 3 ≤ innerCandidates234_9.length then search edge 3 [1489, 8329] innerCandidates234_9 9818 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan234_10

def innerTail234_8 : List Nat := 8263 :: innerTail234_9
def innerCandidates234_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates234_8_checked : candidates edge 4 1489 8263 innerTail234_9 best.bound = innerCandidates234_8 := by
  unfold candidates
  have row : edge 8263 = SearchData.row8263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan234_8 : scan edge (search edge 3) 4 [1489] 1489 innerTail234_8 best = best := by
  rw [innerTail234_8, scan, if_neg (by decide), innerCandidates234_8_checked]
  change scan edge (search edge 3) 4 [1489] 1489 innerTail234_9
    (if 3 ≤ innerCandidates234_8.length then search edge 3 [1489, 8263] innerCandidates234_8 9752 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan234_9

def innerTail234_7 : List Nat := 6883 :: innerTail234_8
def innerCandidates234_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates234_7_checked : candidates edge 4 1489 6883 innerTail234_8 best.bound = innerCandidates234_7 := by
  unfold candidates
  have row : edge 6883 = SearchData.row6883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan234_7 : scan edge (search edge 3) 4 [1489] 1489 innerTail234_7 best = best := by
  rw [innerTail234_7, scan, if_neg (by decide), innerCandidates234_7_checked]
  change scan edge (search edge 3) 4 [1489] 1489 innerTail234_8
    (if 3 ≤ innerCandidates234_7.length then search edge 3 [1489, 6883] innerCandidates234_7 8372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan234_8

def innerTail234_6 : List Nat := 6547 :: innerTail234_7
def innerCandidates234_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates234_6_checked : candidates edge 4 1489 6547 innerTail234_7 best.bound = innerCandidates234_6 := by
  unfold candidates
  have row : edge 6547 = SearchData.row6547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan234_6 : scan edge (search edge 3) 4 [1489] 1489 innerTail234_6 best = best := by
  rw [innerTail234_6, scan, if_neg (by decide), innerCandidates234_6_checked]
  change scan edge (search edge 3) 4 [1489] 1489 innerTail234_7
    (if 3 ≤ innerCandidates234_6.length then search edge 3 [1489, 6547] innerCandidates234_6 8036 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan234_7

def innerTail234_5 : List Nat := 5857 :: innerTail234_6
def innerCandidates234_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates234_5_checked : candidates edge 4 1489 5857 innerTail234_6 best.bound = innerCandidates234_5 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan234_5 : scan edge (search edge 3) 4 [1489] 1489 innerTail234_5 best = best := by
  rw [innerTail234_5, scan, if_neg (by decide), innerCandidates234_5_checked]
  change scan edge (search edge 3) 4 [1489] 1489 innerTail234_6
    (if 3 ≤ innerCandidates234_5.length then search edge 3 [1489, 5857] innerCandidates234_5 7346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan234_6

def innerTail234_4 : List Nat := 5557 :: innerTail234_5
def innerCandidates234_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates234_4_checked : candidates edge 4 1489 5557 innerTail234_5 best.bound = innerCandidates234_4 := by
  unfold candidates
  have row : edge 5557 = SearchData.row5557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan234_4 : scan edge (search edge 3) 4 [1489] 1489 innerTail234_4 best = best := by
  rw [innerTail234_4, scan, if_neg (by decide), innerCandidates234_4_checked]
  change scan edge (search edge 3) 4 [1489] 1489 innerTail234_5
    (if 3 ≤ innerCandidates234_4.length then search edge 3 [1489, 5557] innerCandidates234_4 7046 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan234_5

def innerTail234_3 : List Nat := 5119 :: innerTail234_4
def innerCandidates234_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates234_3_checked : candidates edge 4 1489 5119 innerTail234_4 best.bound = innerCandidates234_3 := by
  unfold candidates
  have row : edge 5119 = SearchData.row5119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan234_3 : scan edge (search edge 3) 4 [1489] 1489 innerTail234_3 best = best := by
  rw [innerTail234_3, scan, if_neg (by decide), innerCandidates234_3_checked]
  change scan edge (search edge 3) 4 [1489] 1489 innerTail234_4
    (if 3 ≤ innerCandidates234_3.length then search edge 3 [1489, 5119] innerCandidates234_3 6608 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan234_4

def innerTail234_2 : List Nat := 4801 :: innerTail234_3
def innerCandidates234_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates234_2_checked : candidates edge 4 1489 4801 innerTail234_3 best.bound = innerCandidates234_2 := by
  unfold candidates
  have row : edge 4801 = SearchData.row4801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan234_2 : scan edge (search edge 3) 4 [1489] 1489 innerTail234_2 best = best := by
  rw [innerTail234_2, scan, if_neg (by decide), innerCandidates234_2_checked]
  change scan edge (search edge 3) 4 [1489] 1489 innerTail234_3
    (if 3 ≤ innerCandidates234_2.length then search edge 3 [1489, 4801] innerCandidates234_2 6290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan234_3

def innerTail234_1 : List Nat := 4597 :: innerTail234_2
def innerCandidates234_1 : List Nat := [8263]
set_option maxRecDepth 4096 in
theorem innerCandidates234_1_checked : candidates edge 4 1489 4597 innerTail234_2 best.bound = innerCandidates234_1 := by
  unfold candidates
  have row : edge 4597 = SearchData.row4597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan234_1 : scan edge (search edge 3) 4 [1489] 1489 innerTail234_1 best = best := by
  rw [innerTail234_1, scan, if_neg (by decide), innerCandidates234_1_checked]
  change scan edge (search edge 3) 4 [1489] 1489 innerTail234_2
    (if 3 ≤ innerCandidates234_1.length then search edge 3 [1489, 4597] innerCandidates234_1 6086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan234_2

def innerTail234_0 : List Nat := 3691 :: innerTail234_1
def innerCandidates234_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates234_0_checked : candidates edge 4 1489 3691 innerTail234_1 best.bound = innerCandidates234_0 := by
  unfold candidates
  have row : edge 3691 = SearchData.row3691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan234_0 : scan edge (search edge 3) 4 [1489] 1489 innerTail234_0 best = best := by
  rw [innerTail234_0, scan, if_neg (by decide), innerCandidates234_0_checked]
  change scan edge (search edge 3) 4 [1489] 1489 innerTail234_1
    (if 3 ≤ innerCandidates234_0.length then search edge 3 [1489, 3691] innerCandidates234_0 5180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan234_1

theorem rootBranch234 : search edge 4 [1489] rootCandidates234 1489 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1489] 1489 innerTail234_0 best = best
  exact innerScan234_0

def rootCandidates233 : List Nat := [1553, 1601, 1907, 1979, 2267, 2729, 3659, 3671, 5717, 5801, 5843, 6473, 6761, 7013, 7727]
set_option maxRecDepth 4096 in
theorem rootCandidates233_checked : candidates edge 5 0 1487 tail234 best.bound = rootCandidates233 := by
  unfold candidates
  have row : edge 1487 = PrimePairCache.smallRow1487.contains := by funext q; rfl
  rw [row]
  decide

def innerTail233_15 : List Nat := []
theorem innerScan233_15 : scan edge (search edge 3) 4 [1487] 1487 innerTail233_15 best = best := rfl

def innerTail233_14 : List Nat := 7727 :: innerTail233_15
def innerCandidates233_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates233_14_checked : candidates edge 4 1487 7727 innerTail233_15 best.bound = innerCandidates233_14 := by
  unfold candidates
  have row : edge 7727 = SearchData.row7727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan233_14 : scan edge (search edge 3) 4 [1487] 1487 innerTail233_14 best = best := by
  rw [innerTail233_14, scan, if_neg (by decide), innerCandidates233_14_checked]
  change scan edge (search edge 3) 4 [1487] 1487 innerTail233_15
    (if 3 ≤ innerCandidates233_14.length then search edge 3 [1487, 7727] innerCandidates233_14 9214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan233_15

def innerTail233_13 : List Nat := 7013 :: innerTail233_14
def innerCandidates233_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates233_13_checked : candidates edge 4 1487 7013 innerTail233_14 best.bound = innerCandidates233_13 := by
  unfold candidates
  have row : edge 7013 = SearchData.row7013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan233_13 : scan edge (search edge 3) 4 [1487] 1487 innerTail233_13 best = best := by
  rw [innerTail233_13, scan, if_neg (by decide), innerCandidates233_13_checked]
  change scan edge (search edge 3) 4 [1487] 1487 innerTail233_14
    (if 3 ≤ innerCandidates233_13.length then search edge 3 [1487, 7013] innerCandidates233_13 8500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan233_14

def innerTail233_12 : List Nat := 6761 :: innerTail233_13
def innerCandidates233_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates233_12_checked : candidates edge 4 1487 6761 innerTail233_13 best.bound = innerCandidates233_12 := by
  unfold candidates
  have row : edge 6761 = SearchData.row6761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan233_12 : scan edge (search edge 3) 4 [1487] 1487 innerTail233_12 best = best := by
  rw [innerTail233_12, scan, if_neg (by decide), innerCandidates233_12_checked]
  change scan edge (search edge 3) 4 [1487] 1487 innerTail233_13
    (if 3 ≤ innerCandidates233_12.length then search edge 3 [1487, 6761] innerCandidates233_12 8248 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan233_13

def innerTail233_11 : List Nat := 6473 :: innerTail233_12
def innerCandidates233_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates233_11_checked : candidates edge 4 1487 6473 innerTail233_12 best.bound = innerCandidates233_11 := by
  unfold candidates
  have row : edge 6473 = SearchData.row6473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan233_11 : scan edge (search edge 3) 4 [1487] 1487 innerTail233_11 best = best := by
  rw [innerTail233_11, scan, if_neg (by decide), innerCandidates233_11_checked]
  change scan edge (search edge 3) 4 [1487] 1487 innerTail233_12
    (if 3 ≤ innerCandidates233_11.length then search edge 3 [1487, 6473] innerCandidates233_11 7960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan233_12

def innerTail233_10 : List Nat := 5843 :: innerTail233_11
def innerCandidates233_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates233_10_checked : candidates edge 4 1487 5843 innerTail233_11 best.bound = innerCandidates233_10 := by
  unfold candidates
  have row : edge 5843 = SearchData.row5843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan233_10 : scan edge (search edge 3) 4 [1487] 1487 innerTail233_10 best = best := by
  rw [innerTail233_10, scan, if_neg (by decide), innerCandidates233_10_checked]
  change scan edge (search edge 3) 4 [1487] 1487 innerTail233_11
    (if 3 ≤ innerCandidates233_10.length then search edge 3 [1487, 5843] innerCandidates233_10 7330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan233_11

def innerTail233_9 : List Nat := 5801 :: innerTail233_10
def innerCandidates233_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates233_9_checked : candidates edge 4 1487 5801 innerTail233_10 best.bound = innerCandidates233_9 := by
  unfold candidates
  have row : edge 5801 = SearchData.row5801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan233_9 : scan edge (search edge 3) 4 [1487] 1487 innerTail233_9 best = best := by
  rw [innerTail233_9, scan, if_neg (by decide), innerCandidates233_9_checked]
  change scan edge (search edge 3) 4 [1487] 1487 innerTail233_10
    (if 3 ≤ innerCandidates233_9.length then search edge 3 [1487, 5801] innerCandidates233_9 7288 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan233_10

def innerTail233_8 : List Nat := 5717 :: innerTail233_9
def innerCandidates233_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates233_8_checked : candidates edge 4 1487 5717 innerTail233_9 best.bound = innerCandidates233_8 := by
  unfold candidates
  have row : edge 5717 = SearchData.row5717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan233_8 : scan edge (search edge 3) 4 [1487] 1487 innerTail233_8 best = best := by
  rw [innerTail233_8, scan, if_neg (by decide), innerCandidates233_8_checked]
  change scan edge (search edge 3) 4 [1487] 1487 innerTail233_9
    (if 3 ≤ innerCandidates233_8.length then search edge 3 [1487, 5717] innerCandidates233_8 7204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan233_9

def innerTail233_7 : List Nat := 3671 :: innerTail233_8
def innerCandidates233_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates233_7_checked : candidates edge 4 1487 3671 innerTail233_8 best.bound = innerCandidates233_7 := by
  unfold candidates
  have row : edge 3671 = SearchData.row3671.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan233_7 : scan edge (search edge 3) 4 [1487] 1487 innerTail233_7 best = best := by
  rw [innerTail233_7, scan, if_neg (by decide), innerCandidates233_7_checked]
  change scan edge (search edge 3) 4 [1487] 1487 innerTail233_8
    (if 3 ≤ innerCandidates233_7.length then search edge 3 [1487, 3671] innerCandidates233_7 5158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan233_8

def innerTail233_6 : List Nat := 3659 :: innerTail233_7
def innerCandidates233_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates233_6_checked : candidates edge 4 1487 3659 innerTail233_7 best.bound = innerCandidates233_6 := by
  unfold candidates
  have row : edge 3659 = SearchData.row3659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan233_6 : scan edge (search edge 3) 4 [1487] 1487 innerTail233_6 best = best := by
  rw [innerTail233_6, scan, if_neg (by decide), innerCandidates233_6_checked]
  change scan edge (search edge 3) 4 [1487] 1487 innerTail233_7
    (if 3 ≤ innerCandidates233_6.length then search edge 3 [1487, 3659] innerCandidates233_6 5146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan233_7

def innerTail233_5 : List Nat := 2729 :: innerTail233_6
def innerCandidates233_5 : List Nat := [6473]
set_option maxRecDepth 4096 in
theorem innerCandidates233_5_checked : candidates edge 4 1487 2729 innerTail233_6 best.bound = innerCandidates233_5 := by
  unfold candidates
  have row : edge 2729 = SearchData.row2729.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan233_5 : scan edge (search edge 3) 4 [1487] 1487 innerTail233_5 best = best := by
  rw [innerTail233_5, scan, if_neg (by decide), innerCandidates233_5_checked]
  change scan edge (search edge 3) 4 [1487] 1487 innerTail233_6
    (if 3 ≤ innerCandidates233_5.length then search edge 3 [1487, 2729] innerCandidates233_5 4216 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan233_6

def innerTail233_4 : List Nat := 2267 :: innerTail233_5
def innerCandidates233_4 : List Nat := [2729, 5843]
set_option maxRecDepth 4096 in
theorem innerCandidates233_4_checked : candidates edge 4 1487 2267 innerTail233_5 best.bound = innerCandidates233_4 := by
  unfold candidates
  have row : edge 2267 = PrimePairCache.smallRow2267.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan233_4 : scan edge (search edge 3) 4 [1487] 1487 innerTail233_4 best = best := by
  rw [innerTail233_4, scan, if_neg (by decide), innerCandidates233_4_checked]
  change scan edge (search edge 3) 4 [1487] 1487 innerTail233_5
    (if 3 ≤ innerCandidates233_4.length then search edge 3 [1487, 2267] innerCandidates233_4 3754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan233_5

def innerTail233_3 : List Nat := 1979 :: innerTail233_4
def innerCandidates233_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates233_3_checked : candidates edge 4 1487 1979 innerTail233_4 best.bound = innerCandidates233_3 := by
  unfold candidates
  have row : edge 1979 = PrimePairCache.smallRow1979.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan233_3 : scan edge (search edge 3) 4 [1487] 1487 innerTail233_3 best = best := by
  rw [innerTail233_3, scan, if_neg (by decide), innerCandidates233_3_checked]
  change scan edge (search edge 3) 4 [1487] 1487 innerTail233_4
    (if 3 ≤ innerCandidates233_3.length then search edge 3 [1487, 1979] innerCandidates233_3 3466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan233_4

def innerTail233_2 : List Nat := 1907 :: innerTail233_3
def innerCandidates233_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates233_2_checked : candidates edge 4 1487 1907 innerTail233_3 best.bound = innerCandidates233_2 := by
  unfold candidates
  have row : edge 1907 = PrimePairCache.smallRow1907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan233_2 : scan edge (search edge 3) 4 [1487] 1487 innerTail233_2 best = best := by
  rw [innerTail233_2, scan, if_neg (by decide), innerCandidates233_2_checked]
  change scan edge (search edge 3) 4 [1487] 1487 innerTail233_3
    (if 3 ≤ innerCandidates233_2.length then search edge 3 [1487, 1907] innerCandidates233_2 3394 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan233_3

def innerTail233_1 : List Nat := 1601 :: innerTail233_2
def innerCandidates233_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates233_1_checked : candidates edge 4 1487 1601 innerTail233_2 best.bound = innerCandidates233_1 := by
  unfold candidates
  have row : edge 1601 = PrimePairCache.smallRow1601.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan233_1 : scan edge (search edge 3) 4 [1487] 1487 innerTail233_1 best = best := by
  rw [innerTail233_1, scan, if_neg (by decide), innerCandidates233_1_checked]
  change scan edge (search edge 3) 4 [1487] 1487 innerTail233_2
    (if 3 ≤ innerCandidates233_1.length then search edge 3 [1487, 1601] innerCandidates233_1 3088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan233_2

def innerTail233_0 : List Nat := 1553 :: innerTail233_1
def innerCandidates233_0 : List Nat := [7013]
set_option maxRecDepth 4096 in
theorem innerCandidates233_0_checked : candidates edge 4 1487 1553 innerTail233_1 best.bound = innerCandidates233_0 := by
  unfold candidates
  have row : edge 1553 = PrimePairCache.smallRow1553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan233_0 : scan edge (search edge 3) 4 [1487] 1487 innerTail233_0 best = best := by
  rw [innerTail233_0, scan, if_neg (by decide), innerCandidates233_0_checked]
  change scan edge (search edge 3) 4 [1487] 1487 innerTail233_1
    (if 3 ≤ innerCandidates233_0.length then search edge 3 [1487, 1553] innerCandidates233_0 3040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan233_1

theorem rootBranch233 : search edge 4 [1487] rootCandidates233 1487 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1487] 1487 innerTail233_0 best = best
  exact innerScan233_0

end ProjectEulerSolutions.PrimePairUpper.Search
