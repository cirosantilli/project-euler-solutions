import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates82 : List Nat := [541, 601, 613, 661, 787, 853, 1399, 1483, 1549, 1579, 2383, 2503, 2593, 2797, 2887, 3511, 4021, 4441, 4447, 4783, 5527, 5557, 5701, 5827, 6481, 8011]
set_option maxRecDepth 4096 in
theorem rootCandidates82_checked : candidates edge 5 0 439 tail83 best.bound = rootCandidates82 := by
  unfold candidates
  have row : edge 439 = PrimePairCache.smallRow439.contains := by funext q; rfl
  rw [row]
  decide

def innerTail82_26 : List Nat := []
theorem innerScan82_26 : scan edge (search edge 3) 4 [439] 439 innerTail82_26 best = best := rfl

def innerTail82_25 : List Nat := 8011 :: innerTail82_26
def innerCandidates82_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates82_25_checked : candidates edge 4 439 8011 innerTail82_26 best.bound = innerCandidates82_25 := by
  unfold candidates
  have row : edge 8011 = SearchData.row8011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_25 : scan edge (search edge 3) 4 [439] 439 innerTail82_25 best = best := by
  rw [innerTail82_25, scan, if_neg (by decide), innerCandidates82_25_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_26
    (if 3 ≤ innerCandidates82_25.length then search edge 3 [439, 8011] innerCandidates82_25 8450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_26

def innerTail82_24 : List Nat := 6481 :: innerTail82_25
def innerCandidates82_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates82_24_checked : candidates edge 4 439 6481 innerTail82_25 best.bound = innerCandidates82_24 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_24 : scan edge (search edge 3) 4 [439] 439 innerTail82_24 best = best := by
  rw [innerTail82_24, scan, if_neg (by decide), innerCandidates82_24_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_25
    (if 3 ≤ innerCandidates82_24.length then search edge 3 [439, 6481] innerCandidates82_24 6920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_25

def innerTail82_23 : List Nat := 5827 :: innerTail82_24
def innerCandidates82_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates82_23_checked : candidates edge 4 439 5827 innerTail82_24 best.bound = innerCandidates82_23 := by
  unfold candidates
  have row : edge 5827 = SearchData.row5827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_23 : scan edge (search edge 3) 4 [439] 439 innerTail82_23 best = best := by
  rw [innerTail82_23, scan, if_neg (by decide), innerCandidates82_23_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_24
    (if 3 ≤ innerCandidates82_23.length then search edge 3 [439, 5827] innerCandidates82_23 6266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_24

def innerTail82_22 : List Nat := 5701 :: innerTail82_23
def innerCandidates82_22 : List Nat := [6481]
set_option maxRecDepth 4096 in
theorem innerCandidates82_22_checked : candidates edge 4 439 5701 innerTail82_23 best.bound = innerCandidates82_22 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_22 : scan edge (search edge 3) 4 [439] 439 innerTail82_22 best = best := by
  rw [innerTail82_22, scan, if_neg (by decide), innerCandidates82_22_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_23
    (if 3 ≤ innerCandidates82_22.length then search edge 3 [439, 5701] innerCandidates82_22 6140 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_23

def innerTail82_21 : List Nat := 5557 :: innerTail82_22
def innerCandidates82_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates82_21_checked : candidates edge 4 439 5557 innerTail82_22 best.bound = innerCandidates82_21 := by
  unfold candidates
  have row : edge 5557 = SearchData.row5557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_21 : scan edge (search edge 3) 4 [439] 439 innerTail82_21 best = best := by
  rw [innerTail82_21, scan, if_neg (by decide), innerCandidates82_21_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_22
    (if 3 ≤ innerCandidates82_21.length then search edge 3 [439, 5557] innerCandidates82_21 5996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_22

def innerTail82_20 : List Nat := 5527 :: innerTail82_21
def innerCandidates82_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates82_20_checked : candidates edge 4 439 5527 innerTail82_21 best.bound = innerCandidates82_20 := by
  unfold candidates
  have row : edge 5527 = SearchData.row5527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_20 : scan edge (search edge 3) 4 [439] 439 innerTail82_20 best = best := by
  rw [innerTail82_20, scan, if_neg (by decide), innerCandidates82_20_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_21
    (if 3 ≤ innerCandidates82_20.length then search edge 3 [439, 5527] innerCandidates82_20 5966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_21

def innerTail82_19 : List Nat := 4783 :: innerTail82_20
def innerCandidates82_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates82_19_checked : candidates edge 4 439 4783 innerTail82_20 best.bound = innerCandidates82_19 := by
  unfold candidates
  have row : edge 4783 = SearchData.row4783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_19 : scan edge (search edge 3) 4 [439] 439 innerTail82_19 best = best := by
  rw [innerTail82_19, scan, if_neg (by decide), innerCandidates82_19_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_20
    (if 3 ≤ innerCandidates82_19.length then search edge 3 [439, 4783] innerCandidates82_19 5222 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_20

def innerTail82_18 : List Nat := 4447 :: innerTail82_19
def innerCandidates82_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates82_18_checked : candidates edge 4 439 4447 innerTail82_19 best.bound = innerCandidates82_18 := by
  unfold candidates
  have row : edge 4447 = SearchData.row4447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_18 : scan edge (search edge 3) 4 [439] 439 innerTail82_18 best = best := by
  rw [innerTail82_18, scan, if_neg (by decide), innerCandidates82_18_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_19
    (if 3 ≤ innerCandidates82_18.length then search edge 3 [439, 4447] innerCandidates82_18 4886 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_19

def innerTail82_17 : List Nat := 4441 :: innerTail82_18
def innerCandidates82_17 : List Nat := [6481]
set_option maxRecDepth 4096 in
theorem innerCandidates82_17_checked : candidates edge 4 439 4441 innerTail82_18 best.bound = innerCandidates82_17 := by
  unfold candidates
  have row : edge 4441 = SearchData.row4441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_17 : scan edge (search edge 3) 4 [439] 439 innerTail82_17 best = best := by
  rw [innerTail82_17, scan, if_neg (by decide), innerCandidates82_17_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_18
    (if 3 ≤ innerCandidates82_17.length then search edge 3 [439, 4441] innerCandidates82_17 4880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_18

def innerTail82_16 : List Nat := 4021 :: innerTail82_17
def innerCandidates82_16 : List Nat := [4441]
set_option maxRecDepth 4096 in
theorem innerCandidates82_16_checked : candidates edge 4 439 4021 innerTail82_17 best.bound = innerCandidates82_16 := by
  unfold candidates
  have row : edge 4021 = SearchData.row4021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_16 : scan edge (search edge 3) 4 [439] 439 innerTail82_16 best = best := by
  rw [innerTail82_16, scan, if_neg (by decide), innerCandidates82_16_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_17
    (if 3 ≤ innerCandidates82_16.length then search edge 3 [439, 4021] innerCandidates82_16 4460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_17

def innerTail82_15 : List Nat := 3511 :: innerTail82_16
def innerCandidates82_15 : List Nat := [8011]
set_option maxRecDepth 4096 in
theorem innerCandidates82_15_checked : candidates edge 4 439 3511 innerTail82_16 best.bound = innerCandidates82_15 := by
  unfold candidates
  have row : edge 3511 = SearchData.row3511.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_15 : scan edge (search edge 3) 4 [439] 439 innerTail82_15 best = best := by
  rw [innerTail82_15, scan, if_neg (by decide), innerCandidates82_15_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_16
    (if 3 ≤ innerCandidates82_15.length then search edge 3 [439, 3511] innerCandidates82_15 3950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_16

def innerTail82_14 : List Nat := 2887 :: innerTail82_15
def innerCandidates82_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates82_14_checked : candidates edge 4 439 2887 innerTail82_15 best.bound = innerCandidates82_14 := by
  unfold candidates
  have row : edge 2887 = SearchData.row2887.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_14 : scan edge (search edge 3) 4 [439] 439 innerTail82_14 best = best := by
  rw [innerTail82_14, scan, if_neg (by decide), innerCandidates82_14_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_15
    (if 3 ≤ innerCandidates82_14.length then search edge 3 [439, 2887] innerCandidates82_14 3326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_15

def innerTail82_13 : List Nat := 2797 :: innerTail82_14
def innerCandidates82_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates82_13_checked : candidates edge 4 439 2797 innerTail82_14 best.bound = innerCandidates82_13 := by
  unfold candidates
  have row : edge 2797 = SearchData.row2797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_13 : scan edge (search edge 3) 4 [439] 439 innerTail82_13 best = best := by
  rw [innerTail82_13, scan, if_neg (by decide), innerCandidates82_13_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_14
    (if 3 ≤ innerCandidates82_13.length then search edge 3 [439, 2797] innerCandidates82_13 3236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_14

def innerTail82_12 : List Nat := 2593 :: innerTail82_13
def innerCandidates82_12 : List Nat := [2887]
set_option maxRecDepth 4096 in
theorem innerCandidates82_12_checked : candidates edge 4 439 2593 innerTail82_13 best.bound = innerCandidates82_12 := by
  unfold candidates
  have row : edge 2593 = PrimePairCache.smallRow2593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_12 : scan edge (search edge 3) 4 [439] 439 innerTail82_12 best = best := by
  rw [innerTail82_12, scan, if_neg (by decide), innerCandidates82_12_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_13
    (if 3 ≤ innerCandidates82_12.length then search edge 3 [439, 2593] innerCandidates82_12 3032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_13

def innerTail82_11 : List Nat := 2503 :: innerTail82_12
def innerCandidates82_11 : List Nat := [4447]
set_option maxRecDepth 4096 in
theorem innerCandidates82_11_checked : candidates edge 4 439 2503 innerTail82_12 best.bound = innerCandidates82_11 := by
  unfold candidates
  have row : edge 2503 = PrimePairCache.smallRow2503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_11 : scan edge (search edge 3) 4 [439] 439 innerTail82_11 best = best := by
  rw [innerTail82_11, scan, if_neg (by decide), innerCandidates82_11_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_12
    (if 3 ≤ innerCandidates82_11.length then search edge 3 [439, 2503] innerCandidates82_11 2942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_12

def innerTail82_10 : List Nat := 2383 :: innerTail82_11
def innerCandidates82_10 : List Nat := [4021, 5527]
set_option maxRecDepth 4096 in
theorem innerCandidates82_10_checked : candidates edge 4 439 2383 innerTail82_11 best.bound = innerCandidates82_10 := by
  unfold candidates
  have row : edge 2383 = PrimePairCache.smallRow2383.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_10 : scan edge (search edge 3) 4 [439] 439 innerTail82_10 best = best := by
  rw [innerTail82_10, scan, if_neg (by decide), innerCandidates82_10_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_11
    (if 3 ≤ innerCandidates82_10.length then search edge 3 [439, 2383] innerCandidates82_10 2822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_11

def innerTail82_9 : List Nat := 1579 :: innerTail82_10
def innerCandidates82_9 : List Nat := [4783]
set_option maxRecDepth 4096 in
theorem innerCandidates82_9_checked : candidates edge 4 439 1579 innerTail82_10 best.bound = innerCandidates82_9 := by
  unfold candidates
  have row : edge 1579 = PrimePairCache.smallRow1579.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_9 : scan edge (search edge 3) 4 [439] 439 innerTail82_9 best = best := by
  rw [innerTail82_9, scan, if_neg (by decide), innerCandidates82_9_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_10
    (if 3 ≤ innerCandidates82_9.length then search edge 3 [439, 1579] innerCandidates82_9 2018 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_10

def innerTail82_8 : List Nat := 1549 :: innerTail82_9
def innerCandidates82_8 : List Nat := [4783]
set_option maxRecDepth 4096 in
theorem innerCandidates82_8_checked : candidates edge 4 439 1549 innerTail82_9 best.bound = innerCandidates82_8 := by
  unfold candidates
  have row : edge 1549 = PrimePairCache.smallRow1549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_8 : scan edge (search edge 3) 4 [439] 439 innerTail82_8 best = best := by
  rw [innerTail82_8, scan, if_neg (by decide), innerCandidates82_8_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_9
    (if 3 ≤ innerCandidates82_8.length then search edge 3 [439, 1549] innerCandidates82_8 1988 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_9

def innerTail82_7 : List Nat := 1483 :: innerTail82_8
def innerCandidates82_7 : List Nat := [2593]
set_option maxRecDepth 4096 in
theorem innerCandidates82_7_checked : candidates edge 4 439 1483 innerTail82_8 best.bound = innerCandidates82_7 := by
  unfold candidates
  have row : edge 1483 = PrimePairCache.smallRow1483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_7 : scan edge (search edge 3) 4 [439] 439 innerTail82_7 best = best := by
  rw [innerTail82_7, scan, if_neg (by decide), innerCandidates82_7_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_8
    (if 3 ≤ innerCandidates82_7.length then search edge 3 [439, 1483] innerCandidates82_7 1922 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_8

def innerTail82_6 : List Nat := 1399 :: innerTail82_7
def innerCandidates82_6 : List Nat := [6481]
set_option maxRecDepth 4096 in
theorem innerCandidates82_6_checked : candidates edge 4 439 1399 innerTail82_7 best.bound = innerCandidates82_6 := by
  unfold candidates
  have row : edge 1399 = PrimePairCache.smallRow1399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_6 : scan edge (search edge 3) 4 [439] 439 innerTail82_6 best = best := by
  rw [innerTail82_6, scan, if_neg (by decide), innerCandidates82_6_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_7
    (if 3 ≤ innerCandidates82_6.length then search edge 3 [439, 1399] innerCandidates82_6 1838 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_7

def innerTail82_5 : List Nat := 853 :: innerTail82_6
def innerCandidates82_5 : List Nat := [2593, 4447]
set_option maxRecDepth 4096 in
theorem innerCandidates82_5_checked : candidates edge 4 439 853 innerTail82_6 best.bound = innerCandidates82_5 := by
  unfold candidates
  have row : edge 853 = PrimePairCache.smallRow853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_5 : scan edge (search edge 3) 4 [439] 439 innerTail82_5 best = best := by
  rw [innerTail82_5, scan, if_neg (by decide), innerCandidates82_5_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_6
    (if 3 ≤ innerCandidates82_5.length then search edge 3 [439, 853] innerCandidates82_5 1292 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_6

def innerTail82_4 : List Nat := 787 :: innerTail82_5
def innerCandidates82_4 : List Nat := [1483]
set_option maxRecDepth 4096 in
theorem innerCandidates82_4_checked : candidates edge 4 439 787 innerTail82_5 best.bound = innerCandidates82_4 := by
  unfold candidates
  have row : edge 787 = PrimePairCache.smallRow787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_4 : scan edge (search edge 3) 4 [439] 439 innerTail82_4 best = best := by
  rw [innerTail82_4, scan, if_neg (by decide), innerCandidates82_4_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_5
    (if 3 ≤ innerCandidates82_4.length then search edge 3 [439, 787] innerCandidates82_4 1226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_5

def innerTail82_3 : List Nat := 661 :: innerTail82_4
def innerCandidates82_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates82_3_checked : candidates edge 4 439 661 innerTail82_4 best.bound = innerCandidates82_3 := by
  unfold candidates
  have row : edge 661 = PrimePairCache.smallRow661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_3 : scan edge (search edge 3) 4 [439] 439 innerTail82_3 best = best := by
  rw [innerTail82_3, scan, if_neg (by decide), innerCandidates82_3_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_4
    (if 3 ≤ innerCandidates82_3.length then search edge 3 [439, 661] innerCandidates82_3 1100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_4

def innerTail82_2 : List Nat := 613 :: innerTail82_3
def innerCandidates82_2 : List Nat := [661, 4021]
set_option maxRecDepth 4096 in
theorem innerCandidates82_2_checked : candidates edge 4 439 613 innerTail82_3 best.bound = innerCandidates82_2 := by
  unfold candidates
  have row : edge 613 = PrimePairCache.smallRow613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_2 : scan edge (search edge 3) 4 [439] 439 innerTail82_2 best = best := by
  rw [innerTail82_2, scan, if_neg (by decide), innerCandidates82_2_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_3
    (if 3 ≤ innerCandidates82_2.length then search edge 3 [439, 613] innerCandidates82_2 1052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_3

def innerTail82_1 : List Nat := 601 :: innerTail82_2
def innerCandidates82_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates82_1_checked : candidates edge 4 439 601 innerTail82_2 best.bound = innerCandidates82_1 := by
  unfold candidates
  have row : edge 601 = PrimePairCache.smallRow601.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_1 : scan edge (search edge 3) 4 [439] 439 innerTail82_1 best = best := by
  rw [innerTail82_1, scan, if_neg (by decide), innerCandidates82_1_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_2
    (if 3 ≤ innerCandidates82_1.length then search edge 3 [439, 601] innerCandidates82_1 1040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_2

def innerTail82_0 : List Nat := 541 :: innerTail82_1
def innerCandidates82_0 : List Nat := [661, 2887]
set_option maxRecDepth 4096 in
theorem innerCandidates82_0_checked : candidates edge 4 439 541 innerTail82_1 best.bound = innerCandidates82_0 := by
  unfold candidates
  have row : edge 541 = PrimePairCache.smallRow541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan82_0 : scan edge (search edge 3) 4 [439] 439 innerTail82_0 best = best := by
  rw [innerTail82_0, scan, if_neg (by decide), innerCandidates82_0_checked]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_1
    (if 3 ≤ innerCandidates82_0.length then search edge 3 [439, 541] innerCandidates82_0 980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan82_1

theorem rootBranch82 : search edge 4 [439] rootCandidates82 439 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [439] 439 innerTail82_0 best = best
  exact innerScan82_0

def rootCandidates81 : List Nat := [571, 787, 859, 883, 1009, 1087, 1471, 1801, 2203, 2719, 2857, 3391, 3529, 3613, 3697, 3739, 4441, 4483, 4663, 4903, 5281, 5407, 6091, 6247, 6379, 6781, 7393, 7717, 8167]
set_option maxRecDepth 4096 in
theorem rootCandidates81_checked : candidates edge 5 0 433 tail82 best.bound = rootCandidates81 := by
  unfold candidates
  have row : edge 433 = PrimePairCache.smallRow433.contains := by funext q; rfl
  rw [row]
  decide

def innerTail81_29 : List Nat := []
theorem innerScan81_29 : scan edge (search edge 3) 4 [433] 433 innerTail81_29 best = best := rfl

def innerTail81_28 : List Nat := 8167 :: innerTail81_29
def innerCandidates81_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates81_28_checked : candidates edge 4 433 8167 innerTail81_29 best.bound = innerCandidates81_28 := by
  unfold candidates
  have row : edge 8167 = SearchData.row8167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_28 : scan edge (search edge 3) 4 [433] 433 innerTail81_28 best = best := by
  rw [innerTail81_28, scan, if_neg (by decide), innerCandidates81_28_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_29
    (if 3 ≤ innerCandidates81_28.length then search edge 3 [433, 8167] innerCandidates81_28 8600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_29

def innerTail81_27 : List Nat := 7717 :: innerTail81_28
def innerCandidates81_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates81_27_checked : candidates edge 4 433 7717 innerTail81_28 best.bound = innerCandidates81_27 := by
  unfold candidates
  have row : edge 7717 = SearchData.row7717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_27 : scan edge (search edge 3) 4 [433] 433 innerTail81_27 best = best := by
  rw [innerTail81_27, scan, if_neg (by decide), innerCandidates81_27_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_28
    (if 3 ≤ innerCandidates81_27.length then search edge 3 [433, 7717] innerCandidates81_27 8150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_28

def innerTail81_26 : List Nat := 7393 :: innerTail81_27
def innerCandidates81_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates81_26_checked : candidates edge 4 433 7393 innerTail81_27 best.bound = innerCandidates81_26 := by
  unfold candidates
  have row : edge 7393 = SearchData.row7393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_26 : scan edge (search edge 3) 4 [433] 433 innerTail81_26 best = best := by
  rw [innerTail81_26, scan, if_neg (by decide), innerCandidates81_26_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_27
    (if 3 ≤ innerCandidates81_26.length then search edge 3 [433, 7393] innerCandidates81_26 7826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_27

def innerTail81_25 : List Nat := 6781 :: innerTail81_26
def innerCandidates81_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates81_25_checked : candidates edge 4 433 6781 innerTail81_26 best.bound = innerCandidates81_25 := by
  unfold candidates
  have row : edge 6781 = SearchData.row6781.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_25 : scan edge (search edge 3) 4 [433] 433 innerTail81_25 best = best := by
  rw [innerTail81_25, scan, if_neg (by decide), innerCandidates81_25_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_26
    (if 3 ≤ innerCandidates81_25.length then search edge 3 [433, 6781] innerCandidates81_25 7214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_26

def innerTail81_24 : List Nat := 6379 :: innerTail81_25
def innerCandidates81_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates81_24_checked : candidates edge 4 433 6379 innerTail81_25 best.bound = innerCandidates81_24 := by
  unfold candidates
  have row : edge 6379 = SearchData.row6379.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_24 : scan edge (search edge 3) 4 [433] 433 innerTail81_24 best = best := by
  rw [innerTail81_24, scan, if_neg (by decide), innerCandidates81_24_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_25
    (if 3 ≤ innerCandidates81_24.length then search edge 3 [433, 6379] innerCandidates81_24 6812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_25

def innerTail81_23 : List Nat := 6247 :: innerTail81_24
def innerCandidates81_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates81_23_checked : candidates edge 4 433 6247 innerTail81_24 best.bound = innerCandidates81_23 := by
  unfold candidates
  have row : edge 6247 = SearchData.row6247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_23 : scan edge (search edge 3) 4 [433] 433 innerTail81_23 best = best := by
  rw [innerTail81_23, scan, if_neg (by decide), innerCandidates81_23_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_24
    (if 3 ≤ innerCandidates81_23.length then search edge 3 [433, 6247] innerCandidates81_23 6680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_24

def innerTail81_22 : List Nat := 6091 :: innerTail81_23
def innerCandidates81_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates81_22_checked : candidates edge 4 433 6091 innerTail81_23 best.bound = innerCandidates81_22 := by
  unfold candidates
  have row : edge 6091 = SearchData.row6091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_22 : scan edge (search edge 3) 4 [433] 433 innerTail81_22 best = best := by
  rw [innerTail81_22, scan, if_neg (by decide), innerCandidates81_22_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_23
    (if 3 ≤ innerCandidates81_22.length then search edge 3 [433, 6091] innerCandidates81_22 6524 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_23

def innerTail81_21 : List Nat := 5407 :: innerTail81_22
def innerCandidates81_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates81_21_checked : candidates edge 4 433 5407 innerTail81_22 best.bound = innerCandidates81_21 := by
  unfold candidates
  have row : edge 5407 = SearchData.row5407.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_21 : scan edge (search edge 3) 4 [433] 433 innerTail81_21 best = best := by
  rw [innerTail81_21, scan, if_neg (by decide), innerCandidates81_21_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_22
    (if 3 ≤ innerCandidates81_21.length then search edge 3 [433, 5407] innerCandidates81_21 5840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_22

def innerTail81_20 : List Nat := 5281 :: innerTail81_21
def innerCandidates81_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates81_20_checked : candidates edge 4 433 5281 innerTail81_21 best.bound = innerCandidates81_20 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_20 : scan edge (search edge 3) 4 [433] 433 innerTail81_20 best = best := by
  rw [innerTail81_20, scan, if_neg (by decide), innerCandidates81_20_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_21
    (if 3 ≤ innerCandidates81_20.length then search edge 3 [433, 5281] innerCandidates81_20 5714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_21

def innerTail81_19 : List Nat := 4903 :: innerTail81_20
def innerCandidates81_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates81_19_checked : candidates edge 4 433 4903 innerTail81_20 best.bound = innerCandidates81_19 := by
  unfold candidates
  have row : edge 4903 = SearchData.row4903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_19 : scan edge (search edge 3) 4 [433] 433 innerTail81_19 best = best := by
  rw [innerTail81_19, scan, if_neg (by decide), innerCandidates81_19_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_20
    (if 3 ≤ innerCandidates81_19.length then search edge 3 [433, 4903] innerCandidates81_19 5336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_20

def innerTail81_18 : List Nat := 4663 :: innerTail81_19
def innerCandidates81_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates81_18_checked : candidates edge 4 433 4663 innerTail81_19 best.bound = innerCandidates81_18 := by
  unfold candidates
  have row : edge 4663 = SearchData.row4663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_18 : scan edge (search edge 3) 4 [433] 433 innerTail81_18 best = best := by
  rw [innerTail81_18, scan, if_neg (by decide), innerCandidates81_18_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_19
    (if 3 ≤ innerCandidates81_18.length then search edge 3 [433, 4663] innerCandidates81_18 5096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_19

def innerTail81_17 : List Nat := 4483 :: innerTail81_18
def innerCandidates81_17 : List Nat := [7717]
set_option maxRecDepth 4096 in
theorem innerCandidates81_17_checked : candidates edge 4 433 4483 innerTail81_18 best.bound = innerCandidates81_17 := by
  unfold candidates
  have row : edge 4483 = SearchData.row4483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_17 : scan edge (search edge 3) 4 [433] 433 innerTail81_17 best = best := by
  rw [innerTail81_17, scan, if_neg (by decide), innerCandidates81_17_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_18
    (if 3 ≤ innerCandidates81_17.length then search edge 3 [433, 4483] innerCandidates81_17 4916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_18

def innerTail81_16 : List Nat := 4441 :: innerTail81_17
def innerCandidates81_16 : List Nat := [6247]
set_option maxRecDepth 4096 in
theorem innerCandidates81_16_checked : candidates edge 4 433 4441 innerTail81_17 best.bound = innerCandidates81_16 := by
  unfold candidates
  have row : edge 4441 = SearchData.row4441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_16 : scan edge (search edge 3) 4 [433] 433 innerTail81_16 best = best := by
  rw [innerTail81_16, scan, if_neg (by decide), innerCandidates81_16_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_17
    (if 3 ≤ innerCandidates81_16.length then search edge 3 [433, 4441] innerCandidates81_16 4874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_17

def innerTail81_15 : List Nat := 3739 :: innerTail81_16
def innerCandidates81_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates81_15_checked : candidates edge 4 433 3739 innerTail81_16 best.bound = innerCandidates81_15 := by
  unfold candidates
  have row : edge 3739 = SearchData.row3739.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_15 : scan edge (search edge 3) 4 [433] 433 innerTail81_15 best = best := by
  rw [innerTail81_15, scan, if_neg (by decide), innerCandidates81_15_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_16
    (if 3 ≤ innerCandidates81_15.length then search edge 3 [433, 3739] innerCandidates81_15 4172 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_16

def innerTail81_14 : List Nat := 3697 :: innerTail81_15
def innerCandidates81_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates81_14_checked : candidates edge 4 433 3697 innerTail81_15 best.bound = innerCandidates81_14 := by
  unfold candidates
  have row : edge 3697 = SearchData.row3697.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_14 : scan edge (search edge 3) 4 [433] 433 innerTail81_14 best = best := by
  rw [innerTail81_14, scan, if_neg (by decide), innerCandidates81_14_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_15
    (if 3 ≤ innerCandidates81_14.length then search edge 3 [433, 3697] innerCandidates81_14 4130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_15

def innerTail81_13 : List Nat := 3613 :: innerTail81_14
def innerCandidates81_13 : List Nat := [3697]
set_option maxRecDepth 4096 in
theorem innerCandidates81_13_checked : candidates edge 4 433 3613 innerTail81_14 best.bound = innerCandidates81_13 := by
  unfold candidates
  have row : edge 3613 = SearchData.row3613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_13 : scan edge (search edge 3) 4 [433] 433 innerTail81_13 best = best := by
  rw [innerTail81_13, scan, if_neg (by decide), innerCandidates81_13_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_14
    (if 3 ≤ innerCandidates81_13.length then search edge 3 [433, 3613] innerCandidates81_13 4046 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_14

def innerTail81_12 : List Nat := 3529 :: innerTail81_13
def innerCandidates81_12 : List Nat := [7393]
set_option maxRecDepth 4096 in
theorem innerCandidates81_12_checked : candidates edge 4 433 3529 innerTail81_13 best.bound = innerCandidates81_12 := by
  unfold candidates
  have row : edge 3529 = SearchData.row3529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_12 : scan edge (search edge 3) 4 [433] 433 innerTail81_12 best = best := by
  rw [innerTail81_12, scan, if_neg (by decide), innerCandidates81_12_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_13
    (if 3 ≤ innerCandidates81_12.length then search edge 3 [433, 3529] innerCandidates81_12 3962 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_13

def innerTail81_11 : List Nat := 3391 :: innerTail81_12
def innerCandidates81_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates81_11_checked : candidates edge 4 433 3391 innerTail81_12 best.bound = innerCandidates81_11 := by
  unfold candidates
  have row : edge 3391 = SearchData.row3391.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_11 : scan edge (search edge 3) 4 [433] 433 innerTail81_11 best = best := by
  rw [innerTail81_11, scan, if_neg (by decide), innerCandidates81_11_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_12
    (if 3 ≤ innerCandidates81_11.length then search edge 3 [433, 3391] innerCandidates81_11 3824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_12

def innerTail81_10 : List Nat := 2857 :: innerTail81_11
def innerCandidates81_10 : List Nat := [6091]
set_option maxRecDepth 4096 in
theorem innerCandidates81_10_checked : candidates edge 4 433 2857 innerTail81_11 best.bound = innerCandidates81_10 := by
  unfold candidates
  have row : edge 2857 = SearchData.row2857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_10 : scan edge (search edge 3) 4 [433] 433 innerTail81_10 best = best := by
  rw [innerTail81_10, scan, if_neg (by decide), innerCandidates81_10_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_11
    (if 3 ≤ innerCandidates81_10.length then search edge 3 [433, 2857] innerCandidates81_10 3290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_11

def innerTail81_9 : List Nat := 2719 :: innerTail81_10
def innerCandidates81_9 : List Nat := [3391]
set_option maxRecDepth 4096 in
theorem innerCandidates81_9_checked : candidates edge 4 433 2719 innerTail81_10 best.bound = innerCandidates81_9 := by
  unfold candidates
  have row : edge 2719 = SearchData.row2719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_9 : scan edge (search edge 3) 4 [433] 433 innerTail81_9 best = best := by
  rw [innerTail81_9, scan, if_neg (by decide), innerCandidates81_9_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_10
    (if 3 ≤ innerCandidates81_9.length then search edge 3 [433, 2719] innerCandidates81_9 3152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_10

def innerTail81_8 : List Nat := 2203 :: innerTail81_9
def innerCandidates81_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates81_8_checked : candidates edge 4 433 2203 innerTail81_9 best.bound = innerCandidates81_8 := by
  unfold candidates
  have row : edge 2203 = PrimePairCache.smallRow2203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_8 : scan edge (search edge 3) 4 [433] 433 innerTail81_8 best = best := by
  rw [innerTail81_8, scan, if_neg (by decide), innerCandidates81_8_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_9
    (if 3 ≤ innerCandidates81_8.length then search edge 3 [433, 2203] innerCandidates81_8 2636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_9

def innerTail81_7 : List Nat := 1801 :: innerTail81_8
def innerCandidates81_7 : List Nat := [3613]
set_option maxRecDepth 4096 in
theorem innerCandidates81_7_checked : candidates edge 4 433 1801 innerTail81_8 best.bound = innerCandidates81_7 := by
  unfold candidates
  have row : edge 1801 = PrimePairCache.smallRow1801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_7 : scan edge (search edge 3) 4 [433] 433 innerTail81_7 best = best := by
  rw [innerTail81_7, scan, if_neg (by decide), innerCandidates81_7_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_8
    (if 3 ≤ innerCandidates81_7.length then search edge 3 [433, 1801] innerCandidates81_7 2234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_8

def innerTail81_6 : List Nat := 1471 :: innerTail81_7
def innerCandidates81_6 : List Nat := [3613]
set_option maxRecDepth 4096 in
theorem innerCandidates81_6_checked : candidates edge 4 433 1471 innerTail81_7 best.bound = innerCandidates81_6 := by
  unfold candidates
  have row : edge 1471 = PrimePairCache.smallRow1471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_6 : scan edge (search edge 3) 4 [433] 433 innerTail81_6 best = best := by
  rw [innerTail81_6, scan, if_neg (by decide), innerCandidates81_6_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_7
    (if 3 ≤ innerCandidates81_6.length then search edge 3 [433, 1471] innerCandidates81_6 1904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_7

def innerTail81_5 : List Nat := 1087 :: innerTail81_6
def innerCandidates81_5 : List Nat := [6091]
set_option maxRecDepth 4096 in
theorem innerCandidates81_5_checked : candidates edge 4 433 1087 innerTail81_6 best.bound = innerCandidates81_5 := by
  unfold candidates
  have row : edge 1087 = PrimePairCache.smallRow1087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_5 : scan edge (search edge 3) 4 [433] 433 innerTail81_5 best = best := by
  rw [innerTail81_5, scan, if_neg (by decide), innerCandidates81_5_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_6
    (if 3 ≤ innerCandidates81_5.length then search edge 3 [433, 1087] innerCandidates81_5 1520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_6

def innerTail81_4 : List Nat := 1009 :: innerTail81_5
def innerCandidates81_4 : List Nat := [3613, 7393]
set_option maxRecDepth 4096 in
theorem innerCandidates81_4_checked : candidates edge 4 433 1009 innerTail81_5 best.bound = innerCandidates81_4 := by
  unfold candidates
  have row : edge 1009 = PrimePairCache.smallRow1009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_4 : scan edge (search edge 3) 4 [433] 433 innerTail81_4 best = best := by
  rw [innerTail81_4, scan, if_neg (by decide), innerCandidates81_4_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_5
    (if 3 ≤ innerCandidates81_4.length then search edge 3 [433, 1009] innerCandidates81_4 1442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_5

def innerTail81_3 : List Nat := 883 :: innerTail81_4
def innerCandidates81_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates81_3_checked : candidates edge 4 433 883 innerTail81_4 best.bound = innerCandidates81_3 := by
  unfold candidates
  have row : edge 883 = PrimePairCache.smallRow883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_3 : scan edge (search edge 3) 4 [433] 433 innerTail81_3 best = best := by
  rw [innerTail81_3, scan, if_neg (by decide), innerCandidates81_3_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_4
    (if 3 ≤ innerCandidates81_3.length then search edge 3 [433, 883] innerCandidates81_3 1316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_4

def innerTail81_2 : List Nat := 859 :: innerTail81_3
def innerCandidates81_2 : List Nat := [3391]
set_option maxRecDepth 4096 in
theorem innerCandidates81_2_checked : candidates edge 4 433 859 innerTail81_3 best.bound = innerCandidates81_2 := by
  unfold candidates
  have row : edge 859 = PrimePairCache.smallRow859.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_2 : scan edge (search edge 3) 4 [433] 433 innerTail81_2 best = best := by
  rw [innerTail81_2, scan, if_neg (by decide), innerCandidates81_2_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_3
    (if 3 ≤ innerCandidates81_2.length then search edge 3 [433, 859] innerCandidates81_2 1292 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_3

def innerTail81_1 : List Nat := 787 :: innerTail81_2
def innerCandidates81_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates81_1_checked : candidates edge 4 433 787 innerTail81_2 best.bound = innerCandidates81_1 := by
  unfold candidates
  have row : edge 787 = PrimePairCache.smallRow787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_1 : scan edge (search edge 3) 4 [433] 433 innerTail81_1 best = best := by
  rw [innerTail81_1, scan, if_neg (by decide), innerCandidates81_1_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_2
    (if 3 ≤ innerCandidates81_1.length then search edge 3 [433, 787] innerCandidates81_1 1220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_2

def innerTail81_0 : List Nat := 571 :: innerTail81_1
def innerCandidates81_0 : List Nat := [4483]
set_option maxRecDepth 4096 in
theorem innerCandidates81_0_checked : candidates edge 4 433 571 innerTail81_1 best.bound = innerCandidates81_0 := by
  unfold candidates
  have row : edge 571 = PrimePairCache.smallRow571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan81_0 : scan edge (search edge 3) 4 [433] 433 innerTail81_0 best = best := by
  rw [innerTail81_0, scan, if_neg (by decide), innerCandidates81_0_checked]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_1
    (if 3 ≤ innerCandidates81_0.length then search edge 3 [433, 571] innerCandidates81_0 1004 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan81_1

theorem rootBranch81 : search edge 4 [433] rootCandidates81 433 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [433] 433 innerTail81_0 best = best
  exact innerScan81_0

end ProjectEulerSolutions.PrimePairUpper.Search
