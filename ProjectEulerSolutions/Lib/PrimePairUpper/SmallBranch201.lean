import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates100 : List Nat := [587, 809, 971, 1013, 1259, 1427, 1523, 1601, 1973, 1979, 2267, 2909, 3119, 3821, 4493, 4799, 5237, 5261, 5297, 5501, 5717, 6551, 6581, 6791, 7019, 7043, 7481]
set_option maxRecDepth 4096 in
theorem rootCandidates100_checked : candidates edge 5 0 563 tail101 best.bound = rootCandidates100 := by
  unfold candidates
  have row : edge 563 = PrimePairCache.smallRow563.contains := by funext q; rfl
  rw [row]
  decide

def innerTail100_27 : List Nat := []
theorem innerScan100_27 : scan edge (search edge 3) 4 [563] 563 innerTail100_27 best = best := rfl

def innerTail100_26 : List Nat := 7481 :: innerTail100_27
def innerCandidates100_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates100_26_checked : candidates edge 4 563 7481 innerTail100_27 best.bound = innerCandidates100_26 := by
  unfold candidates
  have row : edge 7481 = SearchData.row7481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_26 : scan edge (search edge 3) 4 [563] 563 innerTail100_26 best = best := by
  rw [innerTail100_26, scan, if_neg (by decide), innerCandidates100_26_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_27
    (if 3 ≤ innerCandidates100_26.length then search edge 3 [563, 7481] innerCandidates100_26 8044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_27

def innerTail100_25 : List Nat := 7043 :: innerTail100_26
def innerCandidates100_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates100_25_checked : candidates edge 4 563 7043 innerTail100_26 best.bound = innerCandidates100_25 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_25 : scan edge (search edge 3) 4 [563] 563 innerTail100_25 best = best := by
  rw [innerTail100_25, scan, if_neg (by decide), innerCandidates100_25_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_26
    (if 3 ≤ innerCandidates100_25.length then search edge 3 [563, 7043] innerCandidates100_25 7606 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_26

def innerTail100_24 : List Nat := 7019 :: innerTail100_25
def innerCandidates100_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates100_24_checked : candidates edge 4 563 7019 innerTail100_25 best.bound = innerCandidates100_24 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_24 : scan edge (search edge 3) 4 [563] 563 innerTail100_24 best = best := by
  rw [innerTail100_24, scan, if_neg (by decide), innerCandidates100_24_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_25
    (if 3 ≤ innerCandidates100_24.length then search edge 3 [563, 7019] innerCandidates100_24 7582 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_25

def innerTail100_23 : List Nat := 6791 :: innerTail100_24
def innerCandidates100_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates100_23_checked : candidates edge 4 563 6791 innerTail100_24 best.bound = innerCandidates100_23 := by
  unfold candidates
  have row : edge 6791 = SearchData.row6791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_23 : scan edge (search edge 3) 4 [563] 563 innerTail100_23 best = best := by
  rw [innerTail100_23, scan, if_neg (by decide), innerCandidates100_23_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_24
    (if 3 ≤ innerCandidates100_23.length then search edge 3 [563, 6791] innerCandidates100_23 7354 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_24

def innerTail100_22 : List Nat := 6581 :: innerTail100_23
def innerCandidates100_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates100_22_checked : candidates edge 4 563 6581 innerTail100_23 best.bound = innerCandidates100_22 := by
  unfold candidates
  have row : edge 6581 = SearchData.row6581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_22 : scan edge (search edge 3) 4 [563] 563 innerTail100_22 best = best := by
  rw [innerTail100_22, scan, if_neg (by decide), innerCandidates100_22_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_23
    (if 3 ≤ innerCandidates100_22.length then search edge 3 [563, 6581] innerCandidates100_22 7144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_23

def innerTail100_21 : List Nat := 6551 :: innerTail100_22
def innerCandidates100_21 : List Nat := [7481]
set_option maxRecDepth 4096 in
theorem innerCandidates100_21_checked : candidates edge 4 563 6551 innerTail100_22 best.bound = innerCandidates100_21 := by
  unfold candidates
  have row : edge 6551 = SearchData.row6551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_21 : scan edge (search edge 3) 4 [563] 563 innerTail100_21 best = best := by
  rw [innerTail100_21, scan, if_neg (by decide), innerCandidates100_21_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_22
    (if 3 ≤ innerCandidates100_21.length then search edge 3 [563, 6551] innerCandidates100_21 7114 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_22

def innerTail100_20 : List Nat := 5717 :: innerTail100_21
def innerCandidates100_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates100_20_checked : candidates edge 4 563 5717 innerTail100_21 best.bound = innerCandidates100_20 := by
  unfold candidates
  have row : edge 5717 = SearchData.row5717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_20 : scan edge (search edge 3) 4 [563] 563 innerTail100_20 best = best := by
  rw [innerTail100_20, scan, if_neg (by decide), innerCandidates100_20_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_21
    (if 3 ≤ innerCandidates100_20.length then search edge 3 [563, 5717] innerCandidates100_20 6280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_21

def innerTail100_19 : List Nat := 5501 :: innerTail100_20
def innerCandidates100_19 : List Nat := [7481]
set_option maxRecDepth 4096 in
theorem innerCandidates100_19_checked : candidates edge 4 563 5501 innerTail100_20 best.bound = innerCandidates100_19 := by
  unfold candidates
  have row : edge 5501 = SearchData.row5501.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_19 : scan edge (search edge 3) 4 [563] 563 innerTail100_19 best = best := by
  rw [innerTail100_19, scan, if_neg (by decide), innerCandidates100_19_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_20
    (if 3 ≤ innerCandidates100_19.length then search edge 3 [563, 5501] innerCandidates100_19 6064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_20

def innerTail100_18 : List Nat := 5297 :: innerTail100_19
def innerCandidates100_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates100_18_checked : candidates edge 4 563 5297 innerTail100_19 best.bound = innerCandidates100_18 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_18 : scan edge (search edge 3) 4 [563] 563 innerTail100_18 best = best := by
  rw [innerTail100_18, scan, if_neg (by decide), innerCandidates100_18_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_19
    (if 3 ≤ innerCandidates100_18.length then search edge 3 [563, 5297] innerCandidates100_18 5860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_19

def innerTail100_17 : List Nat := 5261 :: innerTail100_18
def innerCandidates100_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates100_17_checked : candidates edge 4 563 5261 innerTail100_18 best.bound = innerCandidates100_17 := by
  unfold candidates
  have row : edge 5261 = SearchData.row5261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_17 : scan edge (search edge 3) 4 [563] 563 innerTail100_17 best = best := by
  rw [innerTail100_17, scan, if_neg (by decide), innerCandidates100_17_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_18
    (if 3 ≤ innerCandidates100_17.length then search edge 3 [563, 5261] innerCandidates100_17 5824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_18

def innerTail100_16 : List Nat := 5237 :: innerTail100_17
def innerCandidates100_16 : List Nat := [7043]
set_option maxRecDepth 4096 in
theorem innerCandidates100_16_checked : candidates edge 4 563 5237 innerTail100_17 best.bound = innerCandidates100_16 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_16 : scan edge (search edge 3) 4 [563] 563 innerTail100_16 best = best := by
  rw [innerTail100_16, scan, if_neg (by decide), innerCandidates100_16_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_17
    (if 3 ≤ innerCandidates100_16.length then search edge 3 [563, 5237] innerCandidates100_16 5800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_17

def innerTail100_15 : List Nat := 4799 :: innerTail100_16
def innerCandidates100_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates100_15_checked : candidates edge 4 563 4799 innerTail100_16 best.bound = innerCandidates100_15 := by
  unfold candidates
  have row : edge 4799 = SearchData.row4799.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_15 : scan edge (search edge 3) 4 [563] 563 innerTail100_15 best = best := by
  rw [innerTail100_15, scan, if_neg (by decide), innerCandidates100_15_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_16
    (if 3 ≤ innerCandidates100_15.length then search edge 3 [563, 4799] innerCandidates100_15 5362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_16

def innerTail100_14 : List Nat := 4493 :: innerTail100_15
def innerCandidates100_14 : List Nat := [5717]
set_option maxRecDepth 4096 in
theorem innerCandidates100_14_checked : candidates edge 4 563 4493 innerTail100_15 best.bound = innerCandidates100_14 := by
  unfold candidates
  have row : edge 4493 = SearchData.row4493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_14 : scan edge (search edge 3) 4 [563] 563 innerTail100_14 best = best := by
  rw [innerTail100_14, scan, if_neg (by decide), innerCandidates100_14_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_15
    (if 3 ≤ innerCandidates100_14.length then search edge 3 [563, 4493] innerCandidates100_14 5056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_15

def innerTail100_13 : List Nat := 3821 :: innerTail100_14
def innerCandidates100_13 : List Nat := [7019]
set_option maxRecDepth 4096 in
theorem innerCandidates100_13_checked : candidates edge 4 563 3821 innerTail100_14 best.bound = innerCandidates100_13 := by
  unfold candidates
  have row : edge 3821 = SearchData.row3821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_13 : scan edge (search edge 3) 4 [563] 563 innerTail100_13 best = best := by
  rw [innerTail100_13, scan, if_neg (by decide), innerCandidates100_13_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_14
    (if 3 ≤ innerCandidates100_13.length then search edge 3 [563, 3821] innerCandidates100_13 4384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_14

def innerTail100_12 : List Nat := 3119 :: innerTail100_13
def innerCandidates100_12 : List Nat := [5237]
set_option maxRecDepth 4096 in
theorem innerCandidates100_12_checked : candidates edge 4 563 3119 innerTail100_13 best.bound = innerCandidates100_12 := by
  unfold candidates
  have row : edge 3119 = SearchData.row3119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_12 : scan edge (search edge 3) 4 [563] 563 innerTail100_12 best = best := by
  rw [innerTail100_12, scan, if_neg (by decide), innerCandidates100_12_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_13
    (if 3 ≤ innerCandidates100_12.length then search edge 3 [563, 3119] innerCandidates100_12 3682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_13

def innerTail100_11 : List Nat := 2909 :: innerTail100_12
def innerCandidates100_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates100_11_checked : candidates edge 4 563 2909 innerTail100_12 best.bound = innerCandidates100_11 := by
  unfold candidates
  have row : edge 2909 = SearchData.row2909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_11 : scan edge (search edge 3) 4 [563] 563 innerTail100_11 best = best := by
  rw [innerTail100_11, scan, if_neg (by decide), innerCandidates100_11_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_12
    (if 3 ≤ innerCandidates100_11.length then search edge 3 [563, 2909] innerCandidates100_11 3472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_12

def innerTail100_10 : List Nat := 2267 :: innerTail100_11
def innerCandidates100_10 : List Nat := [6791]
set_option maxRecDepth 4096 in
theorem innerCandidates100_10_checked : candidates edge 4 563 2267 innerTail100_11 best.bound = innerCandidates100_10 := by
  unfold candidates
  have row : edge 2267 = PrimePairCache.smallRow2267.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_10 : scan edge (search edge 3) 4 [563] 563 innerTail100_10 best = best := by
  rw [innerTail100_10, scan, if_neg (by decide), innerCandidates100_10_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_11
    (if 3 ≤ innerCandidates100_10.length then search edge 3 [563, 2267] innerCandidates100_10 2830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_11

def innerTail100_9 : List Nat := 1979 :: innerTail100_10
def innerCandidates100_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates100_9_checked : candidates edge 4 563 1979 innerTail100_10 best.bound = innerCandidates100_9 := by
  unfold candidates
  have row : edge 1979 = PrimePairCache.smallRow1979.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_9 : scan edge (search edge 3) 4 [563] 563 innerTail100_9 best = best := by
  rw [innerTail100_9, scan, if_neg (by decide), innerCandidates100_9_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_10
    (if 3 ≤ innerCandidates100_9.length then search edge 3 [563, 1979] innerCandidates100_9 2542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_10

def innerTail100_8 : List Nat := 1973 :: innerTail100_9
def innerCandidates100_8 : List Nat := [1979, 4493]
set_option maxRecDepth 4096 in
theorem innerCandidates100_8_checked : candidates edge 4 563 1973 innerTail100_9 best.bound = innerCandidates100_8 := by
  unfold candidates
  have row : edge 1973 = PrimePairCache.smallRow1973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_8 : scan edge (search edge 3) 4 [563] 563 innerTail100_8 best = best := by
  rw [innerTail100_8, scan, if_neg (by decide), innerCandidates100_8_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_9
    (if 3 ≤ innerCandidates100_8.length then search edge 3 [563, 1973] innerCandidates100_8 2536 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_9

def innerTail100_7 : List Nat := 1601 :: innerTail100_8
def innerCandidates100_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates100_7_checked : candidates edge 4 563 1601 innerTail100_8 best.bound = innerCandidates100_7 := by
  unfold candidates
  have row : edge 1601 = PrimePairCache.smallRow1601.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_7 : scan edge (search edge 3) 4 [563] 563 innerTail100_7 best = best := by
  rw [innerTail100_7, scan, if_neg (by decide), innerCandidates100_7_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_8
    (if 3 ≤ innerCandidates100_7.length then search edge 3 [563, 1601] innerCandidates100_7 2164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_8

def innerTail100_6 : List Nat := 1523 :: innerTail100_7
def innerCandidates100_6 : List Nat := [2909, 7043]
set_option maxRecDepth 4096 in
theorem innerCandidates100_6_checked : candidates edge 4 563 1523 innerTail100_7 best.bound = innerCandidates100_6 := by
  unfold candidates
  have row : edge 1523 = PrimePairCache.smallRow1523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_6 : scan edge (search edge 3) 4 [563] 563 innerTail100_6 best = best := by
  rw [innerTail100_6, scan, if_neg (by decide), innerCandidates100_6_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_7
    (if 3 ≤ innerCandidates100_6.length then search edge 3 [563, 1523] innerCandidates100_6 2086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_7

def innerTail100_5 : List Nat := 1427 :: innerTail100_6
def innerCandidates100_5 : List Nat := [5297, 5501]
set_option maxRecDepth 4096 in
theorem innerCandidates100_5_checked : candidates edge 4 563 1427 innerTail100_6 best.bound = innerCandidates100_5 := by
  unfold candidates
  have row : edge 1427 = PrimePairCache.smallRow1427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_5 : scan edge (search edge 3) 4 [563] 563 innerTail100_5 best = best := by
  rw [innerTail100_5, scan, if_neg (by decide), innerCandidates100_5_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_6
    (if 3 ≤ innerCandidates100_5.length then search edge 3 [563, 1427] innerCandidates100_5 1990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_6

def innerTail100_4 : List Nat := 1259 :: innerTail100_5
def innerCandidates100_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates100_4_checked : candidates edge 4 563 1259 innerTail100_5 best.bound = innerCandidates100_4 := by
  unfold candidates
  have row : edge 1259 = PrimePairCache.smallRow1259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_4 : scan edge (search edge 3) 4 [563] 563 innerTail100_4 best = best := by
  rw [innerTail100_4, scan, if_neg (by decide), innerCandidates100_4_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_5
    (if 3 ≤ innerCandidates100_4.length then search edge 3 [563, 1259] innerCandidates100_4 1822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_5

def innerTail100_3 : List Nat := 1013 :: innerTail100_4
def innerCandidates100_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates100_3_checked : candidates edge 4 563 1013 innerTail100_4 best.bound = innerCandidates100_3 := by
  unfold candidates
  have row : edge 1013 = PrimePairCache.smallRow1013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_3 : scan edge (search edge 3) 4 [563] 563 innerTail100_3 best = best := by
  rw [innerTail100_3, scan, if_neg (by decide), innerCandidates100_3_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_4
    (if 3 ≤ innerCandidates100_3.length then search edge 3 [563, 1013] innerCandidates100_3 1576 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_4

def innerTail100_2 : List Nat := 971 :: innerTail100_3
def innerCandidates100_2 : List Nat := [5501]
set_option maxRecDepth 4096 in
theorem innerCandidates100_2_checked : candidates edge 4 563 971 innerTail100_3 best.bound = innerCandidates100_2 := by
  unfold candidates
  have row : edge 971 = PrimePairCache.smallRow971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_2 : scan edge (search edge 3) 4 [563] 563 innerTail100_2 best = best := by
  rw [innerTail100_2, scan, if_neg (by decide), innerCandidates100_2_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_3
    (if 3 ≤ innerCandidates100_2.length then search edge 3 [563, 971] innerCandidates100_2 1534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_3

def innerTail100_1 : List Nat := 809 :: innerTail100_2
def innerCandidates100_1 : List Nat := [1427, 5237]
set_option maxRecDepth 4096 in
theorem innerCandidates100_1_checked : candidates edge 4 563 809 innerTail100_2 best.bound = innerCandidates100_1 := by
  unfold candidates
  have row : edge 809 = PrimePairCache.smallRow809.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_1 : scan edge (search edge 3) 4 [563] 563 innerTail100_1 best = best := by
  rw [innerTail100_1, scan, if_neg (by decide), innerCandidates100_1_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_2
    (if 3 ≤ innerCandidates100_1.length then search edge 3 [563, 809] innerCandidates100_1 1372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_2

def innerTail100_0 : List Nat := 587 :: innerTail100_1
def innerCandidates100_0 : List Nat := [7481]
set_option maxRecDepth 4096 in
theorem innerCandidates100_0_checked : candidates edge 4 563 587 innerTail100_1 best.bound = innerCandidates100_0 := by
  unfold candidates
  have row : edge 587 = PrimePairCache.smallRow587.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan100_0 : scan edge (search edge 3) 4 [563] 563 innerTail100_0 best = best := by
  rw [innerTail100_0, scan, if_neg (by decide), innerCandidates100_0_checked]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_1
    (if 3 ≤ innerCandidates100_0.length then search edge 3 [563, 587] innerCandidates100_0 1150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan100_1

theorem rootBranch100 : search edge 4 [563] rootCandidates100 563 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [563] 563 innerTail100_0 best = best
  exact innerScan100_0

def rootCandidates99 : List Nat := [1487, 1721, 2687, 2729, 3221, 3389, 3929, 3947, 4703, 5651, 5879, 6047, 6257, 6491, 7079, 8087, 8093, 8291, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates99_checked : candidates edge 5 0 557 tail100 best.bound = rootCandidates99 := by
  unfold candidates
  have row : edge 557 = PrimePairCache.smallRow557.contains := by funext q; rfl
  rw [row]
  decide

def innerTail99_19 : List Nat := []
theorem innerScan99_19 : scan edge (search edge 3) 4 [557] 557 innerTail99_19 best = best := rfl

def innerTail99_18 : List Nat := 8387 :: innerTail99_19
def innerCandidates99_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates99_18_checked : candidates edge 4 557 8387 innerTail99_19 best.bound = innerCandidates99_18 := by
  unfold candidates
  have row : edge 8387 = SearchData.row8387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan99_18 : scan edge (search edge 3) 4 [557] 557 innerTail99_18 best = best := by
  rw [innerTail99_18, scan, if_neg (by decide), innerCandidates99_18_checked]
  change scan edge (search edge 3) 4 [557] 557 innerTail99_19
    (if 3 ≤ innerCandidates99_18.length then search edge 3 [557, 8387] innerCandidates99_18 8944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan99_19

def innerTail99_17 : List Nat := 8291 :: innerTail99_18
def innerCandidates99_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates99_17_checked : candidates edge 4 557 8291 innerTail99_18 best.bound = innerCandidates99_17 := by
  unfold candidates
  have row : edge 8291 = SearchData.row8291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan99_17 : scan edge (search edge 3) 4 [557] 557 innerTail99_17 best = best := by
  rw [innerTail99_17, scan, if_neg (by decide), innerCandidates99_17_checked]
  change scan edge (search edge 3) 4 [557] 557 innerTail99_18
    (if 3 ≤ innerCandidates99_17.length then search edge 3 [557, 8291] innerCandidates99_17 8848 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan99_18

def innerTail99_16 : List Nat := 8093 :: innerTail99_17
def innerCandidates99_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates99_16_checked : candidates edge 4 557 8093 innerTail99_17 best.bound = innerCandidates99_16 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan99_16 : scan edge (search edge 3) 4 [557] 557 innerTail99_16 best = best := by
  rw [innerTail99_16, scan, if_neg (by decide), innerCandidates99_16_checked]
  change scan edge (search edge 3) 4 [557] 557 innerTail99_17
    (if 3 ≤ innerCandidates99_16.length then search edge 3 [557, 8093] innerCandidates99_16 8650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan99_17

def innerTail99_15 : List Nat := 8087 :: innerTail99_16
def innerCandidates99_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates99_15_checked : candidates edge 4 557 8087 innerTail99_16 best.bound = innerCandidates99_15 := by
  unfold candidates
  have row : edge 8087 = SearchData.row8087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan99_15 : scan edge (search edge 3) 4 [557] 557 innerTail99_15 best = best := by
  rw [innerTail99_15, scan, if_neg (by decide), innerCandidates99_15_checked]
  change scan edge (search edge 3) 4 [557] 557 innerTail99_16
    (if 3 ≤ innerCandidates99_15.length then search edge 3 [557, 8087] innerCandidates99_15 8644 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan99_16

def innerTail99_14 : List Nat := 7079 :: innerTail99_15
def innerCandidates99_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates99_14_checked : candidates edge 4 557 7079 innerTail99_15 best.bound = innerCandidates99_14 := by
  unfold candidates
  have row : edge 7079 = SearchData.row7079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan99_14 : scan edge (search edge 3) 4 [557] 557 innerTail99_14 best = best := by
  rw [innerTail99_14, scan, if_neg (by decide), innerCandidates99_14_checked]
  change scan edge (search edge 3) 4 [557] 557 innerTail99_15
    (if 3 ≤ innerCandidates99_14.length then search edge 3 [557, 7079] innerCandidates99_14 7636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan99_15

def innerTail99_13 : List Nat := 6491 :: innerTail99_14
def innerCandidates99_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates99_13_checked : candidates edge 4 557 6491 innerTail99_14 best.bound = innerCandidates99_13 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan99_13 : scan edge (search edge 3) 4 [557] 557 innerTail99_13 best = best := by
  rw [innerTail99_13, scan, if_neg (by decide), innerCandidates99_13_checked]
  change scan edge (search edge 3) 4 [557] 557 innerTail99_14
    (if 3 ≤ innerCandidates99_13.length then search edge 3 [557, 6491] innerCandidates99_13 7048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan99_14

def innerTail99_12 : List Nat := 6257 :: innerTail99_13
def innerCandidates99_12 : List Nat := [8387]
set_option maxRecDepth 4096 in
theorem innerCandidates99_12_checked : candidates edge 4 557 6257 innerTail99_13 best.bound = innerCandidates99_12 := by
  unfold candidates
  have row : edge 6257 = SearchData.row6257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan99_12 : scan edge (search edge 3) 4 [557] 557 innerTail99_12 best = best := by
  rw [innerTail99_12, scan, if_neg (by decide), innerCandidates99_12_checked]
  change scan edge (search edge 3) 4 [557] 557 innerTail99_13
    (if 3 ≤ innerCandidates99_12.length then search edge 3 [557, 6257] innerCandidates99_12 6814 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan99_13

def innerTail99_11 : List Nat := 6047 :: innerTail99_12
def innerCandidates99_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates99_11_checked : candidates edge 4 557 6047 innerTail99_12 best.bound = innerCandidates99_11 := by
  unfold candidates
  have row : edge 6047 = SearchData.row6047.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan99_11 : scan edge (search edge 3) 4 [557] 557 innerTail99_11 best = best := by
  rw [innerTail99_11, scan, if_neg (by decide), innerCandidates99_11_checked]
  change scan edge (search edge 3) 4 [557] 557 innerTail99_12
    (if 3 ≤ innerCandidates99_11.length then search edge 3 [557, 6047] innerCandidates99_11 6604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan99_12

def innerTail99_10 : List Nat := 5879 :: innerTail99_11
def innerCandidates99_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates99_10_checked : candidates edge 4 557 5879 innerTail99_11 best.bound = innerCandidates99_10 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan99_10 : scan edge (search edge 3) 4 [557] 557 innerTail99_10 best = best := by
  rw [innerTail99_10, scan, if_neg (by decide), innerCandidates99_10_checked]
  change scan edge (search edge 3) 4 [557] 557 innerTail99_11
    (if 3 ≤ innerCandidates99_10.length then search edge 3 [557, 5879] innerCandidates99_10 6436 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan99_11

def innerTail99_9 : List Nat := 5651 :: innerTail99_10
def innerCandidates99_9 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates99_9_checked : candidates edge 4 557 5651 innerTail99_10 best.bound = innerCandidates99_9 := by
  unfold candidates
  have row : edge 5651 = SearchData.row5651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan99_9 : scan edge (search edge 3) 4 [557] 557 innerTail99_9 best = best := by
  rw [innerTail99_9, scan, if_neg (by decide), innerCandidates99_9_checked]
  change scan edge (search edge 3) 4 [557] 557 innerTail99_10
    (if 3 ≤ innerCandidates99_9.length then search edge 3 [557, 5651] innerCandidates99_9 6208 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan99_10

def innerTail99_8 : List Nat := 4703 :: innerTail99_9
def innerCandidates99_8 : List Nat := [6257]
set_option maxRecDepth 4096 in
theorem innerCandidates99_8_checked : candidates edge 4 557 4703 innerTail99_9 best.bound = innerCandidates99_8 := by
  unfold candidates
  have row : edge 4703 = SearchData.row4703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan99_8 : scan edge (search edge 3) 4 [557] 557 innerTail99_8 best = best := by
  rw [innerTail99_8, scan, if_neg (by decide), innerCandidates99_8_checked]
  change scan edge (search edge 3) 4 [557] 557 innerTail99_9
    (if 3 ≤ innerCandidates99_8.length then search edge 3 [557, 4703] innerCandidates99_8 5260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan99_9

def innerTail99_7 : List Nat := 3947 :: innerTail99_8
def innerCandidates99_7 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates99_7_checked : candidates edge 4 557 3947 innerTail99_8 best.bound = innerCandidates99_7 := by
  unfold candidates
  have row : edge 3947 = SearchData.row3947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan99_7 : scan edge (search edge 3) 4 [557] 557 innerTail99_7 best = best := by
  rw [innerTail99_7, scan, if_neg (by decide), innerCandidates99_7_checked]
  change scan edge (search edge 3) 4 [557] 557 innerTail99_8
    (if 3 ≤ innerCandidates99_7.length then search edge 3 [557, 3947] innerCandidates99_7 4504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan99_8

def innerTail99_6 : List Nat := 3929 :: innerTail99_7
def innerCandidates99_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates99_6_checked : candidates edge 4 557 3929 innerTail99_7 best.bound = innerCandidates99_6 := by
  unfold candidates
  have row : edge 3929 = SearchData.row3929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan99_6 : scan edge (search edge 3) 4 [557] 557 innerTail99_6 best = best := by
  rw [innerTail99_6, scan, if_neg (by decide), innerCandidates99_6_checked]
  change scan edge (search edge 3) 4 [557] 557 innerTail99_7
    (if 3 ≤ innerCandidates99_6.length then search edge 3 [557, 3929] innerCandidates99_6 4486 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan99_7

def innerTail99_5 : List Nat := 3389 :: innerTail99_6
def innerCandidates99_5 : List Nat := [5879]
set_option maxRecDepth 4096 in
theorem innerCandidates99_5_checked : candidates edge 4 557 3389 innerTail99_6 best.bound = innerCandidates99_5 := by
  unfold candidates
  have row : edge 3389 = SearchData.row3389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan99_5 : scan edge (search edge 3) 4 [557] 557 innerTail99_5 best = best := by
  rw [innerTail99_5, scan, if_neg (by decide), innerCandidates99_5_checked]
  change scan edge (search edge 3) 4 [557] 557 innerTail99_6
    (if 3 ≤ innerCandidates99_5.length then search edge 3 [557, 3389] innerCandidates99_5 3946 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan99_6

def innerTail99_4 : List Nat := 3221 :: innerTail99_5
def innerCandidates99_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates99_4_checked : candidates edge 4 557 3221 innerTail99_5 best.bound = innerCandidates99_4 := by
  unfold candidates
  have row : edge 3221 = SearchData.row3221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan99_4 : scan edge (search edge 3) 4 [557] 557 innerTail99_4 best = best := by
  rw [innerTail99_4, scan, if_neg (by decide), innerCandidates99_4_checked]
  change scan edge (search edge 3) 4 [557] 557 innerTail99_5
    (if 3 ≤ innerCandidates99_4.length then search edge 3 [557, 3221] innerCandidates99_4 3778 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan99_5

def innerTail99_3 : List Nat := 2729 :: innerTail99_4
def innerCandidates99_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates99_3_checked : candidates edge 4 557 2729 innerTail99_4 best.bound = innerCandidates99_3 := by
  unfold candidates
  have row : edge 2729 = SearchData.row2729.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan99_3 : scan edge (search edge 3) 4 [557] 557 innerTail99_3 best = best := by
  rw [innerTail99_3, scan, if_neg (by decide), innerCandidates99_3_checked]
  change scan edge (search edge 3) 4 [557] 557 innerTail99_4
    (if 3 ≤ innerCandidates99_3.length then search edge 3 [557, 2729] innerCandidates99_3 3286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan99_4

def innerTail99_2 : List Nat := 2687 :: innerTail99_3
def innerCandidates99_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates99_2_checked : candidates edge 4 557 2687 innerTail99_3 best.bound = innerCandidates99_2 := by
  unfold candidates
  have row : edge 2687 = SearchData.row2687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan99_2 : scan edge (search edge 3) 4 [557] 557 innerTail99_2 best = best := by
  rw [innerTail99_2, scan, if_neg (by decide), innerCandidates99_2_checked]
  change scan edge (search edge 3) 4 [557] 557 innerTail99_3
    (if 3 ≤ innerCandidates99_2.length then search edge 3 [557, 2687] innerCandidates99_2 3244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan99_3

def innerTail99_1 : List Nat := 1721 :: innerTail99_2
def innerCandidates99_1 : List Nat := [8387]
set_option maxRecDepth 4096 in
theorem innerCandidates99_1_checked : candidates edge 4 557 1721 innerTail99_2 best.bound = innerCandidates99_1 := by
  unfold candidates
  have row : edge 1721 = PrimePairCache.smallRow1721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan99_1 : scan edge (search edge 3) 4 [557] 557 innerTail99_1 best = best := by
  rw [innerTail99_1, scan, if_neg (by decide), innerCandidates99_1_checked]
  change scan edge (search edge 3) 4 [557] 557 innerTail99_2
    (if 3 ≤ innerCandidates99_1.length then search edge 3 [557, 1721] innerCandidates99_1 2278 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan99_2

def innerTail99_0 : List Nat := 1487 :: innerTail99_1
def innerCandidates99_0 : List Nat := [2729]
set_option maxRecDepth 4096 in
theorem innerCandidates99_0_checked : candidates edge 4 557 1487 innerTail99_1 best.bound = innerCandidates99_0 := by
  unfold candidates
  have row : edge 1487 = PrimePairCache.smallRow1487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan99_0 : scan edge (search edge 3) 4 [557] 557 innerTail99_0 best = best := by
  rw [innerTail99_0, scan, if_neg (by decide), innerCandidates99_0_checked]
  change scan edge (search edge 3) 4 [557] 557 innerTail99_1
    (if 3 ≤ innerCandidates99_0.length then search edge 3 [557, 1487] innerCandidates99_0 2044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan99_1

theorem rootBranch99 : search edge 4 [557] rootCandidates99 557 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [557] 557 innerTail99_0 best = best
  exact innerScan99_0

end ProjectEulerSolutions.PrimePairUpper.Search
