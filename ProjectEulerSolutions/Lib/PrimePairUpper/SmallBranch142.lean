import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates249 : List Nat := [1811, 1823, 1847, 2063, 2207, 2879, 3533, 4637, 6029, 6113, 6359, 6701, 7229, 7307]
set_option maxRecDepth 4096 in
theorem rootCandidates249_checked : candidates edge 5 0 1601 tail250 best.bound = rootCandidates249 := by
  unfold candidates
  have row : edge 1601 = PrimePairCache.smallRow1601.contains := by funext q; rfl
  rw [row]
  decide

def innerTail249_14 : List Nat := []
theorem innerScan249_14 : scan edge (search edge 3) 4 [1601] 1601 innerTail249_14 best = best := rfl

def innerTail249_13 : List Nat := 7307 :: innerTail249_14
def innerCandidates249_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates249_13_checked : candidates edge 4 1601 7307 innerTail249_14 best.bound = innerCandidates249_13 := by
  unfold candidates
  have row : edge 7307 = SearchData.row7307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan249_13 : scan edge (search edge 3) 4 [1601] 1601 innerTail249_13 best = best := by
  rw [innerTail249_13, scan, if_neg (by decide), innerCandidates249_13_checked]
  change scan edge (search edge 3) 4 [1601] 1601 innerTail249_14
    (if 3 ≤ innerCandidates249_13.length then search edge 3 [1601, 7307] innerCandidates249_13 8908 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan249_14

def innerTail249_12 : List Nat := 7229 :: innerTail249_13
def innerCandidates249_12 : List Nat := [7307]
set_option maxRecDepth 4096 in
theorem innerCandidates249_12_checked : candidates edge 4 1601 7229 innerTail249_13 best.bound = innerCandidates249_12 := by
  unfold candidates
  have row : edge 7229 = SearchData.row7229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan249_12 : scan edge (search edge 3) 4 [1601] 1601 innerTail249_12 best = best := by
  rw [innerTail249_12, scan, if_neg (by decide), innerCandidates249_12_checked]
  change scan edge (search edge 3) 4 [1601] 1601 innerTail249_13
    (if 3 ≤ innerCandidates249_12.length then search edge 3 [1601, 7229] innerCandidates249_12 8830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan249_13

def innerTail249_11 : List Nat := 6701 :: innerTail249_12
def innerCandidates249_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates249_11_checked : candidates edge 4 1601 6701 innerTail249_12 best.bound = innerCandidates249_11 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan249_11 : scan edge (search edge 3) 4 [1601] 1601 innerTail249_11 best = best := by
  rw [innerTail249_11, scan, if_neg (by decide), innerCandidates249_11_checked]
  change scan edge (search edge 3) 4 [1601] 1601 innerTail249_12
    (if 3 ≤ innerCandidates249_11.length then search edge 3 [1601, 6701] innerCandidates249_11 8302 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan249_12

def innerTail249_10 : List Nat := 6359 :: innerTail249_11
def innerCandidates249_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates249_10_checked : candidates edge 4 1601 6359 innerTail249_11 best.bound = innerCandidates249_10 := by
  unfold candidates
  have row : edge 6359 = SearchData.row6359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan249_10 : scan edge (search edge 3) 4 [1601] 1601 innerTail249_10 best = best := by
  rw [innerTail249_10, scan, if_neg (by decide), innerCandidates249_10_checked]
  change scan edge (search edge 3) 4 [1601] 1601 innerTail249_11
    (if 3 ≤ innerCandidates249_10.length then search edge 3 [1601, 6359] innerCandidates249_10 7960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan249_11

def innerTail249_9 : List Nat := 6113 :: innerTail249_10
def innerCandidates249_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates249_9_checked : candidates edge 4 1601 6113 innerTail249_10 best.bound = innerCandidates249_9 := by
  unfold candidates
  have row : edge 6113 = SearchData.row6113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan249_9 : scan edge (search edge 3) 4 [1601] 1601 innerTail249_9 best = best := by
  rw [innerTail249_9, scan, if_neg (by decide), innerCandidates249_9_checked]
  change scan edge (search edge 3) 4 [1601] 1601 innerTail249_10
    (if 3 ≤ innerCandidates249_9.length then search edge 3 [1601, 6113] innerCandidates249_9 7714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan249_10

def innerTail249_8 : List Nat := 6029 :: innerTail249_9
def innerCandidates249_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates249_8_checked : candidates edge 4 1601 6029 innerTail249_9 best.bound = innerCandidates249_8 := by
  unfold candidates
  have row : edge 6029 = SearchData.row6029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan249_8 : scan edge (search edge 3) 4 [1601] 1601 innerTail249_8 best = best := by
  rw [innerTail249_8, scan, if_neg (by decide), innerCandidates249_8_checked]
  change scan edge (search edge 3) 4 [1601] 1601 innerTail249_9
    (if 3 ≤ innerCandidates249_8.length then search edge 3 [1601, 6029] innerCandidates249_8 7630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan249_9

def innerTail249_7 : List Nat := 4637 :: innerTail249_8
def innerCandidates249_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates249_7_checked : candidates edge 4 1601 4637 innerTail249_8 best.bound = innerCandidates249_7 := by
  unfold candidates
  have row : edge 4637 = SearchData.row4637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan249_7 : scan edge (search edge 3) 4 [1601] 1601 innerTail249_7 best = best := by
  rw [innerTail249_7, scan, if_neg (by decide), innerCandidates249_7_checked]
  change scan edge (search edge 3) 4 [1601] 1601 innerTail249_8
    (if 3 ≤ innerCandidates249_7.length then search edge 3 [1601, 4637] innerCandidates249_7 6238 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan249_8

def innerTail249_6 : List Nat := 3533 :: innerTail249_7
def innerCandidates249_6 : List Nat := [6113]
set_option maxRecDepth 4096 in
theorem innerCandidates249_6_checked : candidates edge 4 1601 3533 innerTail249_7 best.bound = innerCandidates249_6 := by
  unfold candidates
  have row : edge 3533 = SearchData.row3533.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan249_6 : scan edge (search edge 3) 4 [1601] 1601 innerTail249_6 best = best := by
  rw [innerTail249_6, scan, if_neg (by decide), innerCandidates249_6_checked]
  change scan edge (search edge 3) 4 [1601] 1601 innerTail249_7
    (if 3 ≤ innerCandidates249_6.length then search edge 3 [1601, 3533] innerCandidates249_6 5134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan249_7

def innerTail249_5 : List Nat := 2879 :: innerTail249_6
def innerCandidates249_5 : List Nat := [6701]
set_option maxRecDepth 4096 in
theorem innerCandidates249_5_checked : candidates edge 4 1601 2879 innerTail249_6 best.bound = innerCandidates249_5 := by
  unfold candidates
  have row : edge 2879 = SearchData.row2879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan249_5 : scan edge (search edge 3) 4 [1601] 1601 innerTail249_5 best = best := by
  rw [innerTail249_5, scan, if_neg (by decide), innerCandidates249_5_checked]
  change scan edge (search edge 3) 4 [1601] 1601 innerTail249_6
    (if 3 ≤ innerCandidates249_5.length then search edge 3 [1601, 2879] innerCandidates249_5 4480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan249_6

def innerTail249_4 : List Nat := 2207 :: innerTail249_5
def innerCandidates249_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates249_4_checked : candidates edge 4 1601 2207 innerTail249_5 best.bound = innerCandidates249_4 := by
  unfold candidates
  have row : edge 2207 = PrimePairCache.smallRow2207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan249_4 : scan edge (search edge 3) 4 [1601] 1601 innerTail249_4 best = best := by
  rw [innerTail249_4, scan, if_neg (by decide), innerCandidates249_4_checked]
  change scan edge (search edge 3) 4 [1601] 1601 innerTail249_5
    (if 3 ≤ innerCandidates249_4.length then search edge 3 [1601, 2207] innerCandidates249_4 3808 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan249_5

def innerTail249_3 : List Nat := 2063 :: innerTail249_4
def innerCandidates249_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates249_3_checked : candidates edge 4 1601 2063 innerTail249_4 best.bound = innerCandidates249_3 := by
  unfold candidates
  have row : edge 2063 = PrimePairCache.smallRow2063.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan249_3 : scan edge (search edge 3) 4 [1601] 1601 innerTail249_3 best = best := by
  rw [innerTail249_3, scan, if_neg (by decide), innerCandidates249_3_checked]
  change scan edge (search edge 3) 4 [1601] 1601 innerTail249_4
    (if 3 ≤ innerCandidates249_3.length then search edge 3 [1601, 2063] innerCandidates249_3 3664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan249_4

def innerTail249_2 : List Nat := 1847 :: innerTail249_3
def innerCandidates249_2 : List Nat := [7307]
set_option maxRecDepth 4096 in
theorem innerCandidates249_2_checked : candidates edge 4 1601 1847 innerTail249_3 best.bound = innerCandidates249_2 := by
  unfold candidates
  have row : edge 1847 = PrimePairCache.smallRow1847.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan249_2 : scan edge (search edge 3) 4 [1601] 1601 innerTail249_2 best = best := by
  rw [innerTail249_2, scan, if_neg (by decide), innerCandidates249_2_checked]
  change scan edge (search edge 3) 4 [1601] 1601 innerTail249_3
    (if 3 ≤ innerCandidates249_2.length then search edge 3 [1601, 1847] innerCandidates249_2 3448 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan249_3

def innerTail249_1 : List Nat := 1823 :: innerTail249_2
def innerCandidates249_1 : List Nat := [2063]
set_option maxRecDepth 4096 in
theorem innerCandidates249_1_checked : candidates edge 4 1601 1823 innerTail249_2 best.bound = innerCandidates249_1 := by
  unfold candidates
  have row : edge 1823 = PrimePairCache.smallRow1823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan249_1 : scan edge (search edge 3) 4 [1601] 1601 innerTail249_1 best = best := by
  rw [innerTail249_1, scan, if_neg (by decide), innerCandidates249_1_checked]
  change scan edge (search edge 3) 4 [1601] 1601 innerTail249_2
    (if 3 ≤ innerCandidates249_1.length then search edge 3 [1601, 1823] innerCandidates249_1 3424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan249_2

def innerTail249_0 : List Nat := 1811 :: innerTail249_1
def innerCandidates249_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates249_0_checked : candidates edge 4 1601 1811 innerTail249_1 best.bound = innerCandidates249_0 := by
  unfold candidates
  have row : edge 1811 = PrimePairCache.smallRow1811.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan249_0 : scan edge (search edge 3) 4 [1601] 1601 innerTail249_0 best = best := by
  rw [innerTail249_0, scan, if_neg (by decide), innerCandidates249_0_checked]
  change scan edge (search edge 3) 4 [1601] 1601 innerTail249_1
    (if 3 ≤ innerCandidates249_0.length then search edge 3 [1601, 1811] innerCandidates249_0 3412 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan249_1

theorem rootBranch249 : search edge 4 [1601] rootCandidates249 1601 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1601] 1601 innerTail249_0 best = best
  exact innerScan249_0

def rootCandidates248 : List Nat := [1777, 1873, 2113, 2251, 2617, 2713, 2731, 3121, 3163, 3301, 3469, 3691, 3823, 4201, 4363, 4657, 5503, 7333, 7573, 7699, 8269]
set_option maxRecDepth 4096 in
theorem rootCandidates248_checked : candidates edge 5 0 1597 tail249 best.bound = rootCandidates248 := by
  unfold candidates
  have row : edge 1597 = PrimePairCache.smallRow1597.contains := by funext q; rfl
  rw [row]
  decide

def innerTail248_21 : List Nat := []
theorem innerScan248_21 : scan edge (search edge 3) 4 [1597] 1597 innerTail248_21 best = best := rfl

def innerTail248_20 : List Nat := 8269 :: innerTail248_21
def innerCandidates248_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates248_20_checked : candidates edge 4 1597 8269 innerTail248_21 best.bound = innerCandidates248_20 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan248_20 : scan edge (search edge 3) 4 [1597] 1597 innerTail248_20 best = best := by
  rw [innerTail248_20, scan, if_neg (by decide), innerCandidates248_20_checked]
  change scan edge (search edge 3) 4 [1597] 1597 innerTail248_21
    (if 3 ≤ innerCandidates248_20.length then search edge 3 [1597, 8269] innerCandidates248_20 9866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan248_21

def innerTail248_19 : List Nat := 7699 :: innerTail248_20
def innerCandidates248_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates248_19_checked : candidates edge 4 1597 7699 innerTail248_20 best.bound = innerCandidates248_19 := by
  unfold candidates
  have row : edge 7699 = SearchData.row7699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan248_19 : scan edge (search edge 3) 4 [1597] 1597 innerTail248_19 best = best := by
  rw [innerTail248_19, scan, if_neg (by decide), innerCandidates248_19_checked]
  change scan edge (search edge 3) 4 [1597] 1597 innerTail248_20
    (if 3 ≤ innerCandidates248_19.length then search edge 3 [1597, 7699] innerCandidates248_19 9296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan248_20

def innerTail248_18 : List Nat := 7573 :: innerTail248_19
def innerCandidates248_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates248_18_checked : candidates edge 4 1597 7573 innerTail248_19 best.bound = innerCandidates248_18 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan248_18 : scan edge (search edge 3) 4 [1597] 1597 innerTail248_18 best = best := by
  rw [innerTail248_18, scan, if_neg (by decide), innerCandidates248_18_checked]
  change scan edge (search edge 3) 4 [1597] 1597 innerTail248_19
    (if 3 ≤ innerCandidates248_18.length then search edge 3 [1597, 7573] innerCandidates248_18 9170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan248_19

def innerTail248_17 : List Nat := 7333 :: innerTail248_18
def innerCandidates248_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates248_17_checked : candidates edge 4 1597 7333 innerTail248_18 best.bound = innerCandidates248_17 := by
  unfold candidates
  have row : edge 7333 = SearchData.row7333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan248_17 : scan edge (search edge 3) 4 [1597] 1597 innerTail248_17 best = best := by
  rw [innerTail248_17, scan, if_neg (by decide), innerCandidates248_17_checked]
  change scan edge (search edge 3) 4 [1597] 1597 innerTail248_18
    (if 3 ≤ innerCandidates248_17.length then search edge 3 [1597, 7333] innerCandidates248_17 8930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan248_18

def innerTail248_16 : List Nat := 5503 :: innerTail248_17
def innerCandidates248_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates248_16_checked : candidates edge 4 1597 5503 innerTail248_17 best.bound = innerCandidates248_16 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan248_16 : scan edge (search edge 3) 4 [1597] 1597 innerTail248_16 best = best := by
  rw [innerTail248_16, scan, if_neg (by decide), innerCandidates248_16_checked]
  change scan edge (search edge 3) 4 [1597] 1597 innerTail248_17
    (if 3 ≤ innerCandidates248_16.length then search edge 3 [1597, 5503] innerCandidates248_16 7100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan248_17

def innerTail248_15 : List Nat := 4657 :: innerTail248_16
def innerCandidates248_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates248_15_checked : candidates edge 4 1597 4657 innerTail248_16 best.bound = innerCandidates248_15 := by
  unfold candidates
  have row : edge 4657 = SearchData.row4657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan248_15 : scan edge (search edge 3) 4 [1597] 1597 innerTail248_15 best = best := by
  rw [innerTail248_15, scan, if_neg (by decide), innerCandidates248_15_checked]
  change scan edge (search edge 3) 4 [1597] 1597 innerTail248_16
    (if 3 ≤ innerCandidates248_15.length then search edge 3 [1597, 4657] innerCandidates248_15 6254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan248_16

def innerTail248_14 : List Nat := 4363 :: innerTail248_15
def innerCandidates248_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates248_14_checked : candidates edge 4 1597 4363 innerTail248_15 best.bound = innerCandidates248_14 := by
  unfold candidates
  have row : edge 4363 = SearchData.row4363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan248_14 : scan edge (search edge 3) 4 [1597] 1597 innerTail248_14 best = best := by
  rw [innerTail248_14, scan, if_neg (by decide), innerCandidates248_14_checked]
  change scan edge (search edge 3) 4 [1597] 1597 innerTail248_15
    (if 3 ≤ innerCandidates248_14.length then search edge 3 [1597, 4363] innerCandidates248_14 5960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan248_15

def innerTail248_13 : List Nat := 4201 :: innerTail248_14
def innerCandidates248_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates248_13_checked : candidates edge 4 1597 4201 innerTail248_14 best.bound = innerCandidates248_13 := by
  unfold candidates
  have row : edge 4201 = SearchData.row4201.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan248_13 : scan edge (search edge 3) 4 [1597] 1597 innerTail248_13 best = best := by
  rw [innerTail248_13, scan, if_neg (by decide), innerCandidates248_13_checked]
  change scan edge (search edge 3) 4 [1597] 1597 innerTail248_14
    (if 3 ≤ innerCandidates248_13.length then search edge 3 [1597, 4201] innerCandidates248_13 5798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan248_14

def innerTail248_12 : List Nat := 3823 :: innerTail248_13
def innerCandidates248_12 : List Nat := [4201]
set_option maxRecDepth 4096 in
theorem innerCandidates248_12_checked : candidates edge 4 1597 3823 innerTail248_13 best.bound = innerCandidates248_12 := by
  unfold candidates
  have row : edge 3823 = SearchData.row3823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan248_12 : scan edge (search edge 3) 4 [1597] 1597 innerTail248_12 best = best := by
  rw [innerTail248_12, scan, if_neg (by decide), innerCandidates248_12_checked]
  change scan edge (search edge 3) 4 [1597] 1597 innerTail248_13
    (if 3 ≤ innerCandidates248_12.length then search edge 3 [1597, 3823] innerCandidates248_12 5420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan248_13

def innerTail248_11 : List Nat := 3691 :: innerTail248_12
def innerCandidates248_11 : List Nat := [7333]
set_option maxRecDepth 4096 in
theorem innerCandidates248_11_checked : candidates edge 4 1597 3691 innerTail248_12 best.bound = innerCandidates248_11 := by
  unfold candidates
  have row : edge 3691 = SearchData.row3691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan248_11 : scan edge (search edge 3) 4 [1597] 1597 innerTail248_11 best = best := by
  rw [innerTail248_11, scan, if_neg (by decide), innerCandidates248_11_checked]
  change scan edge (search edge 3) 4 [1597] 1597 innerTail248_12
    (if 3 ≤ innerCandidates248_11.length then search edge 3 [1597, 3691] innerCandidates248_11 5288 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan248_12

def innerTail248_10 : List Nat := 3469 :: innerTail248_11
def innerCandidates248_10 : List Nat := [5503]
set_option maxRecDepth 4096 in
theorem innerCandidates248_10_checked : candidates edge 4 1597 3469 innerTail248_11 best.bound = innerCandidates248_10 := by
  unfold candidates
  have row : edge 3469 = SearchData.row3469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan248_10 : scan edge (search edge 3) 4 [1597] 1597 innerTail248_10 best = best := by
  rw [innerTail248_10, scan, if_neg (by decide), innerCandidates248_10_checked]
  change scan edge (search edge 3) 4 [1597] 1597 innerTail248_11
    (if 3 ≤ innerCandidates248_10.length then search edge 3 [1597, 3469] innerCandidates248_10 5066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan248_11

def innerTail248_9 : List Nat := 3301 :: innerTail248_10
def innerCandidates248_9 : List Nat := [4363]
set_option maxRecDepth 4096 in
theorem innerCandidates248_9_checked : candidates edge 4 1597 3301 innerTail248_10 best.bound = innerCandidates248_9 := by
  unfold candidates
  have row : edge 3301 = SearchData.row3301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan248_9 : scan edge (search edge 3) 4 [1597] 1597 innerTail248_9 best = best := by
  rw [innerTail248_9, scan, if_neg (by decide), innerCandidates248_9_checked]
  change scan edge (search edge 3) 4 [1597] 1597 innerTail248_10
    (if 3 ≤ innerCandidates248_9.length then search edge 3 [1597, 3301] innerCandidates248_9 4898 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan248_10

def innerTail248_8 : List Nat := 3163 :: innerTail248_9
def innerCandidates248_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates248_8_checked : candidates edge 4 1597 3163 innerTail248_9 best.bound = innerCandidates248_8 := by
  unfold candidates
  have row : edge 3163 = SearchData.row3163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan248_8 : scan edge (search edge 3) 4 [1597] 1597 innerTail248_8 best = best := by
  rw [innerTail248_8, scan, if_neg (by decide), innerCandidates248_8_checked]
  change scan edge (search edge 3) 4 [1597] 1597 innerTail248_9
    (if 3 ≤ innerCandidates248_8.length then search edge 3 [1597, 3163] innerCandidates248_8 4760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan248_9

def innerTail248_7 : List Nat := 3121 :: innerTail248_8
def innerCandidates248_7 : List Nat := [4201]
set_option maxRecDepth 4096 in
theorem innerCandidates248_7_checked : candidates edge 4 1597 3121 innerTail248_8 best.bound = innerCandidates248_7 := by
  unfold candidates
  have row : edge 3121 = SearchData.row3121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan248_7 : scan edge (search edge 3) 4 [1597] 1597 innerTail248_7 best = best := by
  rw [innerTail248_7, scan, if_neg (by decide), innerCandidates248_7_checked]
  change scan edge (search edge 3) 4 [1597] 1597 innerTail248_8
    (if 3 ≤ innerCandidates248_7.length then search edge 3 [1597, 3121] innerCandidates248_7 4718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan248_8

def innerTail248_6 : List Nat := 2731 :: innerTail248_7
def innerCandidates248_6 : List Nat := [3301, 4657]
set_option maxRecDepth 4096 in
theorem innerCandidates248_6_checked : candidates edge 4 1597 2731 innerTail248_7 best.bound = innerCandidates248_6 := by
  unfold candidates
  have row : edge 2731 = SearchData.row2731.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan248_6 : scan edge (search edge 3) 4 [1597] 1597 innerTail248_6 best = best := by
  rw [innerTail248_6, scan, if_neg (by decide), innerCandidates248_6_checked]
  change scan edge (search edge 3) 4 [1597] 1597 innerTail248_7
    (if 3 ≤ innerCandidates248_6.length then search edge 3 [1597, 2731] innerCandidates248_6 4328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan248_7

def innerTail248_5 : List Nat := 2713 :: innerTail248_6
def innerCandidates248_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates248_5_checked : candidates edge 4 1597 2713 innerTail248_6 best.bound = innerCandidates248_5 := by
  unfold candidates
  have row : edge 2713 = SearchData.row2713.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan248_5 : scan edge (search edge 3) 4 [1597] 1597 innerTail248_5 best = best := by
  rw [innerTail248_5, scan, if_neg (by decide), innerCandidates248_5_checked]
  change scan edge (search edge 3) 4 [1597] 1597 innerTail248_6
    (if 3 ≤ innerCandidates248_5.length then search edge 3 [1597, 2713] innerCandidates248_5 4310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan248_6

def innerTail248_4 : List Nat := 2617 :: innerTail248_5
def innerCandidates248_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates248_4_checked : candidates edge 4 1597 2617 innerTail248_5 best.bound = innerCandidates248_4 := by
  unfold candidates
  have row : edge 2617 = PrimePairCache.smallRow2617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan248_4 : scan edge (search edge 3) 4 [1597] 1597 innerTail248_4 best = best := by
  rw [innerTail248_4, scan, if_neg (by decide), innerCandidates248_4_checked]
  change scan edge (search edge 3) 4 [1597] 1597 innerTail248_5
    (if 3 ≤ innerCandidates248_4.length then search edge 3 [1597, 2617] innerCandidates248_4 4214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan248_5

def innerTail248_3 : List Nat := 2251 :: innerTail248_4
def innerCandidates248_3 : List Nat := [4201, 5503]
set_option maxRecDepth 4096 in
theorem innerCandidates248_3_checked : candidates edge 4 1597 2251 innerTail248_4 best.bound = innerCandidates248_3 := by
  unfold candidates
  have row : edge 2251 = PrimePairCache.smallRow2251.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan248_3 : scan edge (search edge 3) 4 [1597] 1597 innerTail248_3 best = best := by
  rw [innerTail248_3, scan, if_neg (by decide), innerCandidates248_3_checked]
  change scan edge (search edge 3) 4 [1597] 1597 innerTail248_4
    (if 3 ≤ innerCandidates248_3.length then search edge 3 [1597, 2251] innerCandidates248_3 3848 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan248_4

def innerTail248_2 : List Nat := 2113 :: innerTail248_3
def innerCandidates248_2 : List Nat := [3121]
set_option maxRecDepth 4096 in
theorem innerCandidates248_2_checked : candidates edge 4 1597 2113 innerTail248_3 best.bound = innerCandidates248_2 := by
  unfold candidates
  have row : edge 2113 = PrimePairCache.smallRow2113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan248_2 : scan edge (search edge 3) 4 [1597] 1597 innerTail248_2 best = best := by
  rw [innerTail248_2, scan, if_neg (by decide), innerCandidates248_2_checked]
  change scan edge (search edge 3) 4 [1597] 1597 innerTail248_3
    (if 3 ≤ innerCandidates248_2.length then search edge 3 [1597, 2113] innerCandidates248_2 3710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan248_3

def innerTail248_1 : List Nat := 1873 :: innerTail248_2
def innerCandidates248_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates248_1_checked : candidates edge 4 1597 1873 innerTail248_2 best.bound = innerCandidates248_1 := by
  unfold candidates
  have row : edge 1873 = PrimePairCache.smallRow1873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan248_1 : scan edge (search edge 3) 4 [1597] 1597 innerTail248_1 best = best := by
  rw [innerTail248_1, scan, if_neg (by decide), innerCandidates248_1_checked]
  change scan edge (search edge 3) 4 [1597] 1597 innerTail248_2
    (if 3 ≤ innerCandidates248_1.length then search edge 3 [1597, 1873] innerCandidates248_1 3470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan248_2

def innerTail248_0 : List Nat := 1777 :: innerTail248_1
def innerCandidates248_0 : List Nat := [3121]
set_option maxRecDepth 4096 in
theorem innerCandidates248_0_checked : candidates edge 4 1597 1777 innerTail248_1 best.bound = innerCandidates248_0 := by
  unfold candidates
  have row : edge 1777 = PrimePairCache.smallRow1777.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan248_0 : scan edge (search edge 3) 4 [1597] 1597 innerTail248_0 best = best := by
  rw [innerTail248_0, scan, if_neg (by decide), innerCandidates248_0_checked]
  change scan edge (search edge 3) 4 [1597] 1597 innerTail248_1
    (if 3 ≤ innerCandidates248_0.length then search edge 3 [1597, 1777] innerCandidates248_0 3374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan248_1

theorem rootBranch248 : search edge 4 [1597] rootCandidates248 1597 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1597] 1597 innerTail248_0 best = best
  exact innerScan248_0

end ProjectEulerSolutions.PrimePairUpper.Search
