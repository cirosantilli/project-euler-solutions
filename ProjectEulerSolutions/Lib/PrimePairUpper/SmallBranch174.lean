import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates161 : List Nat := [977, 1019, 1511, 2111, 2447, 2543, 2699, 2897, 3863, 3947, 4421, 5009, 5309, 5477, 5501, 6269, 6491, 6569, 6689, 6701, 7193, 7517, 7919, 8039, 8081, 8123, 8171]
set_option maxRecDepth 4096 in
theorem rootCandidates161_checked : candidates edge 5 0 971 tail162 best.bound = rootCandidates161 := by
  unfold candidates
  have row : edge 971 = PrimePairCache.smallRow971.contains := by funext q; rfl
  rw [row]
  decide

def innerTail161_27 : List Nat := []
theorem innerScan161_27 : scan edge (search edge 3) 4 [971] 971 innerTail161_27 best = best := rfl

def innerTail161_26 : List Nat := 8171 :: innerTail161_27
def innerCandidates161_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates161_26_checked : candidates edge 4 971 8171 innerTail161_27 best.bound = innerCandidates161_26 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_26 : scan edge (search edge 3) 4 [971] 971 innerTail161_26 best = best := by
  rw [innerTail161_26, scan, if_neg (by decide), innerCandidates161_26_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_27
    (if 3 ≤ innerCandidates161_26.length then search edge 3 [971, 8171] innerCandidates161_26 9142 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_27

def innerTail161_25 : List Nat := 8123 :: innerTail161_26
def innerCandidates161_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates161_25_checked : candidates edge 4 971 8123 innerTail161_26 best.bound = innerCandidates161_25 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_25 : scan edge (search edge 3) 4 [971] 971 innerTail161_25 best = best := by
  rw [innerTail161_25, scan, if_neg (by decide), innerCandidates161_25_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_26
    (if 3 ≤ innerCandidates161_25.length then search edge 3 [971, 8123] innerCandidates161_25 9094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_26

def innerTail161_24 : List Nat := 8081 :: innerTail161_25
def innerCandidates161_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates161_24_checked : candidates edge 4 971 8081 innerTail161_25 best.bound = innerCandidates161_24 := by
  unfold candidates
  have row : edge 8081 = SearchData.row8081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_24 : scan edge (search edge 3) 4 [971] 971 innerTail161_24 best = best := by
  rw [innerTail161_24, scan, if_neg (by decide), innerCandidates161_24_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_25
    (if 3 ≤ innerCandidates161_24.length then search edge 3 [971, 8081] innerCandidates161_24 9052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_25

def innerTail161_23 : List Nat := 8039 :: innerTail161_24
def innerCandidates161_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates161_23_checked : candidates edge 4 971 8039 innerTail161_24 best.bound = innerCandidates161_23 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_23 : scan edge (search edge 3) 4 [971] 971 innerTail161_23 best = best := by
  rw [innerTail161_23, scan, if_neg (by decide), innerCandidates161_23_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_24
    (if 3 ≤ innerCandidates161_23.length then search edge 3 [971, 8039] innerCandidates161_23 9010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_24

def innerTail161_22 : List Nat := 7919 :: innerTail161_23
def innerCandidates161_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates161_22_checked : candidates edge 4 971 7919 innerTail161_23 best.bound = innerCandidates161_22 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_22 : scan edge (search edge 3) 4 [971] 971 innerTail161_22 best = best := by
  rw [innerTail161_22, scan, if_neg (by decide), innerCandidates161_22_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_23
    (if 3 ≤ innerCandidates161_22.length then search edge 3 [971, 7919] innerCandidates161_22 8890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_23

def innerTail161_21 : List Nat := 7517 :: innerTail161_22
def innerCandidates161_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates161_21_checked : candidates edge 4 971 7517 innerTail161_22 best.bound = innerCandidates161_21 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_21 : scan edge (search edge 3) 4 [971] 971 innerTail161_21 best = best := by
  rw [innerTail161_21, scan, if_neg (by decide), innerCandidates161_21_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_22
    (if 3 ≤ innerCandidates161_21.length then search edge 3 [971, 7517] innerCandidates161_21 8488 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_22

def innerTail161_20 : List Nat := 7193 :: innerTail161_21
def innerCandidates161_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates161_20_checked : candidates edge 4 971 7193 innerTail161_21 best.bound = innerCandidates161_20 := by
  unfold candidates
  have row : edge 7193 = SearchData.row7193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_20 : scan edge (search edge 3) 4 [971] 971 innerTail161_20 best = best := by
  rw [innerTail161_20, scan, if_neg (by decide), innerCandidates161_20_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_21
    (if 3 ≤ innerCandidates161_20.length then search edge 3 [971, 7193] innerCandidates161_20 8164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_21

def innerTail161_19 : List Nat := 6701 :: innerTail161_20
def innerCandidates161_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates161_19_checked : candidates edge 4 971 6701 innerTail161_20 best.bound = innerCandidates161_19 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_19 : scan edge (search edge 3) 4 [971] 971 innerTail161_19 best = best := by
  rw [innerTail161_19, scan, if_neg (by decide), innerCandidates161_19_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_20
    (if 3 ≤ innerCandidates161_19.length then search edge 3 [971, 6701] innerCandidates161_19 7672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_20

def innerTail161_18 : List Nat := 6689 :: innerTail161_19
def innerCandidates161_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates161_18_checked : candidates edge 4 971 6689 innerTail161_19 best.bound = innerCandidates161_18 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_18 : scan edge (search edge 3) 4 [971] 971 innerTail161_18 best = best := by
  rw [innerTail161_18, scan, if_neg (by decide), innerCandidates161_18_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_19
    (if 3 ≤ innerCandidates161_18.length then search edge 3 [971, 6689] innerCandidates161_18 7660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_19

def innerTail161_17 : List Nat := 6569 :: innerTail161_18
def innerCandidates161_17 : List Nat := [6689]
set_option maxRecDepth 4096 in
theorem innerCandidates161_17_checked : candidates edge 4 971 6569 innerTail161_18 best.bound = innerCandidates161_17 := by
  unfold candidates
  have row : edge 6569 = SearchData.row6569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_17 : scan edge (search edge 3) 4 [971] 971 innerTail161_17 best = best := by
  rw [innerTail161_17, scan, if_neg (by decide), innerCandidates161_17_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_18
    (if 3 ≤ innerCandidates161_17.length then search edge 3 [971, 6569] innerCandidates161_17 7540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_18

def innerTail161_16 : List Nat := 6491 :: innerTail161_17
def innerCandidates161_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates161_16_checked : candidates edge 4 971 6491 innerTail161_17 best.bound = innerCandidates161_16 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_16 : scan edge (search edge 3) 4 [971] 971 innerTail161_16 best = best := by
  rw [innerTail161_16, scan, if_neg (by decide), innerCandidates161_16_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_17
    (if 3 ≤ innerCandidates161_16.length then search edge 3 [971, 6491] innerCandidates161_16 7462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_17

def innerTail161_15 : List Nat := 6269 :: innerTail161_16
def innerCandidates161_15 : List Nat := [6701]
set_option maxRecDepth 4096 in
theorem innerCandidates161_15_checked : candidates edge 4 971 6269 innerTail161_16 best.bound = innerCandidates161_15 := by
  unfold candidates
  have row : edge 6269 = SearchData.row6269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_15 : scan edge (search edge 3) 4 [971] 971 innerTail161_15 best = best := by
  rw [innerTail161_15, scan, if_neg (by decide), innerCandidates161_15_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_16
    (if 3 ≤ innerCandidates161_15.length then search edge 3 [971, 6269] innerCandidates161_15 7240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_16

def innerTail161_14 : List Nat := 5501 :: innerTail161_15
def innerCandidates161_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates161_14_checked : candidates edge 4 971 5501 innerTail161_15 best.bound = innerCandidates161_14 := by
  unfold candidates
  have row : edge 5501 = SearchData.row5501.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_14 : scan edge (search edge 3) 4 [971] 971 innerTail161_14 best = best := by
  rw [innerTail161_14, scan, if_neg (by decide), innerCandidates161_14_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_15
    (if 3 ≤ innerCandidates161_14.length then search edge 3 [971, 5501] innerCandidates161_14 6472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_15

def innerTail161_13 : List Nat := 5477 :: innerTail161_14
def innerCandidates161_13 : List Nat := [8039, 8081, 8123]
set_option maxRecDepth 4096 in
theorem innerCandidates161_13_checked : candidates edge 4 971 5477 innerTail161_14 best.bound = innerCandidates161_13 := by
  unfold candidates
  have row : edge 5477 = SearchData.row5477.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild161_13 : search edge 3 [971, 5477] innerCandidates161_13 6448 best = best := by decide

theorem innerScan161_13 : scan edge (search edge 3) 4 [971] 971 innerTail161_13 best = best := by
  rw [innerTail161_13, scan, if_neg (by decide), innerCandidates161_13_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_14
    (if 3 ≤ innerCandidates161_13.length then search edge 3 [971, 5477] innerCandidates161_13 6448 best else best) = best
  rw [if_pos (by decide), innerChild161_13]
  exact innerScan161_14

def innerTail161_12 : List Nat := 5309 :: innerTail161_13
def innerCandidates161_12 : List Nat := [5501]
set_option maxRecDepth 4096 in
theorem innerCandidates161_12_checked : candidates edge 4 971 5309 innerTail161_13 best.bound = innerCandidates161_12 := by
  unfold candidates
  have row : edge 5309 = SearchData.row5309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_12 : scan edge (search edge 3) 4 [971] 971 innerTail161_12 best = best := by
  rw [innerTail161_12, scan, if_neg (by decide), innerCandidates161_12_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_13
    (if 3 ≤ innerCandidates161_12.length then search edge 3 [971, 5309] innerCandidates161_12 6280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_13

def innerTail161_11 : List Nat := 5009 :: innerTail161_12
def innerCandidates161_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates161_11_checked : candidates edge 4 971 5009 innerTail161_12 best.bound = innerCandidates161_11 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_11 : scan edge (search edge 3) 4 [971] 971 innerTail161_11 best = best := by
  rw [innerTail161_11, scan, if_neg (by decide), innerCandidates161_11_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_12
    (if 3 ≤ innerCandidates161_11.length then search edge 3 [971, 5009] innerCandidates161_11 5980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_12

def innerTail161_10 : List Nat := 4421 :: innerTail161_11
def innerCandidates161_10 : List Nat := [8039]
set_option maxRecDepth 4096 in
theorem innerCandidates161_10_checked : candidates edge 4 971 4421 innerTail161_11 best.bound = innerCandidates161_10 := by
  unfold candidates
  have row : edge 4421 = SearchData.row4421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_10 : scan edge (search edge 3) 4 [971] 971 innerTail161_10 best = best := by
  rw [innerTail161_10, scan, if_neg (by decide), innerCandidates161_10_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_11
    (if 3 ≤ innerCandidates161_10.length then search edge 3 [971, 4421] innerCandidates161_10 5392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_11

def innerTail161_9 : List Nat := 3947 :: innerTail161_10
def innerCandidates161_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates161_9_checked : candidates edge 4 971 3947 innerTail161_10 best.bound = innerCandidates161_9 := by
  unfold candidates
  have row : edge 3947 = SearchData.row3947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_9 : scan edge (search edge 3) 4 [971] 971 innerTail161_9 best = best := by
  rw [innerTail161_9, scan, if_neg (by decide), innerCandidates161_9_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_10
    (if 3 ≤ innerCandidates161_9.length then search edge 3 [971, 3947] innerCandidates161_9 4918 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_10

def innerTail161_8 : List Nat := 3863 :: innerTail161_9
def innerCandidates161_8 : List Nat := [6569]
set_option maxRecDepth 4096 in
theorem innerCandidates161_8_checked : candidates edge 4 971 3863 innerTail161_9 best.bound = innerCandidates161_8 := by
  unfold candidates
  have row : edge 3863 = SearchData.row3863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_8 : scan edge (search edge 3) 4 [971] 971 innerTail161_8 best = best := by
  rw [innerTail161_8, scan, if_neg (by decide), innerCandidates161_8_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_9
    (if 3 ≤ innerCandidates161_8.length then search edge 3 [971, 3863] innerCandidates161_8 4834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_9

def innerTail161_7 : List Nat := 2897 :: innerTail161_8
def innerCandidates161_7 : List Nat := [3947, 5501]
set_option maxRecDepth 4096 in
theorem innerCandidates161_7_checked : candidates edge 4 971 2897 innerTail161_8 best.bound = innerCandidates161_7 := by
  unfold candidates
  have row : edge 2897 = SearchData.row2897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_7 : scan edge (search edge 3) 4 [971] 971 innerTail161_7 best = best := by
  rw [innerTail161_7, scan, if_neg (by decide), innerCandidates161_7_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_8
    (if 3 ≤ innerCandidates161_7.length then search edge 3 [971, 2897] innerCandidates161_7 3868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_8

def innerTail161_6 : List Nat := 2699 :: innerTail161_7
def innerCandidates161_6 : List Nat := [8039]
set_option maxRecDepth 4096 in
theorem innerCandidates161_6_checked : candidates edge 4 971 2699 innerTail161_7 best.bound = innerCandidates161_6 := by
  unfold candidates
  have row : edge 2699 = SearchData.row2699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_6 : scan edge (search edge 3) 4 [971] 971 innerTail161_6 best = best := by
  rw [innerTail161_6, scan, if_neg (by decide), innerCandidates161_6_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_7
    (if 3 ≤ innerCandidates161_6.length then search edge 3 [971, 2699] innerCandidates161_6 3670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_7

def innerTail161_5 : List Nat := 2543 :: innerTail161_6
def innerCandidates161_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates161_5_checked : candidates edge 4 971 2543 innerTail161_6 best.bound = innerCandidates161_5 := by
  unfold candidates
  have row : edge 2543 = PrimePairCache.smallRow2543.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_5 : scan edge (search edge 3) 4 [971] 971 innerTail161_5 best = best := by
  rw [innerTail161_5, scan, if_neg (by decide), innerCandidates161_5_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_6
    (if 3 ≤ innerCandidates161_5.length then search edge 3 [971, 2543] innerCandidates161_5 3514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_6

def innerTail161_4 : List Nat := 2447 :: innerTail161_5
def innerCandidates161_4 : List Nat := [8123]
set_option maxRecDepth 4096 in
theorem innerCandidates161_4_checked : candidates edge 4 971 2447 innerTail161_5 best.bound = innerCandidates161_4 := by
  unfold candidates
  have row : edge 2447 = PrimePairCache.smallRow2447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_4 : scan edge (search edge 3) 4 [971] 971 innerTail161_4 best = best := by
  rw [innerTail161_4, scan, if_neg (by decide), innerCandidates161_4_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_5
    (if 3 ≤ innerCandidates161_4.length then search edge 3 [971, 2447] innerCandidates161_4 3418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_5

def innerTail161_3 : List Nat := 2111 :: innerTail161_4
def innerCandidates161_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates161_3_checked : candidates edge 4 971 2111 innerTail161_4 best.bound = innerCandidates161_3 := by
  unfold candidates
  have row : edge 2111 = PrimePairCache.smallRow2111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_3 : scan edge (search edge 3) 4 [971] 971 innerTail161_3 best = best := by
  rw [innerTail161_3, scan, if_neg (by decide), innerCandidates161_3_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_4
    (if 3 ≤ innerCandidates161_3.length then search edge 3 [971, 2111] innerCandidates161_3 3082 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_4

def innerTail161_2 : List Nat := 1511 :: innerTail161_3
def innerCandidates161_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates161_2_checked : candidates edge 4 971 1511 innerTail161_3 best.bound = innerCandidates161_2 := by
  unfold candidates
  have row : edge 1511 = PrimePairCache.smallRow1511.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_2 : scan edge (search edge 3) 4 [971] 971 innerTail161_2 best = best := by
  rw [innerTail161_2, scan, if_neg (by decide), innerCandidates161_2_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_3
    (if 3 ≤ innerCandidates161_2.length then search edge 3 [971, 1511] innerCandidates161_2 2482 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_3

def innerTail161_1 : List Nat := 1019 :: innerTail161_2
def innerCandidates161_1 : List Nat := [5477, 7193]
set_option maxRecDepth 4096 in
theorem innerCandidates161_1_checked : candidates edge 4 971 1019 innerTail161_2 best.bound = innerCandidates161_1 := by
  unfold candidates
  have row : edge 1019 = PrimePairCache.smallRow1019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_1 : scan edge (search edge 3) 4 [971] 971 innerTail161_1 best = best := by
  rw [innerTail161_1, scan, if_neg (by decide), innerCandidates161_1_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_2
    (if 3 ≤ innerCandidates161_1.length then search edge 3 [971, 1019] innerCandidates161_1 1990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_2

def innerTail161_0 : List Nat := 977 :: innerTail161_1
def innerCandidates161_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates161_0_checked : candidates edge 4 971 977 innerTail161_1 best.bound = innerCandidates161_0 := by
  unfold candidates
  have row : edge 977 = PrimePairCache.smallRow977.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan161_0 : scan edge (search edge 3) 4 [971] 971 innerTail161_0 best = best := by
  rw [innerTail161_0, scan, if_neg (by decide), innerCandidates161_0_checked]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_1
    (if 3 ≤ innerCandidates161_0.length then search edge 3 [971, 977] innerCandidates161_0 1948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan161_1

theorem rootBranch161 : search edge 4 [971] rootCandidates161 971 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [971] 971 innerTail161_0 best = best
  exact innerScan161_0

def rootCandidates160 : List Nat := [1117, 1381, 1663, 1951, 2389, 2557, 2791, 3049, 3511, 3583, 4507, 5011, 5623, 5779, 5857, 6421, 6607, 6829, 6949, 6967, 8017, 8269]
set_option maxRecDepth 4096 in
theorem rootCandidates160_checked : candidates edge 5 0 967 tail161 best.bound = rootCandidates160 := by
  unfold candidates
  have row : edge 967 = PrimePairCache.smallRow967.contains := by funext q; rfl
  rw [row]
  decide

def innerTail160_22 : List Nat := []
theorem innerScan160_22 : scan edge (search edge 3) 4 [967] 967 innerTail160_22 best = best := rfl

def innerTail160_21 : List Nat := 8269 :: innerTail160_22
def innerCandidates160_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates160_21_checked : candidates edge 4 967 8269 innerTail160_22 best.bound = innerCandidates160_21 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan160_21 : scan edge (search edge 3) 4 [967] 967 innerTail160_21 best = best := by
  rw [innerTail160_21, scan, if_neg (by decide), innerCandidates160_21_checked]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_22
    (if 3 ≤ innerCandidates160_21.length then search edge 3 [967, 8269] innerCandidates160_21 9236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan160_22

def innerTail160_20 : List Nat := 8017 :: innerTail160_21
def innerCandidates160_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates160_20_checked : candidates edge 4 967 8017 innerTail160_21 best.bound = innerCandidates160_20 := by
  unfold candidates
  have row : edge 8017 = SearchData.row8017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan160_20 : scan edge (search edge 3) 4 [967] 967 innerTail160_20 best = best := by
  rw [innerTail160_20, scan, if_neg (by decide), innerCandidates160_20_checked]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_21
    (if 3 ≤ innerCandidates160_20.length then search edge 3 [967, 8017] innerCandidates160_20 8984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan160_21

def innerTail160_19 : List Nat := 6967 :: innerTail160_20
def innerCandidates160_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates160_19_checked : candidates edge 4 967 6967 innerTail160_20 best.bound = innerCandidates160_19 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan160_19 : scan edge (search edge 3) 4 [967] 967 innerTail160_19 best = best := by
  rw [innerTail160_19, scan, if_neg (by decide), innerCandidates160_19_checked]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_20
    (if 3 ≤ innerCandidates160_19.length then search edge 3 [967, 6967] innerCandidates160_19 7934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan160_20

def innerTail160_18 : List Nat := 6949 :: innerTail160_19
def innerCandidates160_18 : List Nat := [6967]
set_option maxRecDepth 4096 in
theorem innerCandidates160_18_checked : candidates edge 4 967 6949 innerTail160_19 best.bound = innerCandidates160_18 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan160_18 : scan edge (search edge 3) 4 [967] 967 innerTail160_18 best = best := by
  rw [innerTail160_18, scan, if_neg (by decide), innerCandidates160_18_checked]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_19
    (if 3 ≤ innerCandidates160_18.length then search edge 3 [967, 6949] innerCandidates160_18 7916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan160_19

def innerTail160_17 : List Nat := 6829 :: innerTail160_18
def innerCandidates160_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates160_17_checked : candidates edge 4 967 6829 innerTail160_18 best.bound = innerCandidates160_17 := by
  unfold candidates
  have row : edge 6829 = SearchData.row6829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan160_17 : scan edge (search edge 3) 4 [967] 967 innerTail160_17 best = best := by
  rw [innerTail160_17, scan, if_neg (by decide), innerCandidates160_17_checked]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_18
    (if 3 ≤ innerCandidates160_17.length then search edge 3 [967, 6829] innerCandidates160_17 7796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan160_18

def innerTail160_16 : List Nat := 6607 :: innerTail160_17
def innerCandidates160_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates160_16_checked : candidates edge 4 967 6607 innerTail160_17 best.bound = innerCandidates160_16 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan160_16 : scan edge (search edge 3) 4 [967] 967 innerTail160_16 best = best := by
  rw [innerTail160_16, scan, if_neg (by decide), innerCandidates160_16_checked]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_17
    (if 3 ≤ innerCandidates160_16.length then search edge 3 [967, 6607] innerCandidates160_16 7574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan160_17

def innerTail160_15 : List Nat := 6421 :: innerTail160_16
def innerCandidates160_15 : List Nat := [6949]
set_option maxRecDepth 4096 in
theorem innerCandidates160_15_checked : candidates edge 4 967 6421 innerTail160_16 best.bound = innerCandidates160_15 := by
  unfold candidates
  have row : edge 6421 = SearchData.row6421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan160_15 : scan edge (search edge 3) 4 [967] 967 innerTail160_15 best = best := by
  rw [innerTail160_15, scan, if_neg (by decide), innerCandidates160_15_checked]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_16
    (if 3 ≤ innerCandidates160_15.length then search edge 3 [967, 6421] innerCandidates160_15 7388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan160_16

def innerTail160_14 : List Nat := 5857 :: innerTail160_15
def innerCandidates160_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates160_14_checked : candidates edge 4 967 5857 innerTail160_15 best.bound = innerCandidates160_14 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan160_14 : scan edge (search edge 3) 4 [967] 967 innerTail160_14 best = best := by
  rw [innerTail160_14, scan, if_neg (by decide), innerCandidates160_14_checked]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_15
    (if 3 ≤ innerCandidates160_14.length then search edge 3 [967, 5857] innerCandidates160_14 6824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan160_15

def innerTail160_13 : List Nat := 5779 :: innerTail160_14
def innerCandidates160_13 : List Nat := [6421, 8269]
set_option maxRecDepth 4096 in
theorem innerCandidates160_13_checked : candidates edge 4 967 5779 innerTail160_14 best.bound = innerCandidates160_13 := by
  unfold candidates
  have row : edge 5779 = SearchData.row5779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan160_13 : scan edge (search edge 3) 4 [967] 967 innerTail160_13 best = best := by
  rw [innerTail160_13, scan, if_neg (by decide), innerCandidates160_13_checked]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_14
    (if 3 ≤ innerCandidates160_13.length then search edge 3 [967, 5779] innerCandidates160_13 6746 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan160_14

def innerTail160_12 : List Nat := 5623 :: innerTail160_13
def innerCandidates160_12 : List Nat := [6421, 6949]
set_option maxRecDepth 4096 in
theorem innerCandidates160_12_checked : candidates edge 4 967 5623 innerTail160_13 best.bound = innerCandidates160_12 := by
  unfold candidates
  have row : edge 5623 = SearchData.row5623.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan160_12 : scan edge (search edge 3) 4 [967] 967 innerTail160_12 best = best := by
  rw [innerTail160_12, scan, if_neg (by decide), innerCandidates160_12_checked]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_13
    (if 3 ≤ innerCandidates160_12.length then search edge 3 [967, 5623] innerCandidates160_12 6590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan160_13

def innerTail160_11 : List Nat := 5011 :: innerTail160_12
def innerCandidates160_11 : List Nat := [6949]
set_option maxRecDepth 4096 in
theorem innerCandidates160_11_checked : candidates edge 4 967 5011 innerTail160_12 best.bound = innerCandidates160_11 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan160_11 : scan edge (search edge 3) 4 [967] 967 innerTail160_11 best = best := by
  rw [innerTail160_11, scan, if_neg (by decide), innerCandidates160_11_checked]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_12
    (if 3 ≤ innerCandidates160_11.length then search edge 3 [967, 5011] innerCandidates160_11 5978 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan160_12

def innerTail160_10 : List Nat := 4507 :: innerTail160_11
def innerCandidates160_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates160_10_checked : candidates edge 4 967 4507 innerTail160_11 best.bound = innerCandidates160_10 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan160_10 : scan edge (search edge 3) 4 [967] 967 innerTail160_10 best = best := by
  rw [innerTail160_10, scan, if_neg (by decide), innerCandidates160_10_checked]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_11
    (if 3 ≤ innerCandidates160_10.length then search edge 3 [967, 4507] innerCandidates160_10 5474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan160_11

def innerTail160_9 : List Nat := 3583 :: innerTail160_10
def innerCandidates160_9 : List Nat := [5857]
set_option maxRecDepth 4096 in
theorem innerCandidates160_9_checked : candidates edge 4 967 3583 innerTail160_10 best.bound = innerCandidates160_9 := by
  unfold candidates
  have row : edge 3583 = SearchData.row3583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan160_9 : scan edge (search edge 3) 4 [967] 967 innerTail160_9 best = best := by
  rw [innerTail160_9, scan, if_neg (by decide), innerCandidates160_9_checked]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_10
    (if 3 ≤ innerCandidates160_9.length then search edge 3 [967, 3583] innerCandidates160_9 4550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan160_10

def innerTail160_8 : List Nat := 3511 :: innerTail160_9
def innerCandidates160_8 : List Nat := [6967]
set_option maxRecDepth 4096 in
theorem innerCandidates160_8_checked : candidates edge 4 967 3511 innerTail160_9 best.bound = innerCandidates160_8 := by
  unfold candidates
  have row : edge 3511 = SearchData.row3511.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan160_8 : scan edge (search edge 3) 4 [967] 967 innerTail160_8 best = best := by
  rw [innerTail160_8, scan, if_neg (by decide), innerCandidates160_8_checked]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_9
    (if 3 ≤ innerCandidates160_8.length then search edge 3 [967, 3511] innerCandidates160_8 4478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan160_9

def innerTail160_7 : List Nat := 3049 :: innerTail160_8
def innerCandidates160_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates160_7_checked : candidates edge 4 967 3049 innerTail160_8 best.bound = innerCandidates160_7 := by
  unfold candidates
  have row : edge 3049 = SearchData.row3049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan160_7 : scan edge (search edge 3) 4 [967] 967 innerTail160_7 best = best := by
  rw [innerTail160_7, scan, if_neg (by decide), innerCandidates160_7_checked]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_8
    (if 3 ≤ innerCandidates160_7.length then search edge 3 [967, 3049] innerCandidates160_7 4016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan160_8

def innerTail160_6 : List Nat := 2791 :: innerTail160_7
def innerCandidates160_6 : List Nat := [6607]
set_option maxRecDepth 4096 in
theorem innerCandidates160_6_checked : candidates edge 4 967 2791 innerTail160_7 best.bound = innerCandidates160_6 := by
  unfold candidates
  have row : edge 2791 = SearchData.row2791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan160_6 : scan edge (search edge 3) 4 [967] 967 innerTail160_6 best = best := by
  rw [innerTail160_6, scan, if_neg (by decide), innerCandidates160_6_checked]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_7
    (if 3 ≤ innerCandidates160_6.length then search edge 3 [967, 2791] innerCandidates160_6 3758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan160_7

def innerTail160_5 : List Nat := 2557 :: innerTail160_6
def innerCandidates160_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates160_5_checked : candidates edge 4 967 2557 innerTail160_6 best.bound = innerCandidates160_5 := by
  unfold candidates
  have row : edge 2557 = PrimePairCache.smallRow2557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan160_5 : scan edge (search edge 3) 4 [967] 967 innerTail160_5 best = best := by
  rw [innerTail160_5, scan, if_neg (by decide), innerCandidates160_5_checked]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_6
    (if 3 ≤ innerCandidates160_5.length then search edge 3 [967, 2557] innerCandidates160_5 3524 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan160_6

def innerTail160_4 : List Nat := 2389 :: innerTail160_5
def innerCandidates160_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates160_4_checked : candidates edge 4 967 2389 innerTail160_5 best.bound = innerCandidates160_4 := by
  unfold candidates
  have row : edge 2389 = PrimePairCache.smallRow2389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan160_4 : scan edge (search edge 3) 4 [967] 967 innerTail160_4 best = best := by
  rw [innerTail160_4, scan, if_neg (by decide), innerCandidates160_4_checked]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_5
    (if 3 ≤ innerCandidates160_4.length then search edge 3 [967, 2389] innerCandidates160_4 3356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan160_5

def innerTail160_3 : List Nat := 1951 :: innerTail160_4
def innerCandidates160_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates160_3_checked : candidates edge 4 967 1951 innerTail160_4 best.bound = innerCandidates160_3 := by
  unfold candidates
  have row : edge 1951 = PrimePairCache.smallRow1951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan160_3 : scan edge (search edge 3) 4 [967] 967 innerTail160_3 best = best := by
  rw [innerTail160_3, scan, if_neg (by decide), innerCandidates160_3_checked]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_4
    (if 3 ≤ innerCandidates160_3.length then search edge 3 [967, 1951] innerCandidates160_3 2918 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan160_4

def innerTail160_2 : List Nat := 1663 :: innerTail160_3
def innerCandidates160_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates160_2_checked : candidates edge 4 967 1663 innerTail160_3 best.bound = innerCandidates160_2 := by
  unfold candidates
  have row : edge 1663 = PrimePairCache.smallRow1663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan160_2 : scan edge (search edge 3) 4 [967] 967 innerTail160_2 best = best := by
  rw [innerTail160_2, scan, if_neg (by decide), innerCandidates160_2_checked]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_3
    (if 3 ≤ innerCandidates160_2.length then search edge 3 [967, 1663] innerCandidates160_2 2630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan160_3

def innerTail160_1 : List Nat := 1381 :: innerTail160_2
def innerCandidates160_1 : List Nat := [8017]
set_option maxRecDepth 4096 in
theorem innerCandidates160_1_checked : candidates edge 4 967 1381 innerTail160_2 best.bound = innerCandidates160_1 := by
  unfold candidates
  have row : edge 1381 = PrimePairCache.smallRow1381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan160_1 : scan edge (search edge 3) 4 [967] 967 innerTail160_1 best = best := by
  rw [innerTail160_1, scan, if_neg (by decide), innerCandidates160_1_checked]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_2
    (if 3 ≤ innerCandidates160_1.length then search edge 3 [967, 1381] innerCandidates160_1 2348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan160_2

def innerTail160_0 : List Nat := 1117 :: innerTail160_1
def innerCandidates160_0 : List Nat := [3511]
set_option maxRecDepth 4096 in
theorem innerCandidates160_0_checked : candidates edge 4 967 1117 innerTail160_1 best.bound = innerCandidates160_0 := by
  unfold candidates
  have row : edge 1117 = PrimePairCache.smallRow1117.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan160_0 : scan edge (search edge 3) 4 [967] 967 innerTail160_0 best = best := by
  rw [innerTail160_0, scan, if_neg (by decide), innerCandidates160_0_checked]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_1
    (if 3 ≤ innerCandidates160_0.length then search edge 3 [967, 1117] innerCandidates160_0 2084 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan160_1

theorem rootBranch160 : search edge 4 [967] rootCandidates160 967 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [967] 967 innerTail160_0 best = best
  exact innerScan160_0

end ProjectEulerSolutions.PrimePairUpper.Search
