import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates368 : List Nat := [3163, 3229, 3463, 4297, 4339, 4441, 4723, 5119, 5179, 5449, 6421, 6823, 7159, 7951]
set_option maxRecDepth 4096 in
theorem rootCandidates368_checked : candidates edge 5 0 2539 tail369 best.bound = rootCandidates368 := by
  unfold candidates
  have row : edge 2539 = PrimePairCache.smallRow2539.contains := by funext q; rfl
  rw [row]
  decide

def innerTail368_14 : List Nat := []
theorem innerScan368_14 : scan edge (search edge 3) 4 [2539] 2539 innerTail368_14 best = best := rfl

def innerTail368_13 : List Nat := 7951 :: innerTail368_14
def innerCandidates368_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates368_13_checked : candidates edge 4 2539 7951 innerTail368_14 best.bound = innerCandidates368_13 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan368_13 : scan edge (search edge 3) 4 [2539] 2539 innerTail368_13 best = best := by
  rw [innerTail368_13, scan, if_neg (by decide), innerCandidates368_13_checked]
  change scan edge (search edge 3) 4 [2539] 2539 innerTail368_14
    (if 3 ≤ innerCandidates368_13.length then search edge 3 [2539, 7951] innerCandidates368_13 10490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan368_14

def innerTail368_12 : List Nat := 7159 :: innerTail368_13
def innerCandidates368_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates368_12_checked : candidates edge 4 2539 7159 innerTail368_13 best.bound = innerCandidates368_12 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan368_12 : scan edge (search edge 3) 4 [2539] 2539 innerTail368_12 best = best := by
  rw [innerTail368_12, scan, if_neg (by decide), innerCandidates368_12_checked]
  change scan edge (search edge 3) 4 [2539] 2539 innerTail368_13
    (if 3 ≤ innerCandidates368_12.length then search edge 3 [2539, 7159] innerCandidates368_12 9698 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan368_13

def innerTail368_11 : List Nat := 6823 :: innerTail368_12
def innerCandidates368_11 : List Nat := [7951]
set_option maxRecDepth 4096 in
theorem innerCandidates368_11_checked : candidates edge 4 2539 6823 innerTail368_12 best.bound = innerCandidates368_11 := by
  unfold candidates
  have row : edge 6823 = SearchData.row6823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan368_11 : scan edge (search edge 3) 4 [2539] 2539 innerTail368_11 best = best := by
  rw [innerTail368_11, scan, if_neg (by decide), innerCandidates368_11_checked]
  change scan edge (search edge 3) 4 [2539] 2539 innerTail368_12
    (if 3 ≤ innerCandidates368_11.length then search edge 3 [2539, 6823] innerCandidates368_11 9362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan368_12

def innerTail368_10 : List Nat := 6421 :: innerTail368_11
def innerCandidates368_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates368_10_checked : candidates edge 4 2539 6421 innerTail368_11 best.bound = innerCandidates368_10 := by
  unfold candidates
  have row : edge 6421 = SearchData.row6421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan368_10 : scan edge (search edge 3) 4 [2539] 2539 innerTail368_10 best = best := by
  rw [innerTail368_10, scan, if_neg (by decide), innerCandidates368_10_checked]
  change scan edge (search edge 3) 4 [2539] 2539 innerTail368_11
    (if 3 ≤ innerCandidates368_10.length then search edge 3 [2539, 6421] innerCandidates368_10 8960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan368_11

def innerTail368_9 : List Nat := 5449 :: innerTail368_10
def innerCandidates368_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates368_9_checked : candidates edge 4 2539 5449 innerTail368_10 best.bound = innerCandidates368_9 := by
  unfold candidates
  have row : edge 5449 = SearchData.row5449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan368_9 : scan edge (search edge 3) 4 [2539] 2539 innerTail368_9 best = best := by
  rw [innerTail368_9, scan, if_neg (by decide), innerCandidates368_9_checked]
  change scan edge (search edge 3) 4 [2539] 2539 innerTail368_10
    (if 3 ≤ innerCandidates368_9.length then search edge 3 [2539, 5449] innerCandidates368_9 7988 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan368_10

def innerTail368_8 : List Nat := 5179 :: innerTail368_9
def innerCandidates368_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates368_8_checked : candidates edge 4 2539 5179 innerTail368_9 best.bound = innerCandidates368_8 := by
  unfold candidates
  have row : edge 5179 = SearchData.row5179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan368_8 : scan edge (search edge 3) 4 [2539] 2539 innerTail368_8 best = best := by
  rw [innerTail368_8, scan, if_neg (by decide), innerCandidates368_8_checked]
  change scan edge (search edge 3) 4 [2539] 2539 innerTail368_9
    (if 3 ≤ innerCandidates368_8.length then search edge 3 [2539, 5179] innerCandidates368_8 7718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan368_9

def innerTail368_7 : List Nat := 5119 :: innerTail368_8
def innerCandidates368_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates368_7_checked : candidates edge 4 2539 5119 innerTail368_8 best.bound = innerCandidates368_7 := by
  unfold candidates
  have row : edge 5119 = SearchData.row5119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan368_7 : scan edge (search edge 3) 4 [2539] 2539 innerTail368_7 best = best := by
  rw [innerTail368_7, scan, if_neg (by decide), innerCandidates368_7_checked]
  change scan edge (search edge 3) 4 [2539] 2539 innerTail368_8
    (if 3 ≤ innerCandidates368_7.length then search edge 3 [2539, 5119] innerCandidates368_7 7658 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan368_8

def innerTail368_6 : List Nat := 4723 :: innerTail368_7
def innerCandidates368_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates368_6_checked : candidates edge 4 2539 4723 innerTail368_7 best.bound = innerCandidates368_6 := by
  unfold candidates
  have row : edge 4723 = SearchData.row4723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan368_6 : scan edge (search edge 3) 4 [2539] 2539 innerTail368_6 best = best := by
  rw [innerTail368_6, scan, if_neg (by decide), innerCandidates368_6_checked]
  change scan edge (search edge 3) 4 [2539] 2539 innerTail368_7
    (if 3 ≤ innerCandidates368_6.length then search edge 3 [2539, 4723] innerCandidates368_6 7262 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan368_7

def innerTail368_5 : List Nat := 4441 :: innerTail368_6
def innerCandidates368_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates368_5_checked : candidates edge 4 2539 4441 innerTail368_6 best.bound = innerCandidates368_5 := by
  unfold candidates
  have row : edge 4441 = SearchData.row4441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan368_5 : scan edge (search edge 3) 4 [2539] 2539 innerTail368_5 best = best := by
  rw [innerTail368_5, scan, if_neg (by decide), innerCandidates368_5_checked]
  change scan edge (search edge 3) 4 [2539] 2539 innerTail368_6
    (if 3 ≤ innerCandidates368_5.length then search edge 3 [2539, 4441] innerCandidates368_5 6980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan368_6

def innerTail368_4 : List Nat := 4339 :: innerTail368_5
def innerCandidates368_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates368_4_checked : candidates edge 4 2539 4339 innerTail368_5 best.bound = innerCandidates368_4 := by
  unfold candidates
  have row : edge 4339 = SearchData.row4339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan368_4 : scan edge (search edge 3) 4 [2539] 2539 innerTail368_4 best = best := by
  rw [innerTail368_4, scan, if_neg (by decide), innerCandidates368_4_checked]
  change scan edge (search edge 3) 4 [2539] 2539 innerTail368_5
    (if 3 ≤ innerCandidates368_4.length then search edge 3 [2539, 4339] innerCandidates368_4 6878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan368_5

def innerTail368_3 : List Nat := 4297 :: innerTail368_4
def innerCandidates368_3 : List Nat := [4441]
set_option maxRecDepth 4096 in
theorem innerCandidates368_3_checked : candidates edge 4 2539 4297 innerTail368_4 best.bound = innerCandidates368_3 := by
  unfold candidates
  have row : edge 4297 = SearchData.row4297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan368_3 : scan edge (search edge 3) 4 [2539] 2539 innerTail368_3 best = best := by
  rw [innerTail368_3, scan, if_neg (by decide), innerCandidates368_3_checked]
  change scan edge (search edge 3) 4 [2539] 2539 innerTail368_4
    (if 3 ≤ innerCandidates368_3.length then search edge 3 [2539, 4297] innerCandidates368_3 6836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan368_4

def innerTail368_2 : List Nat := 3463 :: innerTail368_3
def innerCandidates368_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates368_2_checked : candidates edge 4 2539 3463 innerTail368_3 best.bound = innerCandidates368_2 := by
  unfold candidates
  have row : edge 3463 = SearchData.row3463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan368_2 : scan edge (search edge 3) 4 [2539] 2539 innerTail368_2 best = best := by
  rw [innerTail368_2, scan, if_neg (by decide), innerCandidates368_2_checked]
  change scan edge (search edge 3) 4 [2539] 2539 innerTail368_3
    (if 3 ≤ innerCandidates368_2.length then search edge 3 [2539, 3463] innerCandidates368_2 6002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan368_3

def innerTail368_1 : List Nat := 3229 :: innerTail368_2
def innerCandidates368_1 : List Nat := [3463]
set_option maxRecDepth 4096 in
theorem innerCandidates368_1_checked : candidates edge 4 2539 3229 innerTail368_2 best.bound = innerCandidates368_1 := by
  unfold candidates
  have row : edge 3229 = SearchData.row3229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan368_1 : scan edge (search edge 3) 4 [2539] 2539 innerTail368_1 best = best := by
  rw [innerTail368_1, scan, if_neg (by decide), innerCandidates368_1_checked]
  change scan edge (search edge 3) 4 [2539] 2539 innerTail368_2
    (if 3 ≤ innerCandidates368_1.length then search edge 3 [2539, 3229] innerCandidates368_1 5768 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan368_2

def innerTail368_0 : List Nat := 3163 :: innerTail368_1
def innerCandidates368_0 : List Nat := [7951]
set_option maxRecDepth 4096 in
theorem innerCandidates368_0_checked : candidates edge 4 2539 3163 innerTail368_1 best.bound = innerCandidates368_0 := by
  unfold candidates
  have row : edge 3163 = SearchData.row3163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan368_0 : scan edge (search edge 3) 4 [2539] 2539 innerTail368_0 best = best := by
  rw [innerTail368_0, scan, if_neg (by decide), innerCandidates368_0_checked]
  change scan edge (search edge 3) 4 [2539] 2539 innerTail368_1
    (if 3 ≤ innerCandidates368_0.length then search edge 3 [2539, 3163] innerCandidates368_0 5702 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan368_1

theorem rootBranch368 : search edge 4 [2539] rootCandidates368 2539 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2539] 2539 innerTail368_0 best = best
  exact innerScan368_0

def rootCandidates367 : List Nat := [2549, 2741, 3539, 3797, 4643, 4889, 4967, 5273, 5309, 5651, 6329, 6659, 6701, 7883, 8123, 8171]
set_option maxRecDepth 4096 in
theorem rootCandidates367_checked : candidates edge 5 0 2531 tail368 best.bound = rootCandidates367 := by
  unfold candidates
  have row : edge 2531 = PrimePairCache.smallRow2531.contains := by funext q; rfl
  rw [row]
  decide

def innerTail367_16 : List Nat := []
theorem innerScan367_16 : scan edge (search edge 3) 4 [2531] 2531 innerTail367_16 best = best := rfl

def innerTail367_15 : List Nat := 8171 :: innerTail367_16
def innerCandidates367_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates367_15_checked : candidates edge 4 2531 8171 innerTail367_16 best.bound = innerCandidates367_15 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan367_15 : scan edge (search edge 3) 4 [2531] 2531 innerTail367_15 best = best := by
  rw [innerTail367_15, scan, if_neg (by decide), innerCandidates367_15_checked]
  change scan edge (search edge 3) 4 [2531] 2531 innerTail367_16
    (if 3 ≤ innerCandidates367_15.length then search edge 3 [2531, 8171] innerCandidates367_15 10702 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan367_16

def innerTail367_14 : List Nat := 8123 :: innerTail367_15
def innerCandidates367_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates367_14_checked : candidates edge 4 2531 8123 innerTail367_15 best.bound = innerCandidates367_14 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan367_14 : scan edge (search edge 3) 4 [2531] 2531 innerTail367_14 best = best := by
  rw [innerTail367_14, scan, if_neg (by decide), innerCandidates367_14_checked]
  change scan edge (search edge 3) 4 [2531] 2531 innerTail367_15
    (if 3 ≤ innerCandidates367_14.length then search edge 3 [2531, 8123] innerCandidates367_14 10654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan367_15

def innerTail367_13 : List Nat := 7883 :: innerTail367_14
def innerCandidates367_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates367_13_checked : candidates edge 4 2531 7883 innerTail367_14 best.bound = innerCandidates367_13 := by
  unfold candidates
  have row : edge 7883 = SearchData.row7883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan367_13 : scan edge (search edge 3) 4 [2531] 2531 innerTail367_13 best = best := by
  rw [innerTail367_13, scan, if_neg (by decide), innerCandidates367_13_checked]
  change scan edge (search edge 3) 4 [2531] 2531 innerTail367_14
    (if 3 ≤ innerCandidates367_13.length then search edge 3 [2531, 7883] innerCandidates367_13 10414 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan367_14

def innerTail367_12 : List Nat := 6701 :: innerTail367_13
def innerCandidates367_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates367_12_checked : candidates edge 4 2531 6701 innerTail367_13 best.bound = innerCandidates367_12 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan367_12 : scan edge (search edge 3) 4 [2531] 2531 innerTail367_12 best = best := by
  rw [innerTail367_12, scan, if_neg (by decide), innerCandidates367_12_checked]
  change scan edge (search edge 3) 4 [2531] 2531 innerTail367_13
    (if 3 ≤ innerCandidates367_12.length then search edge 3 [2531, 6701] innerCandidates367_12 9232 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan367_13

def innerTail367_11 : List Nat := 6659 :: innerTail367_12
def innerCandidates367_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates367_11_checked : candidates edge 4 2531 6659 innerTail367_12 best.bound = innerCandidates367_11 := by
  unfold candidates
  have row : edge 6659 = SearchData.row6659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan367_11 : scan edge (search edge 3) 4 [2531] 2531 innerTail367_11 best = best := by
  rw [innerTail367_11, scan, if_neg (by decide), innerCandidates367_11_checked]
  change scan edge (search edge 3) 4 [2531] 2531 innerTail367_12
    (if 3 ≤ innerCandidates367_11.length then search edge 3 [2531, 6659] innerCandidates367_11 9190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan367_12

def innerTail367_10 : List Nat := 6329 :: innerTail367_11
def innerCandidates367_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates367_10_checked : candidates edge 4 2531 6329 innerTail367_11 best.bound = innerCandidates367_10 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan367_10 : scan edge (search edge 3) 4 [2531] 2531 innerTail367_10 best = best := by
  rw [innerTail367_10, scan, if_neg (by decide), innerCandidates367_10_checked]
  change scan edge (search edge 3) 4 [2531] 2531 innerTail367_11
    (if 3 ≤ innerCandidates367_10.length then search edge 3 [2531, 6329] innerCandidates367_10 8860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan367_11

def innerTail367_9 : List Nat := 5651 :: innerTail367_10
def innerCandidates367_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates367_9_checked : candidates edge 4 2531 5651 innerTail367_10 best.bound = innerCandidates367_9 := by
  unfold candidates
  have row : edge 5651 = SearchData.row5651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan367_9 : scan edge (search edge 3) 4 [2531] 2531 innerTail367_9 best = best := by
  rw [innerTail367_9, scan, if_neg (by decide), innerCandidates367_9_checked]
  change scan edge (search edge 3) 4 [2531] 2531 innerTail367_10
    (if 3 ≤ innerCandidates367_9.length then search edge 3 [2531, 5651] innerCandidates367_9 8182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan367_10

def innerTail367_8 : List Nat := 5309 :: innerTail367_9
def innerCandidates367_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates367_8_checked : candidates edge 4 2531 5309 innerTail367_9 best.bound = innerCandidates367_8 := by
  unfold candidates
  have row : edge 5309 = SearchData.row5309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan367_8 : scan edge (search edge 3) 4 [2531] 2531 innerTail367_8 best = best := by
  rw [innerTail367_8, scan, if_neg (by decide), innerCandidates367_8_checked]
  change scan edge (search edge 3) 4 [2531] 2531 innerTail367_9
    (if 3 ≤ innerCandidates367_8.length then search edge 3 [2531, 5309] innerCandidates367_8 7840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan367_9

def innerTail367_7 : List Nat := 5273 :: innerTail367_8
def innerCandidates367_7 : List Nat := [5651]
set_option maxRecDepth 4096 in
theorem innerCandidates367_7_checked : candidates edge 4 2531 5273 innerTail367_8 best.bound = innerCandidates367_7 := by
  unfold candidates
  have row : edge 5273 = SearchData.row5273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan367_7 : scan edge (search edge 3) 4 [2531] 2531 innerTail367_7 best = best := by
  rw [innerTail367_7, scan, if_neg (by decide), innerCandidates367_7_checked]
  change scan edge (search edge 3) 4 [2531] 2531 innerTail367_8
    (if 3 ≤ innerCandidates367_7.length then search edge 3 [2531, 5273] innerCandidates367_7 7804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan367_8

def innerTail367_6 : List Nat := 4967 :: innerTail367_7
def innerCandidates367_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates367_6_checked : candidates edge 4 2531 4967 innerTail367_7 best.bound = innerCandidates367_6 := by
  unfold candidates
  have row : edge 4967 = SearchData.row4967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan367_6 : scan edge (search edge 3) 4 [2531] 2531 innerTail367_6 best = best := by
  rw [innerTail367_6, scan, if_neg (by decide), innerCandidates367_6_checked]
  change scan edge (search edge 3) 4 [2531] 2531 innerTail367_7
    (if 3 ≤ innerCandidates367_6.length then search edge 3 [2531, 4967] innerCandidates367_6 7498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan367_7

def innerTail367_5 : List Nat := 4889 :: innerTail367_6
def innerCandidates367_5 : List Nat := [8123]
set_option maxRecDepth 4096 in
theorem innerCandidates367_5_checked : candidates edge 4 2531 4889 innerTail367_6 best.bound = innerCandidates367_5 := by
  unfold candidates
  have row : edge 4889 = SearchData.row4889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan367_5 : scan edge (search edge 3) 4 [2531] 2531 innerTail367_5 best = best := by
  rw [innerTail367_5, scan, if_neg (by decide), innerCandidates367_5_checked]
  change scan edge (search edge 3) 4 [2531] 2531 innerTail367_6
    (if 3 ≤ innerCandidates367_5.length then search edge 3 [2531, 4889] innerCandidates367_5 7420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan367_6

def innerTail367_4 : List Nat := 4643 :: innerTail367_5
def innerCandidates367_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates367_4_checked : candidates edge 4 2531 4643 innerTail367_5 best.bound = innerCandidates367_4 := by
  unfold candidates
  have row : edge 4643 = SearchData.row4643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan367_4 : scan edge (search edge 3) 4 [2531] 2531 innerTail367_4 best = best := by
  rw [innerTail367_4, scan, if_neg (by decide), innerCandidates367_4_checked]
  change scan edge (search edge 3) 4 [2531] 2531 innerTail367_5
    (if 3 ≤ innerCandidates367_4.length then search edge 3 [2531, 4643] innerCandidates367_4 7174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan367_5

def innerTail367_3 : List Nat := 3797 :: innerTail367_4
def innerCandidates367_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates367_3_checked : candidates edge 4 2531 3797 innerTail367_4 best.bound = innerCandidates367_3 := by
  unfold candidates
  have row : edge 3797 = SearchData.row3797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan367_3 : scan edge (search edge 3) 4 [2531] 2531 innerTail367_3 best = best := by
  rw [innerTail367_3, scan, if_neg (by decide), innerCandidates367_3_checked]
  change scan edge (search edge 3) 4 [2531] 2531 innerTail367_4
    (if 3 ≤ innerCandidates367_3.length then search edge 3 [2531, 3797] innerCandidates367_3 6328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan367_4

def innerTail367_2 : List Nat := 3539 :: innerTail367_3
def innerCandidates367_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates367_2_checked : candidates edge 4 2531 3539 innerTail367_3 best.bound = innerCandidates367_2 := by
  unfold candidates
  have row : edge 3539 = SearchData.row3539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan367_2 : scan edge (search edge 3) 4 [2531] 2531 innerTail367_2 best = best := by
  rw [innerTail367_2, scan, if_neg (by decide), innerCandidates367_2_checked]
  change scan edge (search edge 3) 4 [2531] 2531 innerTail367_3
    (if 3 ≤ innerCandidates367_2.length then search edge 3 [2531, 3539] innerCandidates367_2 6070 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan367_3

def innerTail367_1 : List Nat := 2741 :: innerTail367_2
def innerCandidates367_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates367_1_checked : candidates edge 4 2531 2741 innerTail367_2 best.bound = innerCandidates367_1 := by
  unfold candidates
  have row : edge 2741 = SearchData.row2741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan367_1 : scan edge (search edge 3) 4 [2531] 2531 innerTail367_1 best = best := by
  rw [innerTail367_1, scan, if_neg (by decide), innerCandidates367_1_checked]
  change scan edge (search edge 3) 4 [2531] 2531 innerTail367_2
    (if 3 ≤ innerCandidates367_1.length then search edge 3 [2531, 2741] innerCandidates367_1 5272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan367_2

def innerTail367_0 : List Nat := 2549 :: innerTail367_1
def innerCandidates367_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates367_0_checked : candidates edge 4 2531 2549 innerTail367_1 best.bound = innerCandidates367_0 := by
  unfold candidates
  have row : edge 2549 = PrimePairCache.smallRow2549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan367_0 : scan edge (search edge 3) 4 [2531] 2531 innerTail367_0 best = best := by
  rw [innerTail367_0, scan, if_neg (by decide), innerCandidates367_0_checked]
  change scan edge (search edge 3) 4 [2531] 2531 innerTail367_1
    (if 3 ≤ innerCandidates367_0.length then search edge 3 [2531, 2549] innerCandidates367_0 5080 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan367_1

theorem rootBranch367 : search edge 4 [2531] rootCandidates367 2531 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2531] 2531 innerTail367_0 best = best
  exact innerScan367_0

def rootCandidates366 : List Nat := [2797, 3271, 3697, 3739, 4201, 4363, 4759, 4999, 5557, 6343, 7561]
set_option maxRecDepth 4096 in
theorem rootCandidates366_checked : candidates edge 5 0 2521 tail367 best.bound = rootCandidates366 := by
  unfold candidates
  have row : edge 2521 = PrimePairCache.smallRow2521.contains := by funext q; rfl
  rw [row]
  decide

def innerTail366_11 : List Nat := []
theorem innerScan366_11 : scan edge (search edge 3) 4 [2521] 2521 innerTail366_11 best = best := rfl

def innerTail366_10 : List Nat := 7561 :: innerTail366_11
def innerCandidates366_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates366_10_checked : candidates edge 4 2521 7561 innerTail366_11 best.bound = innerCandidates366_10 := by
  unfold candidates
  have row : edge 7561 = SearchData.row7561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan366_10 : scan edge (search edge 3) 4 [2521] 2521 innerTail366_10 best = best := by
  rw [innerTail366_10, scan, if_neg (by decide), innerCandidates366_10_checked]
  change scan edge (search edge 3) 4 [2521] 2521 innerTail366_11
    (if 3 ≤ innerCandidates366_10.length then search edge 3 [2521, 7561] innerCandidates366_10 10082 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan366_11

def innerTail366_9 : List Nat := 6343 :: innerTail366_10
def innerCandidates366_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates366_9_checked : candidates edge 4 2521 6343 innerTail366_10 best.bound = innerCandidates366_9 := by
  unfold candidates
  have row : edge 6343 = SearchData.row6343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan366_9 : scan edge (search edge 3) 4 [2521] 2521 innerTail366_9 best = best := by
  rw [innerTail366_9, scan, if_neg (by decide), innerCandidates366_9_checked]
  change scan edge (search edge 3) 4 [2521] 2521 innerTail366_10
    (if 3 ≤ innerCandidates366_9.length then search edge 3 [2521, 6343] innerCandidates366_9 8864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan366_10

def innerTail366_8 : List Nat := 5557 :: innerTail366_9
def innerCandidates366_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates366_8_checked : candidates edge 4 2521 5557 innerTail366_9 best.bound = innerCandidates366_8 := by
  unfold candidates
  have row : edge 5557 = SearchData.row5557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan366_8 : scan edge (search edge 3) 4 [2521] 2521 innerTail366_8 best = best := by
  rw [innerTail366_8, scan, if_neg (by decide), innerCandidates366_8_checked]
  change scan edge (search edge 3) 4 [2521] 2521 innerTail366_9
    (if 3 ≤ innerCandidates366_8.length then search edge 3 [2521, 5557] innerCandidates366_8 8078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan366_9

def innerTail366_7 : List Nat := 4999 :: innerTail366_8
def innerCandidates366_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates366_7_checked : candidates edge 4 2521 4999 innerTail366_8 best.bound = innerCandidates366_7 := by
  unfold candidates
  have row : edge 4999 = SearchData.row4999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan366_7 : scan edge (search edge 3) 4 [2521] 2521 innerTail366_7 best = best := by
  rw [innerTail366_7, scan, if_neg (by decide), innerCandidates366_7_checked]
  change scan edge (search edge 3) 4 [2521] 2521 innerTail366_8
    (if 3 ≤ innerCandidates366_7.length then search edge 3 [2521, 4999] innerCandidates366_7 7520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan366_8

def innerTail366_6 : List Nat := 4759 :: innerTail366_7
def innerCandidates366_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates366_6_checked : candidates edge 4 2521 4759 innerTail366_7 best.bound = innerCandidates366_6 := by
  unfold candidates
  have row : edge 4759 = SearchData.row4759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan366_6 : scan edge (search edge 3) 4 [2521] 2521 innerTail366_6 best = best := by
  rw [innerTail366_6, scan, if_neg (by decide), innerCandidates366_6_checked]
  change scan edge (search edge 3) 4 [2521] 2521 innerTail366_7
    (if 3 ≤ innerCandidates366_6.length then search edge 3 [2521, 4759] innerCandidates366_6 7280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan366_7

def innerTail366_5 : List Nat := 4363 :: innerTail366_6
def innerCandidates366_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates366_5_checked : candidates edge 4 2521 4363 innerTail366_6 best.bound = innerCandidates366_5 := by
  unfold candidates
  have row : edge 4363 = SearchData.row4363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan366_5 : scan edge (search edge 3) 4 [2521] 2521 innerTail366_5 best = best := by
  rw [innerTail366_5, scan, if_neg (by decide), innerCandidates366_5_checked]
  change scan edge (search edge 3) 4 [2521] 2521 innerTail366_6
    (if 3 ≤ innerCandidates366_5.length then search edge 3 [2521, 4363] innerCandidates366_5 6884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan366_6

def innerTail366_4 : List Nat := 4201 :: innerTail366_5
def innerCandidates366_4 : List Nat := [4759]
set_option maxRecDepth 4096 in
theorem innerCandidates366_4_checked : candidates edge 4 2521 4201 innerTail366_5 best.bound = innerCandidates366_4 := by
  unfold candidates
  have row : edge 4201 = SearchData.row4201.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan366_4 : scan edge (search edge 3) 4 [2521] 2521 innerTail366_4 best = best := by
  rw [innerTail366_4, scan, if_neg (by decide), innerCandidates366_4_checked]
  change scan edge (search edge 3) 4 [2521] 2521 innerTail366_5
    (if 3 ≤ innerCandidates366_4.length then search edge 3 [2521, 4201] innerCandidates366_4 6722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan366_5

def innerTail366_3 : List Nat := 3739 :: innerTail366_4
def innerCandidates366_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates366_3_checked : candidates edge 4 2521 3739 innerTail366_4 best.bound = innerCandidates366_3 := by
  unfold candidates
  have row : edge 3739 = SearchData.row3739.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan366_3 : scan edge (search edge 3) 4 [2521] 2521 innerTail366_3 best = best := by
  rw [innerTail366_3, scan, if_neg (by decide), innerCandidates366_3_checked]
  change scan edge (search edge 3) 4 [2521] 2521 innerTail366_4
    (if 3 ≤ innerCandidates366_3.length then search edge 3 [2521, 3739] innerCandidates366_3 6260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan366_4

def innerTail366_2 : List Nat := 3697 :: innerTail366_3
def innerCandidates366_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates366_2_checked : candidates edge 4 2521 3697 innerTail366_3 best.bound = innerCandidates366_2 := by
  unfold candidates
  have row : edge 3697 = SearchData.row3697.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan366_2 : scan edge (search edge 3) 4 [2521] 2521 innerTail366_2 best = best := by
  rw [innerTail366_2, scan, if_neg (by decide), innerCandidates366_2_checked]
  change scan edge (search edge 3) 4 [2521] 2521 innerTail366_3
    (if 3 ≤ innerCandidates366_2.length then search edge 3 [2521, 3697] innerCandidates366_2 6218 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan366_3

def innerTail366_1 : List Nat := 3271 :: innerTail366_2
def innerCandidates366_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates366_1_checked : candidates edge 4 2521 3271 innerTail366_2 best.bound = innerCandidates366_1 := by
  unfold candidates
  have row : edge 3271 = SearchData.row3271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan366_1 : scan edge (search edge 3) 4 [2521] 2521 innerTail366_1 best = best := by
  rw [innerTail366_1, scan, if_neg (by decide), innerCandidates366_1_checked]
  change scan edge (search edge 3) 4 [2521] 2521 innerTail366_2
    (if 3 ≤ innerCandidates366_1.length then search edge 3 [2521, 3271] innerCandidates366_1 5792 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan366_2

def innerTail366_0 : List Nat := 2797 :: innerTail366_1
def innerCandidates366_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates366_0_checked : candidates edge 4 2521 2797 innerTail366_1 best.bound = innerCandidates366_0 := by
  unfold candidates
  have row : edge 2797 = SearchData.row2797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan366_0 : scan edge (search edge 3) 4 [2521] 2521 innerTail366_0 best = best := by
  rw [innerTail366_0, scan, if_neg (by decide), innerCandidates366_0_checked]
  change scan edge (search edge 3) 4 [2521] 2521 innerTail366_1
    (if 3 ≤ innerCandidates366_0.length then search edge 3 [2521, 2797] innerCandidates366_0 5318 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan366_1

theorem rootBranch366 : search edge 4 [2521] rootCandidates366 2521 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2521] 2521 innerTail366_0 best = best
  exact innerScan366_0

def rootCandidates365 : List Nat := [3391, 3517, 3559, 4447, 5281, 5581, 5623, 5869, 6079, 6163, 6421, 6763, 6967, 7393, 7591]
set_option maxRecDepth 4096 in
theorem rootCandidates365_checked : candidates edge 5 0 2503 tail366 best.bound = rootCandidates365 := by
  unfold candidates
  have row : edge 2503 = PrimePairCache.smallRow2503.contains := by funext q; rfl
  rw [row]
  decide

def innerTail365_15 : List Nat := []
theorem innerScan365_15 : scan edge (search edge 3) 4 [2503] 2503 innerTail365_15 best = best := rfl

def innerTail365_14 : List Nat := 7591 :: innerTail365_15
def innerCandidates365_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates365_14_checked : candidates edge 4 2503 7591 innerTail365_15 best.bound = innerCandidates365_14 := by
  unfold candidates
  have row : edge 7591 = SearchData.row7591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan365_14 : scan edge (search edge 3) 4 [2503] 2503 innerTail365_14 best = best := by
  rw [innerTail365_14, scan, if_neg (by decide), innerCandidates365_14_checked]
  change scan edge (search edge 3) 4 [2503] 2503 innerTail365_15
    (if 3 ≤ innerCandidates365_14.length then search edge 3 [2503, 7591] innerCandidates365_14 10094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan365_15

def innerTail365_13 : List Nat := 7393 :: innerTail365_14
def innerCandidates365_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates365_13_checked : candidates edge 4 2503 7393 innerTail365_14 best.bound = innerCandidates365_13 := by
  unfold candidates
  have row : edge 7393 = SearchData.row7393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan365_13 : scan edge (search edge 3) 4 [2503] 2503 innerTail365_13 best = best := by
  rw [innerTail365_13, scan, if_neg (by decide), innerCandidates365_13_checked]
  change scan edge (search edge 3) 4 [2503] 2503 innerTail365_14
    (if 3 ≤ innerCandidates365_13.length then search edge 3 [2503, 7393] innerCandidates365_13 9896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan365_14

def innerTail365_12 : List Nat := 6967 :: innerTail365_13
def innerCandidates365_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates365_12_checked : candidates edge 4 2503 6967 innerTail365_13 best.bound = innerCandidates365_12 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan365_12 : scan edge (search edge 3) 4 [2503] 2503 innerTail365_12 best = best := by
  rw [innerTail365_12, scan, if_neg (by decide), innerCandidates365_12_checked]
  change scan edge (search edge 3) 4 [2503] 2503 innerTail365_13
    (if 3 ≤ innerCandidates365_12.length then search edge 3 [2503, 6967] innerCandidates365_12 9470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan365_13

def innerTail365_11 : List Nat := 6763 :: innerTail365_12
def innerCandidates365_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates365_11_checked : candidates edge 4 2503 6763 innerTail365_12 best.bound = innerCandidates365_11 := by
  unfold candidates
  have row : edge 6763 = SearchData.row6763.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan365_11 : scan edge (search edge 3) 4 [2503] 2503 innerTail365_11 best = best := by
  rw [innerTail365_11, scan, if_neg (by decide), innerCandidates365_11_checked]
  change scan edge (search edge 3) 4 [2503] 2503 innerTail365_12
    (if 3 ≤ innerCandidates365_11.length then search edge 3 [2503, 6763] innerCandidates365_11 9266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan365_12

def innerTail365_10 : List Nat := 6421 :: innerTail365_11
def innerCandidates365_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates365_10_checked : candidates edge 4 2503 6421 innerTail365_11 best.bound = innerCandidates365_10 := by
  unfold candidates
  have row : edge 6421 = SearchData.row6421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan365_10 : scan edge (search edge 3) 4 [2503] 2503 innerTail365_10 best = best := by
  rw [innerTail365_10, scan, if_neg (by decide), innerCandidates365_10_checked]
  change scan edge (search edge 3) 4 [2503] 2503 innerTail365_11
    (if 3 ≤ innerCandidates365_10.length then search edge 3 [2503, 6421] innerCandidates365_10 8924 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan365_11

def innerTail365_9 : List Nat := 6163 :: innerTail365_10
def innerCandidates365_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates365_9_checked : candidates edge 4 2503 6163 innerTail365_10 best.bound = innerCandidates365_9 := by
  unfold candidates
  have row : edge 6163 = SearchData.row6163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan365_9 : scan edge (search edge 3) 4 [2503] 2503 innerTail365_9 best = best := by
  rw [innerTail365_9, scan, if_neg (by decide), innerCandidates365_9_checked]
  change scan edge (search edge 3) 4 [2503] 2503 innerTail365_10
    (if 3 ≤ innerCandidates365_9.length then search edge 3 [2503, 6163] innerCandidates365_9 8666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan365_10

def innerTail365_8 : List Nat := 6079 :: innerTail365_9
def innerCandidates365_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates365_8_checked : candidates edge 4 2503 6079 innerTail365_9 best.bound = innerCandidates365_8 := by
  unfold candidates
  have row : edge 6079 = SearchData.row6079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan365_8 : scan edge (search edge 3) 4 [2503] 2503 innerTail365_8 best = best := by
  rw [innerTail365_8, scan, if_neg (by decide), innerCandidates365_8_checked]
  change scan edge (search edge 3) 4 [2503] 2503 innerTail365_9
    (if 3 ≤ innerCandidates365_8.length then search edge 3 [2503, 6079] innerCandidates365_8 8582 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan365_9

def innerTail365_7 : List Nat := 5869 :: innerTail365_8
def innerCandidates365_7 : List Nat := [6163]
set_option maxRecDepth 4096 in
theorem innerCandidates365_7_checked : candidates edge 4 2503 5869 innerTail365_8 best.bound = innerCandidates365_7 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan365_7 : scan edge (search edge 3) 4 [2503] 2503 innerTail365_7 best = best := by
  rw [innerTail365_7, scan, if_neg (by decide), innerCandidates365_7_checked]
  change scan edge (search edge 3) 4 [2503] 2503 innerTail365_8
    (if 3 ≤ innerCandidates365_7.length then search edge 3 [2503, 5869] innerCandidates365_7 8372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan365_8

def innerTail365_6 : List Nat := 5623 :: innerTail365_7
def innerCandidates365_6 : List Nat := [6421]
set_option maxRecDepth 4096 in
theorem innerCandidates365_6_checked : candidates edge 4 2503 5623 innerTail365_7 best.bound = innerCandidates365_6 := by
  unfold candidates
  have row : edge 5623 = SearchData.row5623.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan365_6 : scan edge (search edge 3) 4 [2503] 2503 innerTail365_6 best = best := by
  rw [innerTail365_6, scan, if_neg (by decide), innerCandidates365_6_checked]
  change scan edge (search edge 3) 4 [2503] 2503 innerTail365_7
    (if 3 ≤ innerCandidates365_6.length then search edge 3 [2503, 5623] innerCandidates365_6 8126 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan365_7

def innerTail365_5 : List Nat := 5581 :: innerTail365_6
def innerCandidates365_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates365_5_checked : candidates edge 4 2503 5581 innerTail365_6 best.bound = innerCandidates365_5 := by
  unfold candidates
  have row : edge 5581 = SearchData.row5581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan365_5 : scan edge (search edge 3) 4 [2503] 2503 innerTail365_5 best = best := by
  rw [innerTail365_5, scan, if_neg (by decide), innerCandidates365_5_checked]
  change scan edge (search edge 3) 4 [2503] 2503 innerTail365_6
    (if 3 ≤ innerCandidates365_5.length then search edge 3 [2503, 5581] innerCandidates365_5 8084 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan365_6

def innerTail365_4 : List Nat := 5281 :: innerTail365_5
def innerCandidates365_4 : List Nat := [5869]
set_option maxRecDepth 4096 in
theorem innerCandidates365_4_checked : candidates edge 4 2503 5281 innerTail365_5 best.bound = innerCandidates365_4 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan365_4 : scan edge (search edge 3) 4 [2503] 2503 innerTail365_4 best = best := by
  rw [innerTail365_4, scan, if_neg (by decide), innerCandidates365_4_checked]
  change scan edge (search edge 3) 4 [2503] 2503 innerTail365_5
    (if 3 ≤ innerCandidates365_4.length then search edge 3 [2503, 5281] innerCandidates365_4 7784 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan365_5

def innerTail365_3 : List Nat := 4447 :: innerTail365_4
def innerCandidates365_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates365_3_checked : candidates edge 4 2503 4447 innerTail365_4 best.bound = innerCandidates365_3 := by
  unfold candidates
  have row : edge 4447 = SearchData.row4447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan365_3 : scan edge (search edge 3) 4 [2503] 2503 innerTail365_3 best = best := by
  rw [innerTail365_3, scan, if_neg (by decide), innerCandidates365_3_checked]
  change scan edge (search edge 3) 4 [2503] 2503 innerTail365_4
    (if 3 ≤ innerCandidates365_3.length then search edge 3 [2503, 4447] innerCandidates365_3 6950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan365_4

def innerTail365_2 : List Nat := 3559 :: innerTail365_3
def innerCandidates365_2 : List Nat := [5869]
set_option maxRecDepth 4096 in
theorem innerCandidates365_2_checked : candidates edge 4 2503 3559 innerTail365_3 best.bound = innerCandidates365_2 := by
  unfold candidates
  have row : edge 3559 = SearchData.row3559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan365_2 : scan edge (search edge 3) 4 [2503] 2503 innerTail365_2 best = best := by
  rw [innerTail365_2, scan, if_neg (by decide), innerCandidates365_2_checked]
  change scan edge (search edge 3) 4 [2503] 2503 innerTail365_3
    (if 3 ≤ innerCandidates365_2.length then search edge 3 [2503, 3559] innerCandidates365_2 6062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan365_3

def innerTail365_1 : List Nat := 3517 :: innerTail365_2
def innerCandidates365_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates365_1_checked : candidates edge 4 2503 3517 innerTail365_2 best.bound = innerCandidates365_1 := by
  unfold candidates
  have row : edge 3517 = SearchData.row3517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan365_1 : scan edge (search edge 3) 4 [2503] 2503 innerTail365_1 best = best := by
  rw [innerTail365_1, scan, if_neg (by decide), innerCandidates365_1_checked]
  change scan edge (search edge 3) 4 [2503] 2503 innerTail365_2
    (if 3 ≤ innerCandidates365_1.length then search edge 3 [2503, 3517] innerCandidates365_1 6020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan365_2

def innerTail365_0 : List Nat := 3391 :: innerTail365_1
def innerCandidates365_0 : List Nat := [6079]
set_option maxRecDepth 4096 in
theorem innerCandidates365_0_checked : candidates edge 4 2503 3391 innerTail365_1 best.bound = innerCandidates365_0 := by
  unfold candidates
  have row : edge 3391 = SearchData.row3391.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan365_0 : scan edge (search edge 3) 4 [2503] 2503 innerTail365_0 best = best := by
  rw [innerTail365_0, scan, if_neg (by decide), innerCandidates365_0_checked]
  change scan edge (search edge 3) 4 [2503] 2503 innerTail365_1
    (if 3 ≤ innerCandidates365_0.length then search edge 3 [2503, 3391] innerCandidates365_0 5894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan365_1

theorem rootBranch365 : search edge 4 [2503] rootCandidates365 2503 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2503] 2503 innerTail365_0 best = best
  exact innerScan365_0

end ProjectEulerSolutions.PrimePairUpper.Search
