import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates35 : List Nat := [193, 307, 367, 409, 613, 811, 883, 997, 1579, 1741, 1783, 2311, 2383, 2647, 3823, 4177, 4441, 4597, 5563, 6343, 6577, 6781, 6823, 6997, 7243, 7927]
set_option maxRecDepth 4096 in
theorem rootCandidates35_checked : candidates edge 5 0 163 tail36 best.bound = rootCandidates35 := by
  unfold candidates
  have row : edge 163 = PrimePairCache.smallRow163.contains := by funext q; rfl
  rw [row]
  decide

def innerTail35_26 : List Nat := []
theorem innerScan35_26 : scan edge (search edge 3) 4 [163] 163 innerTail35_26 best = best := rfl

def innerTail35_25 : List Nat := 7927 :: innerTail35_26
def innerCandidates35_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates35_25_checked : candidates edge 4 163 7927 innerTail35_26 best.bound = innerCandidates35_25 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_25 : scan edge (search edge 3) 4 [163] 163 innerTail35_25 best = best := by
  rw [innerTail35_25, scan, if_neg (by decide), innerCandidates35_25_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_26
    (if 3 ≤ innerCandidates35_25.length then search edge 3 [163, 7927] innerCandidates35_25 8090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_26

def innerTail35_24 : List Nat := 7243 :: innerTail35_25
def innerCandidates35_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates35_24_checked : candidates edge 4 163 7243 innerTail35_25 best.bound = innerCandidates35_24 := by
  unfold candidates
  have row : edge 7243 = SearchData.row7243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_24 : scan edge (search edge 3) 4 [163] 163 innerTail35_24 best = best := by
  rw [innerTail35_24, scan, if_neg (by decide), innerCandidates35_24_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_25
    (if 3 ≤ innerCandidates35_24.length then search edge 3 [163, 7243] innerCandidates35_24 7406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_25

def innerTail35_23 : List Nat := 6997 :: innerTail35_24
def innerCandidates35_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates35_23_checked : candidates edge 4 163 6997 innerTail35_24 best.bound = innerCandidates35_23 := by
  unfold candidates
  have row : edge 6997 = SearchData.row6997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_23 : scan edge (search edge 3) 4 [163] 163 innerTail35_23 best = best := by
  rw [innerTail35_23, scan, if_neg (by decide), innerCandidates35_23_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_24
    (if 3 ≤ innerCandidates35_23.length then search edge 3 [163, 6997] innerCandidates35_23 7160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_24

def innerTail35_22 : List Nat := 6823 :: innerTail35_23
def innerCandidates35_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates35_22_checked : candidates edge 4 163 6823 innerTail35_23 best.bound = innerCandidates35_22 := by
  unfold candidates
  have row : edge 6823 = SearchData.row6823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_22 : scan edge (search edge 3) 4 [163] 163 innerTail35_22 best = best := by
  rw [innerTail35_22, scan, if_neg (by decide), innerCandidates35_22_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_23
    (if 3 ≤ innerCandidates35_22.length then search edge 3 [163, 6823] innerCandidates35_22 6986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_23

def innerTail35_21 : List Nat := 6781 :: innerTail35_22
def innerCandidates35_21 : List Nat := [6823]
set_option maxRecDepth 4096 in
theorem innerCandidates35_21_checked : candidates edge 4 163 6781 innerTail35_22 best.bound = innerCandidates35_21 := by
  unfold candidates
  have row : edge 6781 = SearchData.row6781.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_21 : scan edge (search edge 3) 4 [163] 163 innerTail35_21 best = best := by
  rw [innerTail35_21, scan, if_neg (by decide), innerCandidates35_21_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_22
    (if 3 ≤ innerCandidates35_21.length then search edge 3 [163, 6781] innerCandidates35_21 6944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_22

def innerTail35_20 : List Nat := 6577 :: innerTail35_21
def innerCandidates35_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates35_20_checked : candidates edge 4 163 6577 innerTail35_21 best.bound = innerCandidates35_20 := by
  unfold candidates
  have row : edge 6577 = SearchData.row6577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_20 : scan edge (search edge 3) 4 [163] 163 innerTail35_20 best = best := by
  rw [innerTail35_20, scan, if_neg (by decide), innerCandidates35_20_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_21
    (if 3 ≤ innerCandidates35_20.length then search edge 3 [163, 6577] innerCandidates35_20 6740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_21

def innerTail35_19 : List Nat := 6343 :: innerTail35_20
def innerCandidates35_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates35_19_checked : candidates edge 4 163 6343 innerTail35_20 best.bound = innerCandidates35_19 := by
  unfold candidates
  have row : edge 6343 = SearchData.row6343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_19 : scan edge (search edge 3) 4 [163] 163 innerTail35_19 best = best := by
  rw [innerTail35_19, scan, if_neg (by decide), innerCandidates35_19_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_20
    (if 3 ≤ innerCandidates35_19.length then search edge 3 [163, 6343] innerCandidates35_19 6506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_20

def innerTail35_18 : List Nat := 5563 :: innerTail35_19
def innerCandidates35_18 : List Nat := [6781]
set_option maxRecDepth 4096 in
theorem innerCandidates35_18_checked : candidates edge 4 163 5563 innerTail35_19 best.bound = innerCandidates35_18 := by
  unfold candidates
  have row : edge 5563 = SearchData.row5563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_18 : scan edge (search edge 3) 4 [163] 163 innerTail35_18 best = best := by
  rw [innerTail35_18, scan, if_neg (by decide), innerCandidates35_18_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_19
    (if 3 ≤ innerCandidates35_18.length then search edge 3 [163, 5563] innerCandidates35_18 5726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_19

def innerTail35_17 : List Nat := 4597 :: innerTail35_18
def innerCandidates35_17 : List Nat := [5563]
set_option maxRecDepth 4096 in
theorem innerCandidates35_17_checked : candidates edge 4 163 4597 innerTail35_18 best.bound = innerCandidates35_17 := by
  unfold candidates
  have row : edge 4597 = SearchData.row4597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_17 : scan edge (search edge 3) 4 [163] 163 innerTail35_17 best = best := by
  rw [innerTail35_17, scan, if_neg (by decide), innerCandidates35_17_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_18
    (if 3 ≤ innerCandidates35_17.length then search edge 3 [163, 4597] innerCandidates35_17 4760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_18

def innerTail35_16 : List Nat := 4441 :: innerTail35_17
def innerCandidates35_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates35_16_checked : candidates edge 4 163 4441 innerTail35_17 best.bound = innerCandidates35_16 := by
  unfold candidates
  have row : edge 4441 = SearchData.row4441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_16 : scan edge (search edge 3) 4 [163] 163 innerTail35_16 best = best := by
  rw [innerTail35_16, scan, if_neg (by decide), innerCandidates35_16_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_17
    (if 3 ≤ innerCandidates35_16.length then search edge 3 [163, 4441] innerCandidates35_16 4604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_17

def innerTail35_15 : List Nat := 4177 :: innerTail35_16
def innerCandidates35_15 : List Nat := [6781]
set_option maxRecDepth 4096 in
theorem innerCandidates35_15_checked : candidates edge 4 163 4177 innerTail35_16 best.bound = innerCandidates35_15 := by
  unfold candidates
  have row : edge 4177 = SearchData.row4177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_15 : scan edge (search edge 3) 4 [163] 163 innerTail35_15 best = best := by
  rw [innerTail35_15, scan, if_neg (by decide), innerCandidates35_15_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_16
    (if 3 ≤ innerCandidates35_15.length then search edge 3 [163, 4177] innerCandidates35_15 4340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_16

def innerTail35_14 : List Nat := 3823 :: innerTail35_15
def innerCandidates35_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates35_14_checked : candidates edge 4 163 3823 innerTail35_15 best.bound = innerCandidates35_14 := by
  unfold candidates
  have row : edge 3823 = SearchData.row3823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_14 : scan edge (search edge 3) 4 [163] 163 innerTail35_14 best = best := by
  rw [innerTail35_14, scan, if_neg (by decide), innerCandidates35_14_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_15
    (if 3 ≤ innerCandidates35_14.length then search edge 3 [163, 3823] innerCandidates35_14 3986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_15

def innerTail35_13 : List Nat := 2647 :: innerTail35_14
def innerCandidates35_13 : List Nat := [6577]
set_option maxRecDepth 4096 in
theorem innerCandidates35_13_checked : candidates edge 4 163 2647 innerTail35_14 best.bound = innerCandidates35_13 := by
  unfold candidates
  have row : edge 2647 = PrimePairCache.smallRow2647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_13 : scan edge (search edge 3) 4 [163] 163 innerTail35_13 best = best := by
  rw [innerTail35_13, scan, if_neg (by decide), innerCandidates35_13_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_14
    (if 3 ≤ innerCandidates35_13.length then search edge 3 [163, 2647] innerCandidates35_13 2810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_14

def innerTail35_12 : List Nat := 2383 :: innerTail35_13
def innerCandidates35_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates35_12_checked : candidates edge 4 163 2383 innerTail35_13 best.bound = innerCandidates35_12 := by
  unfold candidates
  have row : edge 2383 = PrimePairCache.smallRow2383.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_12 : scan edge (search edge 3) 4 [163] 163 innerTail35_12 best = best := by
  rw [innerTail35_12, scan, if_neg (by decide), innerCandidates35_12_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_13
    (if 3 ≤ innerCandidates35_12.length then search edge 3 [163, 2383] innerCandidates35_12 2546 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_13

def innerTail35_11 : List Nat := 2311 :: innerTail35_12
def innerCandidates35_11 : List Nat := [2647]
set_option maxRecDepth 4096 in
theorem innerCandidates35_11_checked : candidates edge 4 163 2311 innerTail35_12 best.bound = innerCandidates35_11 := by
  unfold candidates
  have row : edge 2311 = PrimePairCache.smallRow2311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_11 : scan edge (search edge 3) 4 [163] 163 innerTail35_11 best = best := by
  rw [innerTail35_11, scan, if_neg (by decide), innerCandidates35_11_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_12
    (if 3 ≤ innerCandidates35_11.length then search edge 3 [163, 2311] innerCandidates35_11 2474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_12

def innerTail35_10 : List Nat := 1783 :: innerTail35_11
def innerCandidates35_10 : List Nat := [6823]
set_option maxRecDepth 4096 in
theorem innerCandidates35_10_checked : candidates edge 4 163 1783 innerTail35_11 best.bound = innerCandidates35_10 := by
  unfold candidates
  have row : edge 1783 = PrimePairCache.smallRow1783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_10 : scan edge (search edge 3) 4 [163] 163 innerTail35_10 best = best := by
  rw [innerTail35_10, scan, if_neg (by decide), innerCandidates35_10_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_11
    (if 3 ≤ innerCandidates35_10.length then search edge 3 [163, 1783] innerCandidates35_10 1946 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_11

def innerTail35_9 : List Nat := 1741 :: innerTail35_10
def innerCandidates35_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates35_9_checked : candidates edge 4 163 1741 innerTail35_10 best.bound = innerCandidates35_9 := by
  unfold candidates
  have row : edge 1741 = PrimePairCache.smallRow1741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_9 : scan edge (search edge 3) 4 [163] 163 innerTail35_9 best = best := by
  rw [innerTail35_9, scan, if_neg (by decide), innerCandidates35_9_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_10
    (if 3 ≤ innerCandidates35_9.length then search edge 3 [163, 1741] innerCandidates35_9 1904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_10

def innerTail35_8 : List Nat := 1579 :: innerTail35_9
def innerCandidates35_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates35_8_checked : candidates edge 4 163 1579 innerTail35_9 best.bound = innerCandidates35_8 := by
  unfold candidates
  have row : edge 1579 = PrimePairCache.smallRow1579.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_8 : scan edge (search edge 3) 4 [163] 163 innerTail35_8 best = best := by
  rw [innerTail35_8, scan, if_neg (by decide), innerCandidates35_8_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_9
    (if 3 ≤ innerCandidates35_8.length then search edge 3 [163, 1579] innerCandidates35_8 1742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_9

def innerTail35_7 : List Nat := 997 :: innerTail35_8
def innerCandidates35_7 : List Nat := [4177]
set_option maxRecDepth 4096 in
theorem innerCandidates35_7_checked : candidates edge 4 163 997 innerTail35_8 best.bound = innerCandidates35_7 := by
  unfold candidates
  have row : edge 997 = PrimePairCache.smallRow997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_7 : scan edge (search edge 3) 4 [163] 163 innerTail35_7 best = best := by
  rw [innerTail35_7, scan, if_neg (by decide), innerCandidates35_7_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_8
    (if 3 ≤ innerCandidates35_7.length then search edge 3 [163, 997] innerCandidates35_7 1160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_8

def innerTail35_6 : List Nat := 883 :: innerTail35_7
def innerCandidates35_6 : List Nat := [1783]
set_option maxRecDepth 4096 in
theorem innerCandidates35_6_checked : candidates edge 4 163 883 innerTail35_7 best.bound = innerCandidates35_6 := by
  unfold candidates
  have row : edge 883 = PrimePairCache.smallRow883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_6 : scan edge (search edge 3) 4 [163] 163 innerTail35_6 best = best := by
  rw [innerTail35_6, scan, if_neg (by decide), innerCandidates35_6_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_7
    (if 3 ≤ innerCandidates35_6.length then search edge 3 [163, 883] innerCandidates35_6 1046 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_7

def innerTail35_5 : List Nat := 811 :: innerTail35_6
def innerCandidates35_5 : List Nat := [997]
set_option maxRecDepth 4096 in
theorem innerCandidates35_5_checked : candidates edge 4 163 811 innerTail35_6 best.bound = innerCandidates35_5 := by
  unfold candidates
  have row : edge 811 = PrimePairCache.smallRow811.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_5 : scan edge (search edge 3) 4 [163] 163 innerTail35_5 best = best := by
  rw [innerTail35_5, scan, if_neg (by decide), innerCandidates35_5_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_6
    (if 3 ≤ innerCandidates35_5.length then search edge 3 [163, 811] innerCandidates35_5 974 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_6

def innerTail35_4 : List Nat := 613 :: innerTail35_5
def innerCandidates35_4 : List Nat := [883]
set_option maxRecDepth 4096 in
theorem innerCandidates35_4_checked : candidates edge 4 163 613 innerTail35_5 best.bound = innerCandidates35_4 := by
  unfold candidates
  have row : edge 613 = PrimePairCache.smallRow613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_4 : scan edge (search edge 3) 4 [163] 163 innerTail35_4 best = best := by
  rw [innerTail35_4, scan, if_neg (by decide), innerCandidates35_4_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_5
    (if 3 ≤ innerCandidates35_4.length then search edge 3 [163, 613] innerCandidates35_4 776 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_5

def innerTail35_3 : List Nat := 409 :: innerTail35_4
def innerCandidates35_3 : List Nat := [1741]
set_option maxRecDepth 4096 in
theorem innerCandidates35_3_checked : candidates edge 4 163 409 innerTail35_4 best.bound = innerCandidates35_3 := by
  unfold candidates
  have row : edge 409 = PrimePairCache.smallRow409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_3 : scan edge (search edge 3) 4 [163] 163 innerTail35_3 best = best := by
  rw [innerTail35_3, scan, if_neg (by decide), innerCandidates35_3_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_4
    (if 3 ≤ innerCandidates35_3.length then search edge 3 [163, 409] innerCandidates35_3 572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_4

def innerTail35_2 : List Nat := 367 :: innerTail35_3
def innerCandidates35_2 : List Nat := [613, 4597, 5563]
set_option maxRecDepth 4096 in
theorem innerCandidates35_2_checked : candidates edge 4 163 367 innerTail35_3 best.bound = innerCandidates35_2 := by
  unfold candidates
  have row : edge 367 = PrimePairCache.smallRow367.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild35_2 : search edge 3 [163, 367] innerCandidates35_2 530 best = best := by decide

theorem innerScan35_2 : scan edge (search edge 3) 4 [163] 163 innerTail35_2 best = best := by
  rw [innerTail35_2, scan, if_neg (by decide), innerCandidates35_2_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_3
    (if 3 ≤ innerCandidates35_2.length then search edge 3 [163, 367] innerCandidates35_2 530 best else best) = best
  rw [if_pos (by decide), innerChild35_2]
  exact innerScan35_3

def innerTail35_1 : List Nat := 307 :: innerTail35_2
def innerCandidates35_1 : List Nat := [367, 7927]
set_option maxRecDepth 4096 in
theorem innerCandidates35_1_checked : candidates edge 4 163 307 innerTail35_2 best.bound = innerCandidates35_1 := by
  unfold candidates
  have row : edge 307 = PrimePairCache.smallRow307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_1 : scan edge (search edge 3) 4 [163] 163 innerTail35_1 best = best := by
  rw [innerTail35_1, scan, if_neg (by decide), innerCandidates35_1_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_2
    (if 3 ≤ innerCandidates35_1.length then search edge 3 [163, 307] innerCandidates35_1 470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_2

def innerTail35_0 : List Nat := 193 :: innerTail35_1
def innerCandidates35_0 : List Nat := [811, 883]
set_option maxRecDepth 4096 in
theorem innerCandidates35_0_checked : candidates edge 4 163 193 innerTail35_1 best.bound = innerCandidates35_0 := by
  unfold candidates
  have row : edge 193 = PrimePairCache.smallRow193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan35_0 : scan edge (search edge 3) 4 [163] 163 innerTail35_0 best = best := by
  rw [innerTail35_0, scan, if_neg (by decide), innerCandidates35_0_checked]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_1
    (if 3 ≤ innerCandidates35_0.length then search edge 3 [163, 193] innerCandidates35_0 356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan35_1

theorem rootBranch35 : search edge 4 [163] rootCandidates35 163 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [163] 163 innerTail35_0 best = best
  exact innerScan35_0

def rootCandidates34 : List Nat := [181, 229, 277, 571, 733, 1657, 1663, 2239, 2377, 2539, 2647, 2713, 2749, 3079, 3109, 3217, 3613, 3709, 4129, 5653, 6073, 6661, 6907, 6997, 7309, 7561, 7699]
set_option maxRecDepth 4096 in
theorem rootCandidates34_checked : candidates edge 5 0 157 tail35 best.bound = rootCandidates34 := by
  unfold candidates
  have row : edge 157 = PrimePairCache.smallRow157.contains := by funext q; rfl
  rw [row]
  decide

def innerTail34_27 : List Nat := []
theorem innerScan34_27 : scan edge (search edge 3) 4 [157] 157 innerTail34_27 best = best := rfl

def innerTail34_26 : List Nat := 7699 :: innerTail34_27
def innerCandidates34_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates34_26_checked : candidates edge 4 157 7699 innerTail34_27 best.bound = innerCandidates34_26 := by
  unfold candidates
  have row : edge 7699 = SearchData.row7699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_26 : scan edge (search edge 3) 4 [157] 157 innerTail34_26 best = best := by
  rw [innerTail34_26, scan, if_neg (by decide), innerCandidates34_26_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_27
    (if 3 ≤ innerCandidates34_26.length then search edge 3 [157, 7699] innerCandidates34_26 7856 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_27

def innerTail34_25 : List Nat := 7561 :: innerTail34_26
def innerCandidates34_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates34_25_checked : candidates edge 4 157 7561 innerTail34_26 best.bound = innerCandidates34_25 := by
  unfold candidates
  have row : edge 7561 = SearchData.row7561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_25 : scan edge (search edge 3) 4 [157] 157 innerTail34_25 best = best := by
  rw [innerTail34_25, scan, if_neg (by decide), innerCandidates34_25_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_26
    (if 3 ≤ innerCandidates34_25.length then search edge 3 [157, 7561] innerCandidates34_25 7718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_26

def innerTail34_24 : List Nat := 7309 :: innerTail34_25
def innerCandidates34_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates34_24_checked : candidates edge 4 157 7309 innerTail34_25 best.bound = innerCandidates34_24 := by
  unfold candidates
  have row : edge 7309 = SearchData.row7309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_24 : scan edge (search edge 3) 4 [157] 157 innerTail34_24 best = best := by
  rw [innerTail34_24, scan, if_neg (by decide), innerCandidates34_24_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_25
    (if 3 ≤ innerCandidates34_24.length then search edge 3 [157, 7309] innerCandidates34_24 7466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_25

def innerTail34_23 : List Nat := 6997 :: innerTail34_24
def innerCandidates34_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates34_23_checked : candidates edge 4 157 6997 innerTail34_24 best.bound = innerCandidates34_23 := by
  unfold candidates
  have row : edge 6997 = SearchData.row6997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_23 : scan edge (search edge 3) 4 [157] 157 innerTail34_23 best = best := by
  rw [innerTail34_23, scan, if_neg (by decide), innerCandidates34_23_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_24
    (if 3 ≤ innerCandidates34_23.length then search edge 3 [157, 6997] innerCandidates34_23 7154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_24

def innerTail34_22 : List Nat := 6907 :: innerTail34_23
def innerCandidates34_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates34_22_checked : candidates edge 4 157 6907 innerTail34_23 best.bound = innerCandidates34_22 := by
  unfold candidates
  have row : edge 6907 = SearchData.row6907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_22 : scan edge (search edge 3) 4 [157] 157 innerTail34_22 best = best := by
  rw [innerTail34_22, scan, if_neg (by decide), innerCandidates34_22_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_23
    (if 3 ≤ innerCandidates34_22.length then search edge 3 [157, 6907] innerCandidates34_22 7064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_23

def innerTail34_21 : List Nat := 6661 :: innerTail34_22
def innerCandidates34_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates34_21_checked : candidates edge 4 157 6661 innerTail34_22 best.bound = innerCandidates34_21 := by
  unfold candidates
  have row : edge 6661 = SearchData.row6661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_21 : scan edge (search edge 3) 4 [157] 157 innerTail34_21 best = best := by
  rw [innerTail34_21, scan, if_neg (by decide), innerCandidates34_21_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_22
    (if 3 ≤ innerCandidates34_21.length then search edge 3 [157, 6661] innerCandidates34_21 6818 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_22

def innerTail34_20 : List Nat := 6073 :: innerTail34_21
def innerCandidates34_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates34_20_checked : candidates edge 4 157 6073 innerTail34_21 best.bound = innerCandidates34_20 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_20 : scan edge (search edge 3) 4 [157] 157 innerTail34_20 best = best := by
  rw [innerTail34_20, scan, if_neg (by decide), innerCandidates34_20_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_21
    (if 3 ≤ innerCandidates34_20.length then search edge 3 [157, 6073] innerCandidates34_20 6230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_21

def innerTail34_19 : List Nat := 5653 :: innerTail34_20
def innerCandidates34_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates34_19_checked : candidates edge 4 157 5653 innerTail34_20 best.bound = innerCandidates34_19 := by
  unfold candidates
  have row : edge 5653 = SearchData.row5653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_19 : scan edge (search edge 3) 4 [157] 157 innerTail34_19 best = best := by
  rw [innerTail34_19, scan, if_neg (by decide), innerCandidates34_19_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_20
    (if 3 ≤ innerCandidates34_19.length then search edge 3 [157, 5653] innerCandidates34_19 5810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_20

def innerTail34_18 : List Nat := 4129 :: innerTail34_19
def innerCandidates34_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates34_18_checked : candidates edge 4 157 4129 innerTail34_19 best.bound = innerCandidates34_18 := by
  unfold candidates
  have row : edge 4129 = SearchData.row4129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_18 : scan edge (search edge 3) 4 [157] 157 innerTail34_18 best = best := by
  rw [innerTail34_18, scan, if_neg (by decide), innerCandidates34_18_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_19
    (if 3 ≤ innerCandidates34_18.length then search edge 3 [157, 4129] innerCandidates34_18 4286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_19

def innerTail34_17 : List Nat := 3709 :: innerTail34_18
def innerCandidates34_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates34_17_checked : candidates edge 4 157 3709 innerTail34_18 best.bound = innerCandidates34_17 := by
  unfold candidates
  have row : edge 3709 = SearchData.row3709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_17 : scan edge (search edge 3) 4 [157] 157 innerTail34_17 best = best := by
  rw [innerTail34_17, scan, if_neg (by decide), innerCandidates34_17_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_18
    (if 3 ≤ innerCandidates34_17.length then search edge 3 [157, 3709] innerCandidates34_17 3866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_18

def innerTail34_16 : List Nat := 3613 :: innerTail34_17
def innerCandidates34_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates34_16_checked : candidates edge 4 157 3613 innerTail34_17 best.bound = innerCandidates34_16 := by
  unfold candidates
  have row : edge 3613 = SearchData.row3613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_16 : scan edge (search edge 3) 4 [157] 157 innerTail34_16 best = best := by
  rw [innerTail34_16, scan, if_neg (by decide), innerCandidates34_16_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_17
    (if 3 ≤ innerCandidates34_16.length then search edge 3 [157, 3613] innerCandidates34_16 3770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_17

def innerTail34_15 : List Nat := 3217 :: innerTail34_16
def innerCandidates34_15 : List Nat := [6073]
set_option maxRecDepth 4096 in
theorem innerCandidates34_15_checked : candidates edge 4 157 3217 innerTail34_16 best.bound = innerCandidates34_15 := by
  unfold candidates
  have row : edge 3217 = SearchData.row3217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_15 : scan edge (search edge 3) 4 [157] 157 innerTail34_15 best = best := by
  rw [innerTail34_15, scan, if_neg (by decide), innerCandidates34_15_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_16
    (if 3 ≤ innerCandidates34_15.length then search edge 3 [157, 3217] innerCandidates34_15 3374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_16

def innerTail34_14 : List Nat := 3109 :: innerTail34_15
def innerCandidates34_14 : List Nat := [6073, 6997]
set_option maxRecDepth 4096 in
theorem innerCandidates34_14_checked : candidates edge 4 157 3109 innerTail34_15 best.bound = innerCandidates34_14 := by
  unfold candidates
  have row : edge 3109 = SearchData.row3109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_14 : scan edge (search edge 3) 4 [157] 157 innerTail34_14 best = best := by
  rw [innerTail34_14, scan, if_neg (by decide), innerCandidates34_14_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_15
    (if 3 ≤ innerCandidates34_14.length then search edge 3 [157, 3109] innerCandidates34_14 3266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_15

def innerTail34_13 : List Nat := 3079 :: innerTail34_14
def innerCandidates34_13 : List Nat := [3709, 7561, 7699]
set_option maxRecDepth 4096 in
theorem innerCandidates34_13_checked : candidates edge 4 157 3079 innerTail34_14 best.bound = innerCandidates34_13 := by
  unfold candidates
  have row : edge 3079 = SearchData.row3079.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild34_13 : search edge 3 [157, 3079] innerCandidates34_13 3236 best = best := by decide

theorem innerScan34_13 : scan edge (search edge 3) 4 [157] 157 innerTail34_13 best = best := by
  rw [innerTail34_13, scan, if_neg (by decide), innerCandidates34_13_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_14
    (if 3 ≤ innerCandidates34_13.length then search edge 3 [157, 3079] innerCandidates34_13 3236 best else best) = best
  rw [if_pos (by decide), innerChild34_13]
  exact innerScan34_14

def innerTail34_12 : List Nat := 2749 :: innerTail34_13
def innerCandidates34_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates34_12_checked : candidates edge 4 157 2749 innerTail34_13 best.bound = innerCandidates34_12 := by
  unfold candidates
  have row : edge 2749 = SearchData.row2749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_12 : scan edge (search edge 3) 4 [157] 157 innerTail34_12 best = best := by
  rw [innerTail34_12, scan, if_neg (by decide), innerCandidates34_12_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_13
    (if 3 ≤ innerCandidates34_12.length then search edge 3 [157, 2749] innerCandidates34_12 2906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_13

def innerTail34_11 : List Nat := 2713 :: innerTail34_12
def innerCandidates34_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates34_11_checked : candidates edge 4 157 2713 innerTail34_12 best.bound = innerCandidates34_11 := by
  unfold candidates
  have row : edge 2713 = SearchData.row2713.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_11 : scan edge (search edge 3) 4 [157] 157 innerTail34_11 best = best := by
  rw [innerTail34_11, scan, if_neg (by decide), innerCandidates34_11_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_12
    (if 3 ≤ innerCandidates34_11.length then search edge 3 [157, 2713] innerCandidates34_11 2870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_12

def innerTail34_10 : List Nat := 2647 :: innerTail34_11
def innerCandidates34_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates34_10_checked : candidates edge 4 157 2647 innerTail34_11 best.bound = innerCandidates34_10 := by
  unfold candidates
  have row : edge 2647 = PrimePairCache.smallRow2647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_10 : scan edge (search edge 3) 4 [157] 157 innerTail34_10 best = best := by
  rw [innerTail34_10, scan, if_neg (by decide), innerCandidates34_10_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_11
    (if 3 ≤ innerCandidates34_10.length then search edge 3 [157, 2647] innerCandidates34_10 2804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_11

def innerTail34_9 : List Nat := 2539 :: innerTail34_10
def innerCandidates34_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates34_9_checked : candidates edge 4 157 2539 innerTail34_10 best.bound = innerCandidates34_9 := by
  unfold candidates
  have row : edge 2539 = PrimePairCache.smallRow2539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_9 : scan edge (search edge 3) 4 [157] 157 innerTail34_9 best = best := by
  rw [innerTail34_9, scan, if_neg (by decide), innerCandidates34_9_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_10
    (if 3 ≤ innerCandidates34_9.length then search edge 3 [157, 2539] innerCandidates34_9 2696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_10

def innerTail34_8 : List Nat := 2377 :: innerTail34_9
def innerCandidates34_8 : List Nat := [6073]
set_option maxRecDepth 4096 in
theorem innerCandidates34_8_checked : candidates edge 4 157 2377 innerTail34_9 best.bound = innerCandidates34_8 := by
  unfold candidates
  have row : edge 2377 = PrimePairCache.smallRow2377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_8 : scan edge (search edge 3) 4 [157] 157 innerTail34_8 best = best := by
  rw [innerTail34_8, scan, if_neg (by decide), innerCandidates34_8_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_9
    (if 3 ≤ innerCandidates34_8.length then search edge 3 [157, 2377] innerCandidates34_8 2534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_9

def innerTail34_7 : List Nat := 2239 :: innerTail34_8
def innerCandidates34_7 : List Nat := [4129]
set_option maxRecDepth 4096 in
theorem innerCandidates34_7_checked : candidates edge 4 157 2239 innerTail34_8 best.bound = innerCandidates34_7 := by
  unfold candidates
  have row : edge 2239 = PrimePairCache.smallRow2239.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_7 : scan edge (search edge 3) 4 [157] 157 innerTail34_7 best = best := by
  rw [innerTail34_7, scan, if_neg (by decide), innerCandidates34_7_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_8
    (if 3 ≤ innerCandidates34_7.length then search edge 3 [157, 2239] innerCandidates34_7 2396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_8

def innerTail34_6 : List Nat := 1663 :: innerTail34_7
def innerCandidates34_6 : List Nat := [3613, 5653]
set_option maxRecDepth 4096 in
theorem innerCandidates34_6_checked : candidates edge 4 157 1663 innerTail34_7 best.bound = innerCandidates34_6 := by
  unfold candidates
  have row : edge 1663 = PrimePairCache.smallRow1663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_6 : scan edge (search edge 3) 4 [157] 157 innerTail34_6 best = best := by
  rw [innerTail34_6, scan, if_neg (by decide), innerCandidates34_6_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_7
    (if 3 ≤ innerCandidates34_6.length then search edge 3 [157, 1663] innerCandidates34_6 1820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_7

def innerTail34_5 : List Nat := 1657 :: innerTail34_6
def innerCandidates34_5 : List Nat := [6907]
set_option maxRecDepth 4096 in
theorem innerCandidates34_5_checked : candidates edge 4 157 1657 innerTail34_6 best.bound = innerCandidates34_5 := by
  unfold candidates
  have row : edge 1657 = PrimePairCache.smallRow1657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_5 : scan edge (search edge 3) 4 [157] 157 innerTail34_5 best = best := by
  rw [innerTail34_5, scan, if_neg (by decide), innerCandidates34_5_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_6
    (if 3 ≤ innerCandidates34_5.length then search edge 3 [157, 1657] innerCandidates34_5 1814 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_6

def innerTail34_4 : List Nat := 733 :: innerTail34_5
def innerCandidates34_4 : List Nat := [3709]
set_option maxRecDepth 4096 in
theorem innerCandidates34_4_checked : candidates edge 4 157 733 innerTail34_5 best.bound = innerCandidates34_4 := by
  unfold candidates
  have row : edge 733 = PrimePairCache.smallRow733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_4 : scan edge (search edge 3) 4 [157] 157 innerTail34_4 best = best := by
  rw [innerTail34_4, scan, if_neg (by decide), innerCandidates34_4_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_5
    (if 3 ≤ innerCandidates34_4.length then search edge 3 [157, 733] innerCandidates34_4 890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_5

def innerTail34_3 : List Nat := 571 :: innerTail34_4
def innerCandidates34_3 : List Nat := [2647]
set_option maxRecDepth 4096 in
theorem innerCandidates34_3_checked : candidates edge 4 157 571 innerTail34_4 best.bound = innerCandidates34_3 := by
  unfold candidates
  have row : edge 571 = PrimePairCache.smallRow571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_3 : scan edge (search edge 3) 4 [157] 157 innerTail34_3 best = best := by
  rw [innerTail34_3, scan, if_neg (by decide), innerCandidates34_3_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_4
    (if 3 ≤ innerCandidates34_3.length then search edge 3 [157, 571] innerCandidates34_3 728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_4

def innerTail34_2 : List Nat := 277 :: innerTail34_3
def innerCandidates34_2 : List Nat := [3079]
set_option maxRecDepth 4096 in
theorem innerCandidates34_2_checked : candidates edge 4 157 277 innerTail34_3 best.bound = innerCandidates34_2 := by
  unfold candidates
  have row : edge 277 = PrimePairCache.smallRow277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_2 : scan edge (search edge 3) 4 [157] 157 innerTail34_2 best = best := by
  rw [innerTail34_2, scan, if_neg (by decide), innerCandidates34_2_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_3
    (if 3 ≤ innerCandidates34_2.length then search edge 3 [157, 277] innerCandidates34_2 434 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_3

def innerTail34_1 : List Nat := 229 :: innerTail34_2
def innerCandidates34_1 : List Nat := [2239]
set_option maxRecDepth 4096 in
theorem innerCandidates34_1_checked : candidates edge 4 157 229 innerTail34_2 best.bound = innerCandidates34_1 := by
  unfold candidates
  have row : edge 229 = PrimePairCache.smallRow229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_1 : scan edge (search edge 3) 4 [157] 157 innerTail34_1 best = best := by
  rw [innerTail34_1, scan, if_neg (by decide), innerCandidates34_1_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_2
    (if 3 ≤ innerCandidates34_1.length then search edge 3 [157, 229] innerCandidates34_1 386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_2

def innerTail34_0 : List Nat := 181 :: innerTail34_1
def innerCandidates34_0 : List Nat := [1657]
set_option maxRecDepth 4096 in
theorem innerCandidates34_0_checked : candidates edge 4 157 181 innerTail34_1 best.bound = innerCandidates34_0 := by
  unfold candidates
  have row : edge 181 = PrimePairCache.smallRow181.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan34_0 : scan edge (search edge 3) 4 [157] 157 innerTail34_0 best = best := by
  rw [innerTail34_0, scan, if_neg (by decide), innerCandidates34_0_checked]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_1
    (if 3 ≤ innerCandidates34_0.length then search edge 3 [157, 181] innerCandidates34_0 338 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan34_1

theorem rootBranch34 : search edge 4 [157] rootCandidates34 157 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [157] 157 innerTail34_0 best = best
  exact innerScan34_0

end ProjectEulerSolutions.PrimePairUpper.Search
