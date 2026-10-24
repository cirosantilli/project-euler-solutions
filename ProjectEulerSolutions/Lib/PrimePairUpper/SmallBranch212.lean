import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates78 : List Nat := [443, 449, 563, 599, 701, 929, 1097, 1373, 2087, 2129, 3209, 3221, 3803, 4583, 5273, 5297, 5441, 7253, 7757, 7883, 8111]
set_option maxRecDepth 4096 in
theorem rootCandidates78_checked : candidates edge 5 0 419 tail79 best.bound = rootCandidates78 := by
  unfold candidates
  have row : edge 419 = PrimePairCache.smallRow419.contains := by funext q; rfl
  rw [row]
  decide

def innerTail78_21 : List Nat := []
theorem innerScan78_21 : scan edge (search edge 3) 4 [419] 419 innerTail78_21 best = best := rfl

def innerTail78_20 : List Nat := 8111 :: innerTail78_21
def innerCandidates78_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates78_20_checked : candidates edge 4 419 8111 innerTail78_21 best.bound = innerCandidates78_20 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan78_20 : scan edge (search edge 3) 4 [419] 419 innerTail78_20 best = best := by
  rw [innerTail78_20, scan, if_neg (by decide), innerCandidates78_20_checked]
  change scan edge (search edge 3) 4 [419] 419 innerTail78_21
    (if 3 ≤ innerCandidates78_20.length then search edge 3 [419, 8111] innerCandidates78_20 8530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan78_21

def innerTail78_19 : List Nat := 7883 :: innerTail78_20
def innerCandidates78_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates78_19_checked : candidates edge 4 419 7883 innerTail78_20 best.bound = innerCandidates78_19 := by
  unfold candidates
  have row : edge 7883 = SearchData.row7883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan78_19 : scan edge (search edge 3) 4 [419] 419 innerTail78_19 best = best := by
  rw [innerTail78_19, scan, if_neg (by decide), innerCandidates78_19_checked]
  change scan edge (search edge 3) 4 [419] 419 innerTail78_20
    (if 3 ≤ innerCandidates78_19.length then search edge 3 [419, 7883] innerCandidates78_19 8302 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan78_20

def innerTail78_18 : List Nat := 7757 :: innerTail78_19
def innerCandidates78_18 : List Nat := [7883]
set_option maxRecDepth 4096 in
theorem innerCandidates78_18_checked : candidates edge 4 419 7757 innerTail78_19 best.bound = innerCandidates78_18 := by
  unfold candidates
  have row : edge 7757 = SearchData.row7757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan78_18 : scan edge (search edge 3) 4 [419] 419 innerTail78_18 best = best := by
  rw [innerTail78_18, scan, if_neg (by decide), innerCandidates78_18_checked]
  change scan edge (search edge 3) 4 [419] 419 innerTail78_19
    (if 3 ≤ innerCandidates78_18.length then search edge 3 [419, 7757] innerCandidates78_18 8176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan78_19

def innerTail78_17 : List Nat := 7253 :: innerTail78_18
def innerCandidates78_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates78_17_checked : candidates edge 4 419 7253 innerTail78_18 best.bound = innerCandidates78_17 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan78_17 : scan edge (search edge 3) 4 [419] 419 innerTail78_17 best = best := by
  rw [innerTail78_17, scan, if_neg (by decide), innerCandidates78_17_checked]
  change scan edge (search edge 3) 4 [419] 419 innerTail78_18
    (if 3 ≤ innerCandidates78_17.length then search edge 3 [419, 7253] innerCandidates78_17 7672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan78_18

def innerTail78_16 : List Nat := 5441 :: innerTail78_17
def innerCandidates78_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates78_16_checked : candidates edge 4 419 5441 innerTail78_17 best.bound = innerCandidates78_16 := by
  unfold candidates
  have row : edge 5441 = SearchData.row5441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan78_16 : scan edge (search edge 3) 4 [419] 419 innerTail78_16 best = best := by
  rw [innerTail78_16, scan, if_neg (by decide), innerCandidates78_16_checked]
  change scan edge (search edge 3) 4 [419] 419 innerTail78_17
    (if 3 ≤ innerCandidates78_16.length then search edge 3 [419, 5441] innerCandidates78_16 5860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan78_17

def innerTail78_15 : List Nat := 5297 :: innerTail78_16
def innerCandidates78_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates78_15_checked : candidates edge 4 419 5297 innerTail78_16 best.bound = innerCandidates78_15 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan78_15 : scan edge (search edge 3) 4 [419] 419 innerTail78_15 best = best := by
  rw [innerTail78_15, scan, if_neg (by decide), innerCandidates78_15_checked]
  change scan edge (search edge 3) 4 [419] 419 innerTail78_16
    (if 3 ≤ innerCandidates78_15.length then search edge 3 [419, 5297] innerCandidates78_15 5716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan78_16

def innerTail78_14 : List Nat := 5273 :: innerTail78_15
def innerCandidates78_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates78_14_checked : candidates edge 4 419 5273 innerTail78_15 best.bound = innerCandidates78_14 := by
  unfold candidates
  have row : edge 5273 = SearchData.row5273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan78_14 : scan edge (search edge 3) 4 [419] 419 innerTail78_14 best = best := by
  rw [innerTail78_14, scan, if_neg (by decide), innerCandidates78_14_checked]
  change scan edge (search edge 3) 4 [419] 419 innerTail78_15
    (if 3 ≤ innerCandidates78_14.length then search edge 3 [419, 5273] innerCandidates78_14 5692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan78_15

def innerTail78_13 : List Nat := 4583 :: innerTail78_14
def innerCandidates78_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates78_13_checked : candidates edge 4 419 4583 innerTail78_14 best.bound = innerCandidates78_13 := by
  unfold candidates
  have row : edge 4583 = SearchData.row4583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan78_13 : scan edge (search edge 3) 4 [419] 419 innerTail78_13 best = best := by
  rw [innerTail78_13, scan, if_neg (by decide), innerCandidates78_13_checked]
  change scan edge (search edge 3) 4 [419] 419 innerTail78_14
    (if 3 ≤ innerCandidates78_13.length then search edge 3 [419, 4583] innerCandidates78_13 5002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan78_14

def innerTail78_12 : List Nat := 3803 :: innerTail78_13
def innerCandidates78_12 : List Nat := [7253]
set_option maxRecDepth 4096 in
theorem innerCandidates78_12_checked : candidates edge 4 419 3803 innerTail78_13 best.bound = innerCandidates78_12 := by
  unfold candidates
  have row : edge 3803 = SearchData.row3803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan78_12 : scan edge (search edge 3) 4 [419] 419 innerTail78_12 best = best := by
  rw [innerTail78_12, scan, if_neg (by decide), innerCandidates78_12_checked]
  change scan edge (search edge 3) 4 [419] 419 innerTail78_13
    (if 3 ≤ innerCandidates78_12.length then search edge 3 [419, 3803] innerCandidates78_12 4222 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan78_13

def innerTail78_11 : List Nat := 3221 :: innerTail78_12
def innerCandidates78_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates78_11_checked : candidates edge 4 419 3221 innerTail78_12 best.bound = innerCandidates78_11 := by
  unfold candidates
  have row : edge 3221 = SearchData.row3221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan78_11 : scan edge (search edge 3) 4 [419] 419 innerTail78_11 best = best := by
  rw [innerTail78_11, scan, if_neg (by decide), innerCandidates78_11_checked]
  change scan edge (search edge 3) 4 [419] 419 innerTail78_12
    (if 3 ≤ innerCandidates78_11.length then search edge 3 [419, 3221] innerCandidates78_11 3640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan78_12

def innerTail78_10 : List Nat := 3209 :: innerTail78_11
def innerCandidates78_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates78_10_checked : candidates edge 4 419 3209 innerTail78_11 best.bound = innerCandidates78_10 := by
  unfold candidates
  have row : edge 3209 = SearchData.row3209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan78_10 : scan edge (search edge 3) 4 [419] 419 innerTail78_10 best = best := by
  rw [innerTail78_10, scan, if_neg (by decide), innerCandidates78_10_checked]
  change scan edge (search edge 3) 4 [419] 419 innerTail78_11
    (if 3 ≤ innerCandidates78_10.length then search edge 3 [419, 3209] innerCandidates78_10 3628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan78_11

def innerTail78_9 : List Nat := 2129 :: innerTail78_10
def innerCandidates78_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates78_9_checked : candidates edge 4 419 2129 innerTail78_10 best.bound = innerCandidates78_9 := by
  unfold candidates
  have row : edge 2129 = PrimePairCache.smallRow2129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan78_9 : scan edge (search edge 3) 4 [419] 419 innerTail78_9 best = best := by
  rw [innerTail78_9, scan, if_neg (by decide), innerCandidates78_9_checked]
  change scan edge (search edge 3) 4 [419] 419 innerTail78_10
    (if 3 ≤ innerCandidates78_9.length then search edge 3 [419, 2129] innerCandidates78_9 2548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan78_10

def innerTail78_8 : List Nat := 2087 :: innerTail78_9
def innerCandidates78_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates78_8_checked : candidates edge 4 419 2087 innerTail78_9 best.bound = innerCandidates78_8 := by
  unfold candidates
  have row : edge 2087 = PrimePairCache.smallRow2087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan78_8 : scan edge (search edge 3) 4 [419] 419 innerTail78_8 best = best := by
  rw [innerTail78_8, scan, if_neg (by decide), innerCandidates78_8_checked]
  change scan edge (search edge 3) 4 [419] 419 innerTail78_9
    (if 3 ≤ innerCandidates78_8.length then search edge 3 [419, 2087] innerCandidates78_8 2506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan78_9

def innerTail78_7 : List Nat := 1373 :: innerTail78_8
def innerCandidates78_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates78_7_checked : candidates edge 4 419 1373 innerTail78_8 best.bound = innerCandidates78_7 := by
  unfold candidates
  have row : edge 1373 = PrimePairCache.smallRow1373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan78_7 : scan edge (search edge 3) 4 [419] 419 innerTail78_7 best = best := by
  rw [innerTail78_7, scan, if_neg (by decide), innerCandidates78_7_checked]
  change scan edge (search edge 3) 4 [419] 419 innerTail78_8
    (if 3 ≤ innerCandidates78_7.length then search edge 3 [419, 1373] innerCandidates78_7 1792 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan78_8

def innerTail78_6 : List Nat := 1097 :: innerTail78_7
def innerCandidates78_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates78_6_checked : candidates edge 4 419 1097 innerTail78_7 best.bound = innerCandidates78_6 := by
  unfold candidates
  have row : edge 1097 = PrimePairCache.smallRow1097.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan78_6 : scan edge (search edge 3) 4 [419] 419 innerTail78_6 best = best := by
  rw [innerTail78_6, scan, if_neg (by decide), innerCandidates78_6_checked]
  change scan edge (search edge 3) 4 [419] 419 innerTail78_7
    (if 3 ≤ innerCandidates78_6.length then search edge 3 [419, 1097] innerCandidates78_6 1516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan78_7

def innerTail78_5 : List Nat := 929 :: innerTail78_6
def innerCandidates78_5 : List Nat := [3209]
set_option maxRecDepth 4096 in
theorem innerCandidates78_5_checked : candidates edge 4 419 929 innerTail78_6 best.bound = innerCandidates78_5 := by
  unfold candidates
  have row : edge 929 = PrimePairCache.smallRow929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan78_5 : scan edge (search edge 3) 4 [419] 419 innerTail78_5 best = best := by
  rw [innerTail78_5, scan, if_neg (by decide), innerCandidates78_5_checked]
  change scan edge (search edge 3) 4 [419] 419 innerTail78_6
    (if 3 ≤ innerCandidates78_5.length then search edge 3 [419, 929] innerCandidates78_5 1348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan78_6

def innerTail78_4 : List Nat := 701 :: innerTail78_5
def innerCandidates78_4 : List Nat := [4583]
set_option maxRecDepth 4096 in
theorem innerCandidates78_4_checked : candidates edge 4 419 701 innerTail78_5 best.bound = innerCandidates78_4 := by
  unfold candidates
  have row : edge 701 = PrimePairCache.smallRow701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan78_4 : scan edge (search edge 3) 4 [419] 419 innerTail78_4 best = best := by
  rw [innerTail78_4, scan, if_neg (by decide), innerCandidates78_4_checked]
  change scan edge (search edge 3) 4 [419] 419 innerTail78_5
    (if 3 ≤ innerCandidates78_4.length then search edge 3 [419, 701] innerCandidates78_4 1120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan78_5

def innerTail78_3 : List Nat := 599 :: innerTail78_4
def innerCandidates78_3 : List Nat := [7883]
set_option maxRecDepth 4096 in
theorem innerCandidates78_3_checked : candidates edge 4 419 599 innerTail78_4 best.bound = innerCandidates78_3 := by
  unfold candidates
  have row : edge 599 = PrimePairCache.smallRow599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan78_3 : scan edge (search edge 3) 4 [419] 419 innerTail78_3 best = best := by
  rw [innerTail78_3, scan, if_neg (by decide), innerCandidates78_3_checked]
  change scan edge (search edge 3) 4 [419] 419 innerTail78_4
    (if 3 ≤ innerCandidates78_3.length then search edge 3 [419, 599] innerCandidates78_3 1018 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan78_4

def innerTail78_2 : List Nat := 563 :: innerTail78_3
def innerCandidates78_2 : List Nat := [5297]
set_option maxRecDepth 4096 in
theorem innerCandidates78_2_checked : candidates edge 4 419 563 innerTail78_3 best.bound = innerCandidates78_2 := by
  unfold candidates
  have row : edge 563 = PrimePairCache.smallRow563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan78_2 : scan edge (search edge 3) 4 [419] 419 innerTail78_2 best = best := by
  rw [innerTail78_2, scan, if_neg (by decide), innerCandidates78_2_checked]
  change scan edge (search edge 3) 4 [419] 419 innerTail78_3
    (if 3 ≤ innerCandidates78_2.length then search edge 3 [419, 563] innerCandidates78_2 982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan78_3

def innerTail78_1 : List Nat := 449 :: innerTail78_2
def innerCandidates78_1 : List Nat := [563, 5297]
set_option maxRecDepth 4096 in
theorem innerCandidates78_1_checked : candidates edge 4 419 449 innerTail78_2 best.bound = innerCandidates78_1 := by
  unfold candidates
  have row : edge 449 = PrimePairCache.smallRow449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan78_1 : scan edge (search edge 3) 4 [419] 419 innerTail78_1 best = best := by
  rw [innerTail78_1, scan, if_neg (by decide), innerCandidates78_1_checked]
  change scan edge (search edge 3) 4 [419] 419 innerTail78_2
    (if 3 ≤ innerCandidates78_1.length then search edge 3 [419, 449] innerCandidates78_1 868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan78_2

def innerTail78_0 : List Nat := 443 :: innerTail78_1
def innerCandidates78_0 : List Nat := [701, 7757, 7883]
set_option maxRecDepth 4096 in
theorem innerCandidates78_0_checked : candidates edge 4 419 443 innerTail78_1 best.bound = innerCandidates78_0 := by
  unfold candidates
  have row : edge 443 = PrimePairCache.smallRow443.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild78_0 : search edge 3 [419, 443] innerCandidates78_0 862 best = best := by decide

theorem innerScan78_0 : scan edge (search edge 3) 4 [419] 419 innerTail78_0 best = best := by
  rw [innerTail78_0, scan, if_neg (by decide), innerCandidates78_0_checked]
  change scan edge (search edge 3) 4 [419] 419 innerTail78_1
    (if 3 ≤ innerCandidates78_0.length then search edge 3 [419, 443] innerCandidates78_0 862 best else best) = best
  rw [if_pos (by decide), innerChild78_0]
  exact innerScan78_1

theorem rootBranch78 : search edge 4 [419] rootCandidates78 419 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [419] 419 innerTail78_0 best = best
  exact innerScan78_0

def rootCandidates77 : List Nat := [691, 709, 733, 1471, 1741, 1873, 2269, 2287, 2503, 2791, 3163, 3301, 3343, 3847, 4243, 4561, 4999, 5647, 5857, 6199, 6607, 7057, 7069, 7573, 7699, 8089, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates77_checked : candidates edge 5 0 409 tail78 best.bound = rootCandidates77 := by
  unfold candidates
  have row : edge 409 = PrimePairCache.smallRow409.contains := by funext q; rfl
  rw [row]
  decide

def innerTail77_27 : List Nat := []
theorem innerScan77_27 : scan edge (search edge 3) 4 [409] 409 innerTail77_27 best = best := rfl

def innerTail77_26 : List Nat := 8233 :: innerTail77_27
def innerCandidates77_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates77_26_checked : candidates edge 4 409 8233 innerTail77_27 best.bound = innerCandidates77_26 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_26 : scan edge (search edge 3) 4 [409] 409 innerTail77_26 best = best := by
  rw [innerTail77_26, scan, if_neg (by decide), innerCandidates77_26_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_27
    (if 3 ≤ innerCandidates77_26.length then search edge 3 [409, 8233] innerCandidates77_26 8642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_27

def innerTail77_25 : List Nat := 8089 :: innerTail77_26
def innerCandidates77_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates77_25_checked : candidates edge 4 409 8089 innerTail77_26 best.bound = innerCandidates77_25 := by
  unfold candidates
  have row : edge 8089 = SearchData.row8089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_25 : scan edge (search edge 3) 4 [409] 409 innerTail77_25 best = best := by
  rw [innerTail77_25, scan, if_neg (by decide), innerCandidates77_25_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_26
    (if 3 ≤ innerCandidates77_25.length then search edge 3 [409, 8089] innerCandidates77_25 8498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_26

def innerTail77_24 : List Nat := 7699 :: innerTail77_25
def innerCandidates77_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates77_24_checked : candidates edge 4 409 7699 innerTail77_25 best.bound = innerCandidates77_24 := by
  unfold candidates
  have row : edge 7699 = SearchData.row7699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_24 : scan edge (search edge 3) 4 [409] 409 innerTail77_24 best = best := by
  rw [innerTail77_24, scan, if_neg (by decide), innerCandidates77_24_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_25
    (if 3 ≤ innerCandidates77_24.length then search edge 3 [409, 7699] innerCandidates77_24 8108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_25

def innerTail77_23 : List Nat := 7573 :: innerTail77_24
def innerCandidates77_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates77_23_checked : candidates edge 4 409 7573 innerTail77_24 best.bound = innerCandidates77_23 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_23 : scan edge (search edge 3) 4 [409] 409 innerTail77_23 best = best := by
  rw [innerTail77_23, scan, if_neg (by decide), innerCandidates77_23_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_24
    (if 3 ≤ innerCandidates77_23.length then search edge 3 [409, 7573] innerCandidates77_23 7982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_24

def innerTail77_22 : List Nat := 7069 :: innerTail77_23
def innerCandidates77_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates77_22_checked : candidates edge 4 409 7069 innerTail77_23 best.bound = innerCandidates77_22 := by
  unfold candidates
  have row : edge 7069 = SearchData.row7069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_22 : scan edge (search edge 3) 4 [409] 409 innerTail77_22 best = best := by
  rw [innerTail77_22, scan, if_neg (by decide), innerCandidates77_22_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_23
    (if 3 ≤ innerCandidates77_22.length then search edge 3 [409, 7069] innerCandidates77_22 7478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_23

def innerTail77_21 : List Nat := 7057 :: innerTail77_22
def innerCandidates77_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates77_21_checked : candidates edge 4 409 7057 innerTail77_22 best.bound = innerCandidates77_21 := by
  unfold candidates
  have row : edge 7057 = SearchData.row7057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_21 : scan edge (search edge 3) 4 [409] 409 innerTail77_21 best = best := by
  rw [innerTail77_21, scan, if_neg (by decide), innerCandidates77_21_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_22
    (if 3 ≤ innerCandidates77_21.length then search edge 3 [409, 7057] innerCandidates77_21 7466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_22

def innerTail77_20 : List Nat := 6607 :: innerTail77_21
def innerCandidates77_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates77_20_checked : candidates edge 4 409 6607 innerTail77_21 best.bound = innerCandidates77_20 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_20 : scan edge (search edge 3) 4 [409] 409 innerTail77_20 best = best := by
  rw [innerTail77_20, scan, if_neg (by decide), innerCandidates77_20_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_21
    (if 3 ≤ innerCandidates77_20.length then search edge 3 [409, 6607] innerCandidates77_20 7016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_21

def innerTail77_19 : List Nat := 6199 :: innerTail77_20
def innerCandidates77_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates77_19_checked : candidates edge 4 409 6199 innerTail77_20 best.bound = innerCandidates77_19 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_19 : scan edge (search edge 3) 4 [409] 409 innerTail77_19 best = best := by
  rw [innerTail77_19, scan, if_neg (by decide), innerCandidates77_19_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_20
    (if 3 ≤ innerCandidates77_19.length then search edge 3 [409, 6199] innerCandidates77_19 6608 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_20

def innerTail77_18 : List Nat := 5857 :: innerTail77_19
def innerCandidates77_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates77_18_checked : candidates edge 4 409 5857 innerTail77_19 best.bound = innerCandidates77_18 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_18 : scan edge (search edge 3) 4 [409] 409 innerTail77_18 best = best := by
  rw [innerTail77_18, scan, if_neg (by decide), innerCandidates77_18_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_19
    (if 3 ≤ innerCandidates77_18.length then search edge 3 [409, 5857] innerCandidates77_18 6266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_19

def innerTail77_17 : List Nat := 5647 :: innerTail77_18
def innerCandidates77_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates77_17_checked : candidates edge 4 409 5647 innerTail77_18 best.bound = innerCandidates77_17 := by
  unfold candidates
  have row : edge 5647 = SearchData.row5647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_17 : scan edge (search edge 3) 4 [409] 409 innerTail77_17 best = best := by
  rw [innerTail77_17, scan, if_neg (by decide), innerCandidates77_17_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_18
    (if 3 ≤ innerCandidates77_17.length then search edge 3 [409, 5647] innerCandidates77_17 6056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_18

def innerTail77_16 : List Nat := 4999 :: innerTail77_17
def innerCandidates77_16 : List Nat := [7069]
set_option maxRecDepth 4096 in
theorem innerCandidates77_16_checked : candidates edge 4 409 4999 innerTail77_17 best.bound = innerCandidates77_16 := by
  unfold candidates
  have row : edge 4999 = SearchData.row4999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_16 : scan edge (search edge 3) 4 [409] 409 innerTail77_16 best = best := by
  rw [innerTail77_16, scan, if_neg (by decide), innerCandidates77_16_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_17
    (if 3 ≤ innerCandidates77_16.length then search edge 3 [409, 4999] innerCandidates77_16 5408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_17

def innerTail77_15 : List Nat := 4561 :: innerTail77_16
def innerCandidates77_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates77_15_checked : candidates edge 4 409 4561 innerTail77_16 best.bound = innerCandidates77_15 := by
  unfold candidates
  have row : edge 4561 = SearchData.row4561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_15 : scan edge (search edge 3) 4 [409] 409 innerTail77_15 best = best := by
  rw [innerTail77_15, scan, if_neg (by decide), innerCandidates77_15_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_16
    (if 3 ≤ innerCandidates77_15.length then search edge 3 [409, 4561] innerCandidates77_15 4970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_16

def innerTail77_14 : List Nat := 4243 :: innerTail77_15
def innerCandidates77_14 : List Nat := [7699]
set_option maxRecDepth 4096 in
theorem innerCandidates77_14_checked : candidates edge 4 409 4243 innerTail77_15 best.bound = innerCandidates77_14 := by
  unfold candidates
  have row : edge 4243 = SearchData.row4243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_14 : scan edge (search edge 3) 4 [409] 409 innerTail77_14 best = best := by
  rw [innerTail77_14, scan, if_neg (by decide), innerCandidates77_14_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_15
    (if 3 ≤ innerCandidates77_14.length then search edge 3 [409, 4243] innerCandidates77_14 4652 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_15

def innerTail77_13 : List Nat := 3847 :: innerTail77_14
def innerCandidates77_13 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates77_13_checked : candidates edge 4 409 3847 innerTail77_14 best.bound = innerCandidates77_13 := by
  unfold candidates
  have row : edge 3847 = SearchData.row3847.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_13 : scan edge (search edge 3) 4 [409] 409 innerTail77_13 best = best := by
  rw [innerTail77_13, scan, if_neg (by decide), innerCandidates77_13_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_14
    (if 3 ≤ innerCandidates77_13.length then search edge 3 [409, 3847] innerCandidates77_13 4256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_14

def innerTail77_12 : List Nat := 3343 :: innerTail77_13
def innerCandidates77_12 : List Nat := [4561]
set_option maxRecDepth 4096 in
theorem innerCandidates77_12_checked : candidates edge 4 409 3343 innerTail77_13 best.bound = innerCandidates77_12 := by
  unfold candidates
  have row : edge 3343 = SearchData.row3343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_12 : scan edge (search edge 3) 4 [409] 409 innerTail77_12 best = best := by
  rw [innerTail77_12, scan, if_neg (by decide), innerCandidates77_12_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_13
    (if 3 ≤ innerCandidates77_12.length then search edge 3 [409, 3343] innerCandidates77_12 3752 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_13

def innerTail77_11 : List Nat := 3301 :: innerTail77_12
def innerCandidates77_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates77_11_checked : candidates edge 4 409 3301 innerTail77_12 best.bound = innerCandidates77_11 := by
  unfold candidates
  have row : edge 3301 = SearchData.row3301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_11 : scan edge (search edge 3) 4 [409] 409 innerTail77_11 best = best := by
  rw [innerTail77_11, scan, if_neg (by decide), innerCandidates77_11_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_12
    (if 3 ≤ innerCandidates77_11.length then search edge 3 [409, 3301] innerCandidates77_11 3710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_12

def innerTail77_10 : List Nat := 3163 :: innerTail77_11
def innerCandidates77_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates77_10_checked : candidates edge 4 409 3163 innerTail77_11 best.bound = innerCandidates77_10 := by
  unfold candidates
  have row : edge 3163 = SearchData.row3163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_10 : scan edge (search edge 3) 4 [409] 409 innerTail77_10 best = best := by
  rw [innerTail77_10, scan, if_neg (by decide), innerCandidates77_10_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_11
    (if 3 ≤ innerCandidates77_10.length then search edge 3 [409, 3163] innerCandidates77_10 3572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_11

def innerTail77_9 : List Nat := 2791 :: innerTail77_10
def innerCandidates77_9 : List Nat := [6607]
set_option maxRecDepth 4096 in
theorem innerCandidates77_9_checked : candidates edge 4 409 2791 innerTail77_10 best.bound = innerCandidates77_9 := by
  unfold candidates
  have row : edge 2791 = SearchData.row2791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_9 : scan edge (search edge 3) 4 [409] 409 innerTail77_9 best = best := by
  rw [innerTail77_9, scan, if_neg (by decide), innerCandidates77_9_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_10
    (if 3 ≤ innerCandidates77_9.length then search edge 3 [409, 2791] innerCandidates77_9 3200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_10

def innerTail77_8 : List Nat := 2503 :: innerTail77_9
def innerCandidates77_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates77_8_checked : candidates edge 4 409 2503 innerTail77_9 best.bound = innerCandidates77_8 := by
  unfold candidates
  have row : edge 2503 = PrimePairCache.smallRow2503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_8 : scan edge (search edge 3) 4 [409] 409 innerTail77_8 best = best := by
  rw [innerTail77_8, scan, if_neg (by decide), innerCandidates77_8_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_9
    (if 3 ≤ innerCandidates77_8.length then search edge 3 [409, 2503] innerCandidates77_8 2912 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_9

def innerTail77_7 : List Nat := 2287 :: innerTail77_8
def innerCandidates77_7 : List Nat := [5857]
set_option maxRecDepth 4096 in
theorem innerCandidates77_7_checked : candidates edge 4 409 2287 innerTail77_8 best.bound = innerCandidates77_7 := by
  unfold candidates
  have row : edge 2287 = PrimePairCache.smallRow2287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_7 : scan edge (search edge 3) 4 [409] 409 innerTail77_7 best = best := by
  rw [innerTail77_7, scan, if_neg (by decide), innerCandidates77_7_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_8
    (if 3 ≤ innerCandidates77_7.length then search edge 3 [409, 2287] innerCandidates77_7 2696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_8

def innerTail77_6 : List Nat := 2269 :: innerTail77_7
def innerCandidates77_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates77_6_checked : candidates edge 4 409 2269 innerTail77_7 best.bound = innerCandidates77_6 := by
  unfold candidates
  have row : edge 2269 = PrimePairCache.smallRow2269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_6 : scan edge (search edge 3) 4 [409] 409 innerTail77_6 best = best := by
  rw [innerTail77_6, scan, if_neg (by decide), innerCandidates77_6_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_7
    (if 3 ≤ innerCandidates77_6.length then search edge 3 [409, 2269] innerCandidates77_6 2678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_7

def innerTail77_5 : List Nat := 1873 :: innerTail77_6
def innerCandidates77_5 : List Nat := [2269]
set_option maxRecDepth 4096 in
theorem innerCandidates77_5_checked : candidates edge 4 409 1873 innerTail77_6 best.bound = innerCandidates77_5 := by
  unfold candidates
  have row : edge 1873 = PrimePairCache.smallRow1873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_5 : scan edge (search edge 3) 4 [409] 409 innerTail77_5 best = best := by
  rw [innerTail77_5, scan, if_neg (by decide), innerCandidates77_5_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_6
    (if 3 ≤ innerCandidates77_5.length then search edge 3 [409, 1873] innerCandidates77_5 2282 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_6

def innerTail77_4 : List Nat := 1741 :: innerTail77_5
def innerCandidates77_4 : List Nat := [2287, 7573]
set_option maxRecDepth 4096 in
theorem innerCandidates77_4_checked : candidates edge 4 409 1741 innerTail77_5 best.bound = innerCandidates77_4 := by
  unfold candidates
  have row : edge 1741 = PrimePairCache.smallRow1741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_4 : scan edge (search edge 3) 4 [409] 409 innerTail77_4 best = best := by
  rw [innerTail77_4, scan, if_neg (by decide), innerCandidates77_4_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_5
    (if 3 ≤ innerCandidates77_4.length then search edge 3 [409, 1741] innerCandidates77_4 2150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_5

def innerTail77_3 : List Nat := 1471 :: innerTail77_4
def innerCandidates77_3 : List Nat := [4561, 7069]
set_option maxRecDepth 4096 in
theorem innerCandidates77_3_checked : candidates edge 4 409 1471 innerTail77_4 best.bound = innerCandidates77_3 := by
  unfold candidates
  have row : edge 1471 = PrimePairCache.smallRow1471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_3 : scan edge (search edge 3) 4 [409] 409 innerTail77_3 best = best := by
  rw [innerTail77_3, scan, if_neg (by decide), innerCandidates77_3_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_4
    (if 3 ≤ innerCandidates77_3.length then search edge 3 [409, 1471] innerCandidates77_3 1880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_4

def innerTail77_2 : List Nat := 733 :: innerTail77_3
def innerCandidates77_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates77_2_checked : candidates edge 4 409 733 innerTail77_3 best.bound = innerCandidates77_2 := by
  unfold candidates
  have row : edge 733 = PrimePairCache.smallRow733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_2 : scan edge (search edge 3) 4 [409] 409 innerTail77_2 best = best := by
  rw [innerTail77_2, scan, if_neg (by decide), innerCandidates77_2_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_3
    (if 3 ≤ innerCandidates77_2.length then search edge 3 [409, 733] innerCandidates77_2 1142 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_3

def innerTail77_1 : List Nat := 709 :: innerTail77_2
def innerCandidates77_1 : List Nat := [1471]
set_option maxRecDepth 4096 in
theorem innerCandidates77_1_checked : candidates edge 4 409 709 innerTail77_2 best.bound = innerCandidates77_1 := by
  unfold candidates
  have row : edge 709 = PrimePairCache.smallRow709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_1 : scan edge (search edge 3) 4 [409] 409 innerTail77_1 best = best := by
  rw [innerTail77_1, scan, if_neg (by decide), innerCandidates77_1_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_2
    (if 3 ≤ innerCandidates77_1.length then search edge 3 [409, 709] innerCandidates77_1 1118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_2

def innerTail77_0 : List Nat := 691 :: innerTail77_1
def innerCandidates77_0 : List Nat := [709, 6199]
set_option maxRecDepth 4096 in
theorem innerCandidates77_0_checked : candidates edge 4 409 691 innerTail77_1 best.bound = innerCandidates77_0 := by
  unfold candidates
  have row : edge 691 = PrimePairCache.smallRow691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan77_0 : scan edge (search edge 3) 4 [409] 409 innerTail77_0 best = best := by
  rw [innerTail77_0, scan, if_neg (by decide), innerCandidates77_0_checked]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_1
    (if 3 ≤ innerCandidates77_0.length then search edge 3 [409, 691] innerCandidates77_0 1100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan77_1

theorem rootBranch77 : search edge 4 [409] rootCandidates77 409 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [409] 409 innerTail77_0 best = best
  exact innerScan77_0

end ProjectEulerSolutions.PrimePairUpper.Search
