import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates84 : List Nat := [557, 563, 821, 941, 1871, 2069, 2273, 2309, 2441, 2459, 3659, 4157, 5009, 5171, 5297, 5801, 6353, 6491, 6599, 7109, 7583, 7817, 7919]
set_option maxRecDepth 4096 in
theorem rootCandidates84_checked : candidates edge 5 0 449 tail85 best.bound = rootCandidates84 := by
  unfold candidates
  have row : edge 449 = PrimePairCache.smallRow449.contains := by funext q; rfl
  rw [row]
  decide

def innerTail84_23 : List Nat := []
theorem innerScan84_23 : scan edge (search edge 3) 4 [449] 449 innerTail84_23 best = best := rfl

def innerTail84_22 : List Nat := 7919 :: innerTail84_23
def innerCandidates84_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates84_22_checked : candidates edge 4 449 7919 innerTail84_23 best.bound = innerCandidates84_22 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_22 : scan edge (search edge 3) 4 [449] 449 innerTail84_22 best = best := by
  rw [innerTail84_22, scan, if_neg (by decide), innerCandidates84_22_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_23
    (if 3 ≤ innerCandidates84_22.length then search edge 3 [449, 7919] innerCandidates84_22 8368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_23

def innerTail84_21 : List Nat := 7817 :: innerTail84_22
def innerCandidates84_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates84_21_checked : candidates edge 4 449 7817 innerTail84_22 best.bound = innerCandidates84_21 := by
  unfold candidates
  have row : edge 7817 = SearchData.row7817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_21 : scan edge (search edge 3) 4 [449] 449 innerTail84_21 best = best := by
  rw [innerTail84_21, scan, if_neg (by decide), innerCandidates84_21_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_22
    (if 3 ≤ innerCandidates84_21.length then search edge 3 [449, 7817] innerCandidates84_21 8266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_22

def innerTail84_20 : List Nat := 7583 :: innerTail84_21
def innerCandidates84_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates84_20_checked : candidates edge 4 449 7583 innerTail84_21 best.bound = innerCandidates84_20 := by
  unfold candidates
  have row : edge 7583 = SearchData.row7583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_20 : scan edge (search edge 3) 4 [449] 449 innerTail84_20 best = best := by
  rw [innerTail84_20, scan, if_neg (by decide), innerCandidates84_20_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_21
    (if 3 ≤ innerCandidates84_20.length then search edge 3 [449, 7583] innerCandidates84_20 8032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_21

def innerTail84_19 : List Nat := 7109 :: innerTail84_20
def innerCandidates84_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates84_19_checked : candidates edge 4 449 7109 innerTail84_20 best.bound = innerCandidates84_19 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_19 : scan edge (search edge 3) 4 [449] 449 innerTail84_19 best = best := by
  rw [innerTail84_19, scan, if_neg (by decide), innerCandidates84_19_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_20
    (if 3 ≤ innerCandidates84_19.length then search edge 3 [449, 7109] innerCandidates84_19 7558 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_20

def innerTail84_18 : List Nat := 6599 :: innerTail84_19
def innerCandidates84_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates84_18_checked : candidates edge 4 449 6599 innerTail84_19 best.bound = innerCandidates84_18 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_18 : scan edge (search edge 3) 4 [449] 449 innerTail84_18 best = best := by
  rw [innerTail84_18, scan, if_neg (by decide), innerCandidates84_18_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_19
    (if 3 ≤ innerCandidates84_18.length then search edge 3 [449, 6599] innerCandidates84_18 7048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_19

def innerTail84_17 : List Nat := 6491 :: innerTail84_18
def innerCandidates84_17 : List Nat := [7583]
set_option maxRecDepth 4096 in
theorem innerCandidates84_17_checked : candidates edge 4 449 6491 innerTail84_18 best.bound = innerCandidates84_17 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_17 : scan edge (search edge 3) 4 [449] 449 innerTail84_17 best = best := by
  rw [innerTail84_17, scan, if_neg (by decide), innerCandidates84_17_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_18
    (if 3 ≤ innerCandidates84_17.length then search edge 3 [449, 6491] innerCandidates84_17 6940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_18

def innerTail84_16 : List Nat := 6353 :: innerTail84_17
def innerCandidates84_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates84_16_checked : candidates edge 4 449 6353 innerTail84_17 best.bound = innerCandidates84_16 := by
  unfold candidates
  have row : edge 6353 = SearchData.row6353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_16 : scan edge (search edge 3) 4 [449] 449 innerTail84_16 best = best := by
  rw [innerTail84_16, scan, if_neg (by decide), innerCandidates84_16_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_17
    (if 3 ≤ innerCandidates84_16.length then search edge 3 [449, 6353] innerCandidates84_16 6802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_17

def innerTail84_15 : List Nat := 5801 :: innerTail84_16
def innerCandidates84_15 : List Nat := [7919]
set_option maxRecDepth 4096 in
theorem innerCandidates84_15_checked : candidates edge 4 449 5801 innerTail84_16 best.bound = innerCandidates84_15 := by
  unfold candidates
  have row : edge 5801 = SearchData.row5801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_15 : scan edge (search edge 3) 4 [449] 449 innerTail84_15 best = best := by
  rw [innerTail84_15, scan, if_neg (by decide), innerCandidates84_15_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_16
    (if 3 ≤ innerCandidates84_15.length then search edge 3 [449, 5801] innerCandidates84_15 6250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_16

def innerTail84_14 : List Nat := 5297 :: innerTail84_15
def innerCandidates84_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates84_14_checked : candidates edge 4 449 5297 innerTail84_15 best.bound = innerCandidates84_14 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_14 : scan edge (search edge 3) 4 [449] 449 innerTail84_14 best = best := by
  rw [innerTail84_14, scan, if_neg (by decide), innerCandidates84_14_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_15
    (if 3 ≤ innerCandidates84_14.length then search edge 3 [449, 5297] innerCandidates84_14 5746 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_15

def innerTail84_13 : List Nat := 5171 :: innerTail84_14
def innerCandidates84_13 : List Nat := [5297]
set_option maxRecDepth 4096 in
theorem innerCandidates84_13_checked : candidates edge 4 449 5171 innerTail84_14 best.bound = innerCandidates84_13 := by
  unfold candidates
  have row : edge 5171 = SearchData.row5171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_13 : scan edge (search edge 3) 4 [449] 449 innerTail84_13 best = best := by
  rw [innerTail84_13, scan, if_neg (by decide), innerCandidates84_13_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_14
    (if 3 ≤ innerCandidates84_13.length then search edge 3 [449, 5171] innerCandidates84_13 5620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_14

def innerTail84_12 : List Nat := 5009 :: innerTail84_13
def innerCandidates84_12 : List Nat := [6353, 7109]
set_option maxRecDepth 4096 in
theorem innerCandidates84_12_checked : candidates edge 4 449 5009 innerTail84_13 best.bound = innerCandidates84_12 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_12 : scan edge (search edge 3) 4 [449] 449 innerTail84_12 best = best := by
  rw [innerTail84_12, scan, if_neg (by decide), innerCandidates84_12_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_13
    (if 3 ≤ innerCandidates84_12.length then search edge 3 [449, 5009] innerCandidates84_12 5458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_13

def innerTail84_11 : List Nat := 4157 :: innerTail84_12
def innerCandidates84_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates84_11_checked : candidates edge 4 449 4157 innerTail84_12 best.bound = innerCandidates84_11 := by
  unfold candidates
  have row : edge 4157 = SearchData.row4157.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_11 : scan edge (search edge 3) 4 [449] 449 innerTail84_11 best = best := by
  rw [innerTail84_11, scan, if_neg (by decide), innerCandidates84_11_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_12
    (if 3 ≤ innerCandidates84_11.length then search edge 3 [449, 4157] innerCandidates84_11 4606 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_12

def innerTail84_10 : List Nat := 3659 :: innerTail84_11
def innerCandidates84_10 : List Nat := [4157]
set_option maxRecDepth 4096 in
theorem innerCandidates84_10_checked : candidates edge 4 449 3659 innerTail84_11 best.bound = innerCandidates84_10 := by
  unfold candidates
  have row : edge 3659 = SearchData.row3659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_10 : scan edge (search edge 3) 4 [449] 449 innerTail84_10 best = best := by
  rw [innerTail84_10, scan, if_neg (by decide), innerCandidates84_10_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_11
    (if 3 ≤ innerCandidates84_10.length then search edge 3 [449, 3659] innerCandidates84_10 4108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_11

def innerTail84_9 : List Nat := 2459 :: innerTail84_10
def innerCandidates84_9 : List Nat := [5171, 6353]
set_option maxRecDepth 4096 in
theorem innerCandidates84_9_checked : candidates edge 4 449 2459 innerTail84_10 best.bound = innerCandidates84_9 := by
  unfold candidates
  have row : edge 2459 = PrimePairCache.smallRow2459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_9 : scan edge (search edge 3) 4 [449] 449 innerTail84_9 best = best := by
  rw [innerTail84_9, scan, if_neg (by decide), innerCandidates84_9_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_10
    (if 3 ≤ innerCandidates84_9.length then search edge 3 [449, 2459] innerCandidates84_9 2908 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_10

def innerTail84_8 : List Nat := 2441 :: innerTail84_9
def innerCandidates84_8 : List Nat := [4157, 5297]
set_option maxRecDepth 4096 in
theorem innerCandidates84_8_checked : candidates edge 4 449 2441 innerTail84_9 best.bound = innerCandidates84_8 := by
  unfold candidates
  have row : edge 2441 = PrimePairCache.smallRow2441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_8 : scan edge (search edge 3) 4 [449] 449 innerTail84_8 best = best := by
  rw [innerTail84_8, scan, if_neg (by decide), innerCandidates84_8_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_9
    (if 3 ≤ innerCandidates84_8.length then search edge 3 [449, 2441] innerCandidates84_8 2890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_9

def innerTail84_7 : List Nat := 2309 :: innerTail84_8
def innerCandidates84_7 : List Nat := [7817]
set_option maxRecDepth 4096 in
theorem innerCandidates84_7_checked : candidates edge 4 449 2309 innerTail84_8 best.bound = innerCandidates84_7 := by
  unfold candidates
  have row : edge 2309 = PrimePairCache.smallRow2309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_7 : scan edge (search edge 3) 4 [449] 449 innerTail84_7 best = best := by
  rw [innerTail84_7, scan, if_neg (by decide), innerCandidates84_7_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_8
    (if 3 ≤ innerCandidates84_7.length then search edge 3 [449, 2309] innerCandidates84_7 2758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_8

def innerTail84_6 : List Nat := 2273 :: innerTail84_7
def innerCandidates84_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates84_6_checked : candidates edge 4 449 2273 innerTail84_7 best.bound = innerCandidates84_6 := by
  unfold candidates
  have row : edge 2273 = PrimePairCache.smallRow2273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_6 : scan edge (search edge 3) 4 [449] 449 innerTail84_6 best = best := by
  rw [innerTail84_6, scan, if_neg (by decide), innerCandidates84_6_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_7
    (if 3 ≤ innerCandidates84_6.length then search edge 3 [449, 2273] innerCandidates84_6 2722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_7

def innerTail84_5 : List Nat := 2069 :: innerTail84_6
def innerCandidates84_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates84_5_checked : candidates edge 4 449 2069 innerTail84_6 best.bound = innerCandidates84_5 := by
  unfold candidates
  have row : edge 2069 = PrimePairCache.smallRow2069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_5 : scan edge (search edge 3) 4 [449] 449 innerTail84_5 best = best := by
  rw [innerTail84_5, scan, if_neg (by decide), innerCandidates84_5_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_6
    (if 3 ≤ innerCandidates84_5.length then search edge 3 [449, 2069] innerCandidates84_5 2518 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_6

def innerTail84_4 : List Nat := 1871 :: innerTail84_5
def innerCandidates84_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates84_4_checked : candidates edge 4 449 1871 innerTail84_5 best.bound = innerCandidates84_4 := by
  unfold candidates
  have row : edge 1871 = PrimePairCache.smallRow1871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_4 : scan edge (search edge 3) 4 [449] 449 innerTail84_4 best = best := by
  rw [innerTail84_4, scan, if_neg (by decide), innerCandidates84_4_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_5
    (if 3 ≤ innerCandidates84_4.length then search edge 3 [449, 1871] innerCandidates84_4 2320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_5

def innerTail84_3 : List Nat := 941 :: innerTail84_4
def innerCandidates84_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates84_3_checked : candidates edge 4 449 941 innerTail84_4 best.bound = innerCandidates84_3 := by
  unfold candidates
  have row : edge 941 = PrimePairCache.smallRow941.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_3 : scan edge (search edge 3) 4 [449] 449 innerTail84_3 best = best := by
  rw [innerTail84_3, scan, if_neg (by decide), innerCandidates84_3_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_4
    (if 3 ≤ innerCandidates84_3.length then search edge 3 [449, 941] innerCandidates84_3 1390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_4

def innerTail84_2 : List Nat := 821 :: innerTail84_3
def innerCandidates84_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates84_2_checked : candidates edge 4 449 821 innerTail84_3 best.bound = innerCandidates84_2 := by
  unfold candidates
  have row : edge 821 = PrimePairCache.smallRow821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_2 : scan edge (search edge 3) 4 [449] 449 innerTail84_2 best = best := by
  rw [innerTail84_2, scan, if_neg (by decide), innerCandidates84_2_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_3
    (if 3 ≤ innerCandidates84_2.length then search edge 3 [449, 821] innerCandidates84_2 1270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_3

def innerTail84_1 : List Nat := 563 :: innerTail84_2
def innerCandidates84_1 : List Nat := [5297]
set_option maxRecDepth 4096 in
theorem innerCandidates84_1_checked : candidates edge 4 449 563 innerTail84_2 best.bound = innerCandidates84_1 := by
  unfold candidates
  have row : edge 563 = PrimePairCache.smallRow563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_1 : scan edge (search edge 3) 4 [449] 449 innerTail84_1 best = best := by
  rw [innerTail84_1, scan, if_neg (by decide), innerCandidates84_1_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_2
    (if 3 ≤ innerCandidates84_1.length then search edge 3 [449, 563] innerCandidates84_1 1012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_2

def innerTail84_0 : List Nat := 557 :: innerTail84_1
def innerCandidates84_0 : List Nat := [6491]
set_option maxRecDepth 4096 in
theorem innerCandidates84_0_checked : candidates edge 4 449 557 innerTail84_1 best.bound = innerCandidates84_0 := by
  unfold candidates
  have row : edge 557 = PrimePairCache.smallRow557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan84_0 : scan edge (search edge 3) 4 [449] 449 innerTail84_0 best = best := by
  rw [innerTail84_0, scan, if_neg (by decide), innerCandidates84_0_checked]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_1
    (if 3 ≤ innerCandidates84_0.length then search edge 3 [449, 557] innerCandidates84_0 1006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan84_1

theorem rootBranch84 : search edge 4 [449] rootCandidates84 449 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [449] 449 innerTail84_0 best = best
  exact innerScan84_0

def rootCandidates83 : List Nat := [701, 761, 953, 983, 1151, 1523, 1811, 2069, 2531, 2693, 3167, 3203, 3449, 3833, 3881, 4673, 4691, 5051, 5741, 5897, 6389, 6737, 6863, 7283, 7523, 7529, 7547, 7757, 7883]
set_option maxRecDepth 4096 in
theorem rootCandidates83_checked : candidates edge 5 0 443 tail84 best.bound = rootCandidates83 := by
  unfold candidates
  have row : edge 443 = PrimePairCache.smallRow443.contains := by funext q; rfl
  rw [row]
  decide

def innerTail83_29 : List Nat := []
theorem innerScan83_29 : scan edge (search edge 3) 4 [443] 443 innerTail83_29 best = best := rfl

def innerTail83_28 : List Nat := 7883 :: innerTail83_29
def innerCandidates83_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates83_28_checked : candidates edge 4 443 7883 innerTail83_29 best.bound = innerCandidates83_28 := by
  unfold candidates
  have row : edge 7883 = SearchData.row7883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_28 : scan edge (search edge 3) 4 [443] 443 innerTail83_28 best = best := by
  rw [innerTail83_28, scan, if_neg (by decide), innerCandidates83_28_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_29
    (if 3 ≤ innerCandidates83_28.length then search edge 3 [443, 7883] innerCandidates83_28 8326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_29

def innerTail83_27 : List Nat := 7757 :: innerTail83_28
def innerCandidates83_27 : List Nat := [7883]
set_option maxRecDepth 4096 in
theorem innerCandidates83_27_checked : candidates edge 4 443 7757 innerTail83_28 best.bound = innerCandidates83_27 := by
  unfold candidates
  have row : edge 7757 = SearchData.row7757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_27 : scan edge (search edge 3) 4 [443] 443 innerTail83_27 best = best := by
  rw [innerTail83_27, scan, if_neg (by decide), innerCandidates83_27_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_28
    (if 3 ≤ innerCandidates83_27.length then search edge 3 [443, 7757] innerCandidates83_27 8200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_28

def innerTail83_26 : List Nat := 7547 :: innerTail83_27
def innerCandidates83_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates83_26_checked : candidates edge 4 443 7547 innerTail83_27 best.bound = innerCandidates83_26 := by
  unfold candidates
  have row : edge 7547 = SearchData.row7547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_26 : scan edge (search edge 3) 4 [443] 443 innerTail83_26 best = best := by
  rw [innerTail83_26, scan, if_neg (by decide), innerCandidates83_26_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_27
    (if 3 ≤ innerCandidates83_26.length then search edge 3 [443, 7547] innerCandidates83_26 7990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_27

def innerTail83_25 : List Nat := 7529 :: innerTail83_26
def innerCandidates83_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates83_25_checked : candidates edge 4 443 7529 innerTail83_26 best.bound = innerCandidates83_25 := by
  unfold candidates
  have row : edge 7529 = SearchData.row7529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_25 : scan edge (search edge 3) 4 [443] 443 innerTail83_25 best = best := by
  rw [innerTail83_25, scan, if_neg (by decide), innerCandidates83_25_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_26
    (if 3 ≤ innerCandidates83_25.length then search edge 3 [443, 7529] innerCandidates83_25 7972 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_26

def innerTail83_24 : List Nat := 7523 :: innerTail83_25
def innerCandidates83_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates83_24_checked : candidates edge 4 443 7523 innerTail83_25 best.bound = innerCandidates83_24 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_24 : scan edge (search edge 3) 4 [443] 443 innerTail83_24 best = best := by
  rw [innerTail83_24, scan, if_neg (by decide), innerCandidates83_24_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_25
    (if 3 ≤ innerCandidates83_24.length then search edge 3 [443, 7523] innerCandidates83_24 7966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_25

def innerTail83_23 : List Nat := 7283 :: innerTail83_24
def innerCandidates83_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates83_23_checked : candidates edge 4 443 7283 innerTail83_24 best.bound = innerCandidates83_23 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_23 : scan edge (search edge 3) 4 [443] 443 innerTail83_23 best = best := by
  rw [innerTail83_23, scan, if_neg (by decide), innerCandidates83_23_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_24
    (if 3 ≤ innerCandidates83_23.length then search edge 3 [443, 7283] innerCandidates83_23 7726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_24

def innerTail83_22 : List Nat := 6863 :: innerTail83_23
def innerCandidates83_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates83_22_checked : candidates edge 4 443 6863 innerTail83_23 best.bound = innerCandidates83_22 := by
  unfold candidates
  have row : edge 6863 = SearchData.row6863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_22 : scan edge (search edge 3) 4 [443] 443 innerTail83_22 best = best := by
  rw [innerTail83_22, scan, if_neg (by decide), innerCandidates83_22_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_23
    (if 3 ≤ innerCandidates83_22.length then search edge 3 [443, 6863] innerCandidates83_22 7306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_23

def innerTail83_21 : List Nat := 6737 :: innerTail83_22
def innerCandidates83_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates83_21_checked : candidates edge 4 443 6737 innerTail83_22 best.bound = innerCandidates83_21 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_21 : scan edge (search edge 3) 4 [443] 443 innerTail83_21 best = best := by
  rw [innerTail83_21, scan, if_neg (by decide), innerCandidates83_21_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_22
    (if 3 ≤ innerCandidates83_21.length then search edge 3 [443, 6737] innerCandidates83_21 7180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_22

def innerTail83_20 : List Nat := 6389 :: innerTail83_21
def innerCandidates83_20 : List Nat := [7523]
set_option maxRecDepth 4096 in
theorem innerCandidates83_20_checked : candidates edge 4 443 6389 innerTail83_21 best.bound = innerCandidates83_20 := by
  unfold candidates
  have row : edge 6389 = SearchData.row6389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_20 : scan edge (search edge 3) 4 [443] 443 innerTail83_20 best = best := by
  rw [innerTail83_20, scan, if_neg (by decide), innerCandidates83_20_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_21
    (if 3 ≤ innerCandidates83_20.length then search edge 3 [443, 6389] innerCandidates83_20 6832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_21

def innerTail83_19 : List Nat := 5897 :: innerTail83_20
def innerCandidates83_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates83_19_checked : candidates edge 4 443 5897 innerTail83_20 best.bound = innerCandidates83_19 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_19 : scan edge (search edge 3) 4 [443] 443 innerTail83_19 best = best := by
  rw [innerTail83_19, scan, if_neg (by decide), innerCandidates83_19_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_20
    (if 3 ≤ innerCandidates83_19.length then search edge 3 [443, 5897] innerCandidates83_19 6340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_20

def innerTail83_18 : List Nat := 5741 :: innerTail83_19
def innerCandidates83_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates83_18_checked : candidates edge 4 443 5741 innerTail83_19 best.bound = innerCandidates83_18 := by
  unfold candidates
  have row : edge 5741 = SearchData.row5741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_18 : scan edge (search edge 3) 4 [443] 443 innerTail83_18 best = best := by
  rw [innerTail83_18, scan, if_neg (by decide), innerCandidates83_18_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_19
    (if 3 ≤ innerCandidates83_18.length then search edge 3 [443, 5741] innerCandidates83_18 6184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_19

def innerTail83_17 : List Nat := 5051 :: innerTail83_18
def innerCandidates83_17 : List Nat := [6737, 6863, 7283]
set_option maxRecDepth 4096 in
theorem innerCandidates83_17_checked : candidates edge 4 443 5051 innerTail83_18 best.bound = innerCandidates83_17 := by
  unfold candidates
  have row : edge 5051 = SearchData.row5051.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild83_17 : search edge 3 [443, 5051] innerCandidates83_17 5494 best = best := by decide

theorem innerScan83_17 : scan edge (search edge 3) 4 [443] 443 innerTail83_17 best = best := by
  rw [innerTail83_17, scan, if_neg (by decide), innerCandidates83_17_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_18
    (if 3 ≤ innerCandidates83_17.length then search edge 3 [443, 5051] innerCandidates83_17 5494 best else best) = best
  rw [if_pos (by decide), innerChild83_17]
  exact innerScan83_18

def innerTail83_16 : List Nat := 4691 :: innerTail83_17
def innerCandidates83_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates83_16_checked : candidates edge 4 443 4691 innerTail83_17 best.bound = innerCandidates83_16 := by
  unfold candidates
  have row : edge 4691 = SearchData.row4691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_16 : scan edge (search edge 3) 4 [443] 443 innerTail83_16 best = best := by
  rw [innerTail83_16, scan, if_neg (by decide), innerCandidates83_16_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_17
    (if 3 ≤ innerCandidates83_16.length then search edge 3 [443, 4691] innerCandidates83_16 5134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_17

def innerTail83_15 : List Nat := 4673 :: innerTail83_16
def innerCandidates83_15 : List Nat := [6737]
set_option maxRecDepth 4096 in
theorem innerCandidates83_15_checked : candidates edge 4 443 4673 innerTail83_16 best.bound = innerCandidates83_15 := by
  unfold candidates
  have row : edge 4673 = SearchData.row4673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_15 : scan edge (search edge 3) 4 [443] 443 innerTail83_15 best = best := by
  rw [innerTail83_15, scan, if_neg (by decide), innerCandidates83_15_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_16
    (if 3 ≤ innerCandidates83_15.length then search edge 3 [443, 4673] innerCandidates83_15 5116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_16

def innerTail83_14 : List Nat := 3881 :: innerTail83_15
def innerCandidates83_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates83_14_checked : candidates edge 4 443 3881 innerTail83_15 best.bound = innerCandidates83_14 := by
  unfold candidates
  have row : edge 3881 = SearchData.row3881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_14 : scan edge (search edge 3) 4 [443] 443 innerTail83_14 best = best := by
  rw [innerTail83_14, scan, if_neg (by decide), innerCandidates83_14_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_15
    (if 3 ≤ innerCandidates83_14.length then search edge 3 [443, 3881] innerCandidates83_14 4324 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_15

def innerTail83_13 : List Nat := 3833 :: innerTail83_14
def innerCandidates83_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates83_13_checked : candidates edge 4 443 3833 innerTail83_14 best.bound = innerCandidates83_13 := by
  unfold candidates
  have row : edge 3833 = SearchData.row3833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_13 : scan edge (search edge 3) 4 [443] 443 innerTail83_13 best = best := by
  rw [innerTail83_13, scan, if_neg (by decide), innerCandidates83_13_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_14
    (if 3 ≤ innerCandidates83_13.length then search edge 3 [443, 3833] innerCandidates83_13 4276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_14

def innerTail83_12 : List Nat := 3449 :: innerTail83_13
def innerCandidates83_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates83_12_checked : candidates edge 4 443 3449 innerTail83_13 best.bound = innerCandidates83_12 := by
  unfold candidates
  have row : edge 3449 = SearchData.row3449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_12 : scan edge (search edge 3) 4 [443] 443 innerTail83_12 best = best := by
  rw [innerTail83_12, scan, if_neg (by decide), innerCandidates83_12_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_13
    (if 3 ≤ innerCandidates83_12.length then search edge 3 [443, 3449] innerCandidates83_12 3892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_13

def innerTail83_11 : List Nat := 3203 :: innerTail83_12
def innerCandidates83_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates83_11_checked : candidates edge 4 443 3203 innerTail83_12 best.bound = innerCandidates83_11 := by
  unfold candidates
  have row : edge 3203 = SearchData.row3203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_11 : scan edge (search edge 3) 4 [443] 443 innerTail83_11 best = best := by
  rw [innerTail83_11, scan, if_neg (by decide), innerCandidates83_11_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_12
    (if 3 ≤ innerCandidates83_11.length then search edge 3 [443, 3203] innerCandidates83_11 3646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_12

def innerTail83_10 : List Nat := 3167 :: innerTail83_11
def innerCandidates83_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates83_10_checked : candidates edge 4 443 3167 innerTail83_11 best.bound = innerCandidates83_10 := by
  unfold candidates
  have row : edge 3167 = SearchData.row3167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_10 : scan edge (search edge 3) 4 [443] 443 innerTail83_10 best = best := by
  rw [innerTail83_10, scan, if_neg (by decide), innerCandidates83_10_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_11
    (if 3 ≤ innerCandidates83_10.length then search edge 3 [443, 3167] innerCandidates83_10 3610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_11

def innerTail83_9 : List Nat := 2693 :: innerTail83_10
def innerCandidates83_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates83_9_checked : candidates edge 4 443 2693 innerTail83_10 best.bound = innerCandidates83_9 := by
  unfold candidates
  have row : edge 2693 = SearchData.row2693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_9 : scan edge (search edge 3) 4 [443] 443 innerTail83_9 best = best := by
  rw [innerTail83_9, scan, if_neg (by decide), innerCandidates83_9_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_10
    (if 3 ≤ innerCandidates83_9.length then search edge 3 [443, 2693] innerCandidates83_9 3136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_10

def innerTail83_8 : List Nat := 2531 :: innerTail83_9
def innerCandidates83_8 : List Nat := [7883]
set_option maxRecDepth 4096 in
theorem innerCandidates83_8_checked : candidates edge 4 443 2531 innerTail83_9 best.bound = innerCandidates83_8 := by
  unfold candidates
  have row : edge 2531 = PrimePairCache.smallRow2531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_8 : scan edge (search edge 3) 4 [443] 443 innerTail83_8 best = best := by
  rw [innerTail83_8, scan, if_neg (by decide), innerCandidates83_8_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_9
    (if 3 ≤ innerCandidates83_8.length then search edge 3 [443, 2531] innerCandidates83_8 2974 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_9

def innerTail83_7 : List Nat := 2069 :: innerTail83_8
def innerCandidates83_7 : List Nat := [4691, 7283, 7757]
set_option maxRecDepth 4096 in
theorem innerCandidates83_7_checked : candidates edge 4 443 2069 innerTail83_8 best.bound = innerCandidates83_7 := by
  unfold candidates
  have row : edge 2069 = PrimePairCache.smallRow2069.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild83_7 : search edge 3 [443, 2069] innerCandidates83_7 2512 best = best := by decide

theorem innerScan83_7 : scan edge (search edge 3) 4 [443] 443 innerTail83_7 best = best := by
  rw [innerTail83_7, scan, if_neg (by decide), innerCandidates83_7_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_8
    (if 3 ≤ innerCandidates83_7.length then search edge 3 [443, 2069] innerCandidates83_7 2512 best else best) = best
  rw [if_pos (by decide), innerChild83_7]
  exact innerScan83_8

def innerTail83_6 : List Nat := 1811 :: innerTail83_7
def innerCandidates83_6 : List Nat := [6863]
set_option maxRecDepth 4096 in
theorem innerCandidates83_6_checked : candidates edge 4 443 1811 innerTail83_7 best.bound = innerCandidates83_6 := by
  unfold candidates
  have row : edge 1811 = PrimePairCache.smallRow1811.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_6 : scan edge (search edge 3) 4 [443] 443 innerTail83_6 best = best := by
  rw [innerTail83_6, scan, if_neg (by decide), innerCandidates83_6_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_7
    (if 3 ≤ innerCandidates83_6.length then search edge 3 [443, 1811] innerCandidates83_6 2254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_7

def innerTail83_5 : List Nat := 1523 :: innerTail83_6
def innerCandidates83_5 : List Nat := [3203, 3833]
set_option maxRecDepth 4096 in
theorem innerCandidates83_5_checked : candidates edge 4 443 1523 innerTail83_6 best.bound = innerCandidates83_5 := by
  unfold candidates
  have row : edge 1523 = PrimePairCache.smallRow1523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_5 : scan edge (search edge 3) 4 [443] 443 innerTail83_5 best = best := by
  rw [innerTail83_5, scan, if_neg (by decide), innerCandidates83_5_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_6
    (if 3 ≤ innerCandidates83_5.length then search edge 3 [443, 1523] innerCandidates83_5 1966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_6

def innerTail83_4 : List Nat := 1151 :: innerTail83_5
def innerCandidates83_4 : List Nat := [5897, 6863]
set_option maxRecDepth 4096 in
theorem innerCandidates83_4_checked : candidates edge 4 443 1151 innerTail83_5 best.bound = innerCandidates83_4 := by
  unfold candidates
  have row : edge 1151 = PrimePairCache.smallRow1151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_4 : scan edge (search edge 3) 4 [443] 443 innerTail83_4 best = best := by
  rw [innerTail83_4, scan, if_neg (by decide), innerCandidates83_4_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_5
    (if 3 ≤ innerCandidates83_4.length then search edge 3 [443, 1151] innerCandidates83_4 1594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_5

def innerTail83_3 : List Nat := 983 :: innerTail83_4
def innerCandidates83_3 : List Nat := [6389]
set_option maxRecDepth 4096 in
theorem innerCandidates83_3_checked : candidates edge 4 443 983 innerTail83_4 best.bound = innerCandidates83_3 := by
  unfold candidates
  have row : edge 983 = PrimePairCache.smallRow983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_3 : scan edge (search edge 3) 4 [443] 443 innerTail83_3 best = best := by
  rw [innerTail83_3, scan, if_neg (by decide), innerCandidates83_3_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_4
    (if 3 ≤ innerCandidates83_3.length then search edge 3 [443, 983] innerCandidates83_3 1426 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_4

def innerTail83_2 : List Nat := 953 :: innerTail83_3
def innerCandidates83_2 : List Nat := [1523]
set_option maxRecDepth 4096 in
theorem innerCandidates83_2_checked : candidates edge 4 443 953 innerTail83_3 best.bound = innerCandidates83_2 := by
  unfold candidates
  have row : edge 953 = PrimePairCache.smallRow953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_2 : scan edge (search edge 3) 4 [443] 443 innerTail83_2 best = best := by
  rw [innerTail83_2, scan, if_neg (by decide), innerCandidates83_2_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_3
    (if 3 ≤ innerCandidates83_2.length then search edge 3 [443, 953] innerCandidates83_2 1396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_3

def innerTail83_1 : List Nat := 761 :: innerTail83_2
def innerCandidates83_1 : List Nat := [2531, 7283, 7529]
set_option maxRecDepth 4096 in
theorem innerCandidates83_1_checked : candidates edge 4 443 761 innerTail83_2 best.bound = innerCandidates83_1 := by
  unfold candidates
  have row : edge 761 = PrimePairCache.smallRow761.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild83_1 : search edge 3 [443, 761] innerCandidates83_1 1204 best = best := by decide

theorem innerScan83_1 : scan edge (search edge 3) 4 [443] 443 innerTail83_1 best = best := by
  rw [innerTail83_1, scan, if_neg (by decide), innerCandidates83_1_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_2
    (if 3 ≤ innerCandidates83_1.length then search edge 3 [443, 761] innerCandidates83_1 1204 best else best) = best
  rw [if_pos (by decide), innerChild83_1]
  exact innerScan83_2

def innerTail83_0 : List Nat := 701 :: innerTail83_1
def innerCandidates83_0 : List Nat := [6389]
set_option maxRecDepth 4096 in
theorem innerCandidates83_0_checked : candidates edge 4 443 701 innerTail83_1 best.bound = innerCandidates83_0 := by
  unfold candidates
  have row : edge 701 = PrimePairCache.smallRow701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan83_0 : scan edge (search edge 3) 4 [443] 443 innerTail83_0 best = best := by
  rw [innerTail83_0, scan, if_neg (by decide), innerCandidates83_0_checked]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_1
    (if 3 ≤ innerCandidates83_0.length then search edge 3 [443, 701] innerCandidates83_0 1144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan83_1

theorem rootBranch83 : search edge 4 [443] rootCandidates83 443 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [443] 443 innerTail83_0 best = best
  exact innerScan83_0

end ProjectEulerSolutions.PrimePairUpper.Search
