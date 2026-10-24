import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates62 : List Nat := [619, 727, 991, 1231, 1669, 1747, 2281, 3469, 3583, 3733, 3853, 4231, 4591, 6037, 6079, 8221, 8329]
set_option maxRecDepth 4096 in
theorem rootCandidates62_checked : candidates edge 5 0 313 tail63 best.bound = rootCandidates62 := by
  unfold candidates
  have row : edge 313 = PrimePairCache.smallRow313.contains := by funext q; rfl
  rw [row]
  decide

def innerTail62_17 : List Nat := []
theorem innerScan62_17 : scan edge (search edge 3) 4 [313] 313 innerTail62_17 best = best := rfl

def innerTail62_16 : List Nat := 8329 :: innerTail62_17
def innerCandidates62_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates62_16_checked : candidates edge 4 313 8329 innerTail62_17 best.bound = innerCandidates62_16 := by
  unfold candidates
  have row : edge 8329 = SearchData.row8329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan62_16 : scan edge (search edge 3) 4 [313] 313 innerTail62_16 best = best := by
  rw [innerTail62_16, scan, if_neg (by decide), innerCandidates62_16_checked]
  change scan edge (search edge 3) 4 [313] 313 innerTail62_17
    (if 3 ≤ innerCandidates62_16.length then search edge 3 [313, 8329] innerCandidates62_16 8642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan62_17

def innerTail62_15 : List Nat := 8221 :: innerTail62_16
def innerCandidates62_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates62_15_checked : candidates edge 4 313 8221 innerTail62_16 best.bound = innerCandidates62_15 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan62_15 : scan edge (search edge 3) 4 [313] 313 innerTail62_15 best = best := by
  rw [innerTail62_15, scan, if_neg (by decide), innerCandidates62_15_checked]
  change scan edge (search edge 3) 4 [313] 313 innerTail62_16
    (if 3 ≤ innerCandidates62_15.length then search edge 3 [313, 8221] innerCandidates62_15 8534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan62_16

def innerTail62_14 : List Nat := 6079 :: innerTail62_15
def innerCandidates62_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates62_14_checked : candidates edge 4 313 6079 innerTail62_15 best.bound = innerCandidates62_14 := by
  unfold candidates
  have row : edge 6079 = SearchData.row6079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan62_14 : scan edge (search edge 3) 4 [313] 313 innerTail62_14 best = best := by
  rw [innerTail62_14, scan, if_neg (by decide), innerCandidates62_14_checked]
  change scan edge (search edge 3) 4 [313] 313 innerTail62_15
    (if 3 ≤ innerCandidates62_14.length then search edge 3 [313, 6079] innerCandidates62_14 6392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan62_15

def innerTail62_13 : List Nat := 6037 :: innerTail62_14
def innerCandidates62_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates62_13_checked : candidates edge 4 313 6037 innerTail62_14 best.bound = innerCandidates62_13 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan62_13 : scan edge (search edge 3) 4 [313] 313 innerTail62_13 best = best := by
  rw [innerTail62_13, scan, if_neg (by decide), innerCandidates62_13_checked]
  change scan edge (search edge 3) 4 [313] 313 innerTail62_14
    (if 3 ≤ innerCandidates62_13.length then search edge 3 [313, 6037] innerCandidates62_13 6350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan62_14

def innerTail62_12 : List Nat := 4591 :: innerTail62_13
def innerCandidates62_12 : List Nat := [8329]
set_option maxRecDepth 4096 in
theorem innerCandidates62_12_checked : candidates edge 4 313 4591 innerTail62_13 best.bound = innerCandidates62_12 := by
  unfold candidates
  have row : edge 4591 = SearchData.row4591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan62_12 : scan edge (search edge 3) 4 [313] 313 innerTail62_12 best = best := by
  rw [innerTail62_12, scan, if_neg (by decide), innerCandidates62_12_checked]
  change scan edge (search edge 3) 4 [313] 313 innerTail62_13
    (if 3 ≤ innerCandidates62_12.length then search edge 3 [313, 4591] innerCandidates62_12 4904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan62_13

def innerTail62_11 : List Nat := 4231 :: innerTail62_12
def innerCandidates62_11 : List Nat := [6037]
set_option maxRecDepth 4096 in
theorem innerCandidates62_11_checked : candidates edge 4 313 4231 innerTail62_12 best.bound = innerCandidates62_11 := by
  unfold candidates
  have row : edge 4231 = SearchData.row4231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan62_11 : scan edge (search edge 3) 4 [313] 313 innerTail62_11 best = best := by
  rw [innerTail62_11, scan, if_neg (by decide), innerCandidates62_11_checked]
  change scan edge (search edge 3) 4 [313] 313 innerTail62_12
    (if 3 ≤ innerCandidates62_11.length then search edge 3 [313, 4231] innerCandidates62_11 4544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan62_12

def innerTail62_10 : List Nat := 3853 :: innerTail62_11
def innerCandidates62_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates62_10_checked : candidates edge 4 313 3853 innerTail62_11 best.bound = innerCandidates62_10 := by
  unfold candidates
  have row : edge 3853 = SearchData.row3853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan62_10 : scan edge (search edge 3) 4 [313] 313 innerTail62_10 best = best := by
  rw [innerTail62_10, scan, if_neg (by decide), innerCandidates62_10_checked]
  change scan edge (search edge 3) 4 [313] 313 innerTail62_11
    (if 3 ≤ innerCandidates62_10.length then search edge 3 [313, 3853] innerCandidates62_10 4166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan62_11

def innerTail62_9 : List Nat := 3733 :: innerTail62_10
def innerCandidates62_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates62_9_checked : candidates edge 4 313 3733 innerTail62_10 best.bound = innerCandidates62_9 := by
  unfold candidates
  have row : edge 3733 = SearchData.row3733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan62_9 : scan edge (search edge 3) 4 [313] 313 innerTail62_9 best = best := by
  rw [innerTail62_9, scan, if_neg (by decide), innerCandidates62_9_checked]
  change scan edge (search edge 3) 4 [313] 313 innerTail62_10
    (if 3 ≤ innerCandidates62_9.length then search edge 3 [313, 3733] innerCandidates62_9 4046 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan62_10

def innerTail62_8 : List Nat := 3583 :: innerTail62_9
def innerCandidates62_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates62_8_checked : candidates edge 4 313 3583 innerTail62_9 best.bound = innerCandidates62_8 := by
  unfold candidates
  have row : edge 3583 = SearchData.row3583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan62_8 : scan edge (search edge 3) 4 [313] 313 innerTail62_8 best = best := by
  rw [innerTail62_8, scan, if_neg (by decide), innerCandidates62_8_checked]
  change scan edge (search edge 3) 4 [313] 313 innerTail62_9
    (if 3 ≤ innerCandidates62_8.length then search edge 3 [313, 3583] innerCandidates62_8 3896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan62_9

def innerTail62_7 : List Nat := 3469 :: innerTail62_8
def innerCandidates62_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates62_7_checked : candidates edge 4 313 3469 innerTail62_8 best.bound = innerCandidates62_7 := by
  unfold candidates
  have row : edge 3469 = SearchData.row3469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan62_7 : scan edge (search edge 3) 4 [313] 313 innerTail62_7 best = best := by
  rw [innerTail62_7, scan, if_neg (by decide), innerCandidates62_7_checked]
  change scan edge (search edge 3) 4 [313] 313 innerTail62_8
    (if 3 ≤ innerCandidates62_7.length then search edge 3 [313, 3469] innerCandidates62_7 3782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan62_8

def innerTail62_6 : List Nat := 2281 :: innerTail62_7
def innerCandidates62_6 : List Nat := [4591]
set_option maxRecDepth 4096 in
theorem innerCandidates62_6_checked : candidates edge 4 313 2281 innerTail62_7 best.bound = innerCandidates62_6 := by
  unfold candidates
  have row : edge 2281 = PrimePairCache.smallRow2281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan62_6 : scan edge (search edge 3) 4 [313] 313 innerTail62_6 best = best := by
  rw [innerTail62_6, scan, if_neg (by decide), innerCandidates62_6_checked]
  change scan edge (search edge 3) 4 [313] 313 innerTail62_7
    (if 3 ≤ innerCandidates62_6.length then search edge 3 [313, 2281] innerCandidates62_6 2594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan62_7

def innerTail62_5 : List Nat := 1747 :: innerTail62_6
def innerCandidates62_5 : List Nat := [4231, 6037]
set_option maxRecDepth 4096 in
theorem innerCandidates62_5_checked : candidates edge 4 313 1747 innerTail62_6 best.bound = innerCandidates62_5 := by
  unfold candidates
  have row : edge 1747 = PrimePairCache.smallRow1747.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan62_5 : scan edge (search edge 3) 4 [313] 313 innerTail62_5 best = best := by
  rw [innerTail62_5, scan, if_neg (by decide), innerCandidates62_5_checked]
  change scan edge (search edge 3) 4 [313] 313 innerTail62_6
    (if 3 ≤ innerCandidates62_5.length then search edge 3 [313, 1747] innerCandidates62_5 2060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan62_6

def innerTail62_4 : List Nat := 1669 :: innerTail62_5
def innerCandidates62_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates62_4_checked : candidates edge 4 313 1669 innerTail62_5 best.bound = innerCandidates62_4 := by
  unfold candidates
  have row : edge 1669 = PrimePairCache.smallRow1669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan62_4 : scan edge (search edge 3) 4 [313] 313 innerTail62_4 best = best := by
  rw [innerTail62_4, scan, if_neg (by decide), innerCandidates62_4_checked]
  change scan edge (search edge 3) 4 [313] 313 innerTail62_5
    (if 3 ≤ innerCandidates62_4.length then search edge 3 [313, 1669] innerCandidates62_4 1982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan62_5

def innerTail62_3 : List Nat := 1231 :: innerTail62_4
def innerCandidates62_3 : List Nat := [4231, 6037, 8221]
set_option maxRecDepth 4096 in
theorem innerCandidates62_3_checked : candidates edge 4 313 1231 innerTail62_4 best.bound = innerCandidates62_3 := by
  unfold candidates
  have row : edge 1231 = PrimePairCache.smallRow1231.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild62_3 : search edge 3 [313, 1231] innerCandidates62_3 1544 best = best := by decide

theorem innerScan62_3 : scan edge (search edge 3) 4 [313] 313 innerTail62_3 best = best := by
  rw [innerTail62_3, scan, if_neg (by decide), innerCandidates62_3_checked]
  change scan edge (search edge 3) 4 [313] 313 innerTail62_4
    (if 3 ≤ innerCandidates62_3.length then search edge 3 [313, 1231] innerCandidates62_3 1544 best else best) = best
  rw [if_pos (by decide), innerChild62_3]
  exact innerScan62_4

def innerTail62_2 : List Nat := 991 :: innerTail62_3
def innerCandidates62_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates62_2_checked : candidates edge 4 313 991 innerTail62_3 best.bound = innerCandidates62_2 := by
  unfold candidates
  have row : edge 991 = PrimePairCache.smallRow991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan62_2 : scan edge (search edge 3) 4 [313] 313 innerTail62_2 best = best := by
  rw [innerTail62_2, scan, if_neg (by decide), innerCandidates62_2_checked]
  change scan edge (search edge 3) 4 [313] 313 innerTail62_3
    (if 3 ≤ innerCandidates62_2.length then search edge 3 [313, 991] innerCandidates62_2 1304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan62_3

def innerTail62_1 : List Nat := 727 :: innerTail62_2
def innerCandidates62_1 : List Nat := [1669]
set_option maxRecDepth 4096 in
theorem innerCandidates62_1_checked : candidates edge 4 313 727 innerTail62_2 best.bound = innerCandidates62_1 := by
  unfold candidates
  have row : edge 727 = PrimePairCache.smallRow727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan62_1 : scan edge (search edge 3) 4 [313] 313 innerTail62_1 best = best := by
  rw [innerTail62_1, scan, if_neg (by decide), innerCandidates62_1_checked]
  change scan edge (search edge 3) 4 [313] 313 innerTail62_2
    (if 3 ≤ innerCandidates62_1.length then search edge 3 [313, 727] innerCandidates62_1 1040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan62_2

def innerTail62_0 : List Nat := 619 :: innerTail62_1
def innerCandidates62_0 : List Nat := [1747]
set_option maxRecDepth 4096 in
theorem innerCandidates62_0_checked : candidates edge 4 313 619 innerTail62_1 best.bound = innerCandidates62_0 := by
  unfold candidates
  have row : edge 619 = PrimePairCache.smallRow619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan62_0 : scan edge (search edge 3) 4 [313] 313 innerTail62_0 best = best := by
  rw [innerTail62_0, scan, if_neg (by decide), innerCandidates62_0_checked]
  change scan edge (search edge 3) 4 [313] 313 innerTail62_1
    (if 3 ≤ innerCandidates62_0.length then search edge 3 [313, 619] innerCandidates62_0 932 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan62_1

theorem rootBranch62 : search edge 4 [313] rootCandidates62 313 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [313] 313 innerTail62_0 best = best
  exact innerScan62_0

def rootCandidates61 : List Nat := [359, 653, 677, 821, 827, 881, 1223, 2111, 2141, 2381, 2441, 2609, 2909, 3863, 4019, 4871, 4967, 5171, 5237, 5279, 6011, 6131, 6263, 6761, 7013, 7211, 7523, 7559, 8069, 8237, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates61_checked : candidates edge 5 0 311 tail62 best.bound = rootCandidates61 := by
  unfold candidates
  have row : edge 311 = PrimePairCache.smallRow311.contains := by funext q; rfl
  rw [row]
  decide

def innerTail61_31 : List Nat := []
theorem innerScan61_31 : scan edge (search edge 3) 4 [311] 311 innerTail61_31 best = best := rfl

def innerTail61_30 : List Nat := 8387 :: innerTail61_31
def innerCandidates61_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates61_30_checked : candidates edge 4 311 8387 innerTail61_31 best.bound = innerCandidates61_30 := by
  unfold candidates
  have row : edge 8387 = SearchData.row8387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_30 : scan edge (search edge 3) 4 [311] 311 innerTail61_30 best = best := by
  rw [innerTail61_30, scan, if_neg (by decide), innerCandidates61_30_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_31
    (if 3 ≤ innerCandidates61_30.length then search edge 3 [311, 8387] innerCandidates61_30 8698 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_31

def innerTail61_29 : List Nat := 8237 :: innerTail61_30
def innerCandidates61_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates61_29_checked : candidates edge 4 311 8237 innerTail61_30 best.bound = innerCandidates61_29 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_29 : scan edge (search edge 3) 4 [311] 311 innerTail61_29 best = best := by
  rw [innerTail61_29, scan, if_neg (by decide), innerCandidates61_29_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_30
    (if 3 ≤ innerCandidates61_29.length then search edge 3 [311, 8237] innerCandidates61_29 8548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_30

def innerTail61_28 : List Nat := 8069 :: innerTail61_29
def innerCandidates61_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates61_28_checked : candidates edge 4 311 8069 innerTail61_29 best.bound = innerCandidates61_28 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_28 : scan edge (search edge 3) 4 [311] 311 innerTail61_28 best = best := by
  rw [innerTail61_28, scan, if_neg (by decide), innerCandidates61_28_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_29
    (if 3 ≤ innerCandidates61_28.length then search edge 3 [311, 8069] innerCandidates61_28 8380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_29

def innerTail61_27 : List Nat := 7559 :: innerTail61_28
def innerCandidates61_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates61_27_checked : candidates edge 4 311 7559 innerTail61_28 best.bound = innerCandidates61_27 := by
  unfold candidates
  have row : edge 7559 = SearchData.row7559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_27 : scan edge (search edge 3) 4 [311] 311 innerTail61_27 best = best := by
  rw [innerTail61_27, scan, if_neg (by decide), innerCandidates61_27_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_28
    (if 3 ≤ innerCandidates61_27.length then search edge 3 [311, 7559] innerCandidates61_27 7870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_28

def innerTail61_26 : List Nat := 7523 :: innerTail61_27
def innerCandidates61_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates61_26_checked : candidates edge 4 311 7523 innerTail61_27 best.bound = innerCandidates61_26 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_26 : scan edge (search edge 3) 4 [311] 311 innerTail61_26 best = best := by
  rw [innerTail61_26, scan, if_neg (by decide), innerCandidates61_26_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_27
    (if 3 ≤ innerCandidates61_26.length then search edge 3 [311, 7523] innerCandidates61_26 7834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_27

def innerTail61_25 : List Nat := 7211 :: innerTail61_26
def innerCandidates61_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates61_25_checked : candidates edge 4 311 7211 innerTail61_26 best.bound = innerCandidates61_25 := by
  unfold candidates
  have row : edge 7211 = SearchData.row7211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_25 : scan edge (search edge 3) 4 [311] 311 innerTail61_25 best = best := by
  rw [innerTail61_25, scan, if_neg (by decide), innerCandidates61_25_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_26
    (if 3 ≤ innerCandidates61_25.length then search edge 3 [311, 7211] innerCandidates61_25 7522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_26

def innerTail61_24 : List Nat := 7013 :: innerTail61_25
def innerCandidates61_24 : List Nat := [7211]
set_option maxRecDepth 4096 in
theorem innerCandidates61_24_checked : candidates edge 4 311 7013 innerTail61_25 best.bound = innerCandidates61_24 := by
  unfold candidates
  have row : edge 7013 = SearchData.row7013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_24 : scan edge (search edge 3) 4 [311] 311 innerTail61_24 best = best := by
  rw [innerTail61_24, scan, if_neg (by decide), innerCandidates61_24_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_25
    (if 3 ≤ innerCandidates61_24.length then search edge 3 [311, 7013] innerCandidates61_24 7324 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_25

def innerTail61_23 : List Nat := 6761 :: innerTail61_24
def innerCandidates61_23 : List Nat := [8387]
set_option maxRecDepth 4096 in
theorem innerCandidates61_23_checked : candidates edge 4 311 6761 innerTail61_24 best.bound = innerCandidates61_23 := by
  unfold candidates
  have row : edge 6761 = SearchData.row6761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_23 : scan edge (search edge 3) 4 [311] 311 innerTail61_23 best = best := by
  rw [innerTail61_23, scan, if_neg (by decide), innerCandidates61_23_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_24
    (if 3 ≤ innerCandidates61_23.length then search edge 3 [311, 6761] innerCandidates61_23 7072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_24

def innerTail61_22 : List Nat := 6263 :: innerTail61_23
def innerCandidates61_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates61_22_checked : candidates edge 4 311 6263 innerTail61_23 best.bound = innerCandidates61_22 := by
  unfold candidates
  have row : edge 6263 = SearchData.row6263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_22 : scan edge (search edge 3) 4 [311] 311 innerTail61_22 best = best := by
  rw [innerTail61_22, scan, if_neg (by decide), innerCandidates61_22_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_23
    (if 3 ≤ innerCandidates61_22.length then search edge 3 [311, 6263] innerCandidates61_22 6574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_23

def innerTail61_21 : List Nat := 6131 :: innerTail61_22
def innerCandidates61_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates61_21_checked : candidates edge 4 311 6131 innerTail61_22 best.bound = innerCandidates61_21 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_21 : scan edge (search edge 3) 4 [311] 311 innerTail61_21 best = best := by
  rw [innerTail61_21, scan, if_neg (by decide), innerCandidates61_21_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_22
    (if 3 ≤ innerCandidates61_21.length then search edge 3 [311, 6131] innerCandidates61_21 6442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_22

def innerTail61_20 : List Nat := 6011 :: innerTail61_21
def innerCandidates61_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates61_20_checked : candidates edge 4 311 6011 innerTail61_21 best.bound = innerCandidates61_20 := by
  unfold candidates
  have row : edge 6011 = SearchData.row6011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_20 : scan edge (search edge 3) 4 [311] 311 innerTail61_20 best = best := by
  rw [innerTail61_20, scan, if_neg (by decide), innerCandidates61_20_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_21
    (if 3 ≤ innerCandidates61_20.length then search edge 3 [311, 6011] innerCandidates61_20 6322 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_21

def innerTail61_19 : List Nat := 5279 :: innerTail61_20
def innerCandidates61_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates61_19_checked : candidates edge 4 311 5279 innerTail61_20 best.bound = innerCandidates61_19 := by
  unfold candidates
  have row : edge 5279 = SearchData.row5279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_19 : scan edge (search edge 3) 4 [311] 311 innerTail61_19 best = best := by
  rw [innerTail61_19, scan, if_neg (by decide), innerCandidates61_19_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_20
    (if 3 ≤ innerCandidates61_19.length then search edge 3 [311, 5279] innerCandidates61_19 5590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_20

def innerTail61_18 : List Nat := 5237 :: innerTail61_19
def innerCandidates61_18 : List Nat := [6761]
set_option maxRecDepth 4096 in
theorem innerCandidates61_18_checked : candidates edge 4 311 5237 innerTail61_19 best.bound = innerCandidates61_18 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_18 : scan edge (search edge 3) 4 [311] 311 innerTail61_18 best = best := by
  rw [innerTail61_18, scan, if_neg (by decide), innerCandidates61_18_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_19
    (if 3 ≤ innerCandidates61_18.length then search edge 3 [311, 5237] innerCandidates61_18 5548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_19

def innerTail61_17 : List Nat := 5171 :: innerTail61_18
def innerCandidates61_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates61_17_checked : candidates edge 4 311 5171 innerTail61_18 best.bound = innerCandidates61_17 := by
  unfold candidates
  have row : edge 5171 = SearchData.row5171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_17 : scan edge (search edge 3) 4 [311] 311 innerTail61_17 best = best := by
  rw [innerTail61_17, scan, if_neg (by decide), innerCandidates61_17_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_18
    (if 3 ≤ innerCandidates61_17.length then search edge 3 [311, 5171] innerCandidates61_17 5482 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_18

def innerTail61_16 : List Nat := 4967 :: innerTail61_17
def innerCandidates61_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates61_16_checked : candidates edge 4 311 4967 innerTail61_17 best.bound = innerCandidates61_16 := by
  unfold candidates
  have row : edge 4967 = SearchData.row4967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_16 : scan edge (search edge 3) 4 [311] 311 innerTail61_16 best = best := by
  rw [innerTail61_16, scan, if_neg (by decide), innerCandidates61_16_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_17
    (if 3 ≤ innerCandidates61_16.length then search edge 3 [311, 4967] innerCandidates61_16 5278 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_17

def innerTail61_15 : List Nat := 4871 :: innerTail61_16
def innerCandidates61_15 : List Nat := [5237, 8069, 8237]
set_option maxRecDepth 4096 in
theorem innerCandidates61_15_checked : candidates edge 4 311 4871 innerTail61_16 best.bound = innerCandidates61_15 := by
  unfold candidates
  have row : edge 4871 = SearchData.row4871.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild61_15 : search edge 3 [311, 4871] innerCandidates61_15 5182 best = best := by decide

theorem innerScan61_15 : scan edge (search edge 3) 4 [311] 311 innerTail61_15 best = best := by
  rw [innerTail61_15, scan, if_neg (by decide), innerCandidates61_15_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_16
    (if 3 ≤ innerCandidates61_15.length then search edge 3 [311, 4871] innerCandidates61_15 5182 best else best) = best
  rw [if_pos (by decide), innerChild61_15]
  exact innerScan61_16

def innerTail61_14 : List Nat := 4019 :: innerTail61_15
def innerCandidates61_14 : List Nat := [7211]
set_option maxRecDepth 4096 in
theorem innerCandidates61_14_checked : candidates edge 4 311 4019 innerTail61_15 best.bound = innerCandidates61_14 := by
  unfold candidates
  have row : edge 4019 = SearchData.row4019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_14 : scan edge (search edge 3) 4 [311] 311 innerTail61_14 best = best := by
  rw [innerTail61_14, scan, if_neg (by decide), innerCandidates61_14_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_15
    (if 3 ≤ innerCandidates61_14.length then search edge 3 [311, 4019] innerCandidates61_14 4330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_15

def innerTail61_13 : List Nat := 3863 :: innerTail61_14
def innerCandidates61_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates61_13_checked : candidates edge 4 311 3863 innerTail61_14 best.bound = innerCandidates61_13 := by
  unfold candidates
  have row : edge 3863 = SearchData.row3863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_13 : scan edge (search edge 3) 4 [311] 311 innerTail61_13 best = best := by
  rw [innerTail61_13, scan, if_neg (by decide), innerCandidates61_13_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_14
    (if 3 ≤ innerCandidates61_13.length then search edge 3 [311, 3863] innerCandidates61_13 4174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_14

def innerTail61_12 : List Nat := 2909 :: innerTail61_13
def innerCandidates61_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates61_12_checked : candidates edge 4 311 2909 innerTail61_13 best.bound = innerCandidates61_12 := by
  unfold candidates
  have row : edge 2909 = SearchData.row2909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_12 : scan edge (search edge 3) 4 [311] 311 innerTail61_12 best = best := by
  rw [innerTail61_12, scan, if_neg (by decide), innerCandidates61_12_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_13
    (if 3 ≤ innerCandidates61_12.length then search edge 3 [311, 2909] innerCandidates61_12 3220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_13

def innerTail61_11 : List Nat := 2609 :: innerTail61_12
def innerCandidates61_11 : List Nat := [8069]
set_option maxRecDepth 4096 in
theorem innerCandidates61_11_checked : candidates edge 4 311 2609 innerTail61_12 best.bound = innerCandidates61_11 := by
  unfold candidates
  have row : edge 2609 = PrimePairCache.smallRow2609.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_11 : scan edge (search edge 3) 4 [311] 311 innerTail61_11 best = best := by
  rw [innerTail61_11, scan, if_neg (by decide), innerCandidates61_11_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_12
    (if 3 ≤ innerCandidates61_11.length then search edge 3 [311, 2609] innerCandidates61_11 2920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_12

def innerTail61_10 : List Nat := 2441 :: innerTail61_11
def innerCandidates61_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates61_10_checked : candidates edge 4 311 2441 innerTail61_11 best.bound = innerCandidates61_10 := by
  unfold candidates
  have row : edge 2441 = PrimePairCache.smallRow2441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_10 : scan edge (search edge 3) 4 [311] 311 innerTail61_10 best = best := by
  rw [innerTail61_10, scan, if_neg (by decide), innerCandidates61_10_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_11
    (if 3 ≤ innerCandidates61_10.length then search edge 3 [311, 2441] innerCandidates61_10 2752 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_11

def innerTail61_9 : List Nat := 2381 :: innerTail61_10
def innerCandidates61_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates61_9_checked : candidates edge 4 311 2381 innerTail61_10 best.bound = innerCandidates61_9 := by
  unfold candidates
  have row : edge 2381 = PrimePairCache.smallRow2381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_9 : scan edge (search edge 3) 4 [311] 311 innerTail61_9 best = best := by
  rw [innerTail61_9, scan, if_neg (by decide), innerCandidates61_9_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_10
    (if 3 ≤ innerCandidates61_9.length then search edge 3 [311, 2381] innerCandidates61_9 2692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_10

def innerTail61_8 : List Nat := 2141 :: innerTail61_9
def innerCandidates61_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates61_8_checked : candidates edge 4 311 2141 innerTail61_9 best.bound = innerCandidates61_8 := by
  unfold candidates
  have row : edge 2141 = PrimePairCache.smallRow2141.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_8 : scan edge (search edge 3) 4 [311] 311 innerTail61_8 best = best := by
  rw [innerTail61_8, scan, if_neg (by decide), innerCandidates61_8_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_9
    (if 3 ≤ innerCandidates61_8.length then search edge 3 [311, 2141] innerCandidates61_8 2452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_9

def innerTail61_7 : List Nat := 2111 :: innerTail61_8
def innerCandidates61_7 : List Nat := [4019]
set_option maxRecDepth 4096 in
theorem innerCandidates61_7_checked : candidates edge 4 311 2111 innerTail61_8 best.bound = innerCandidates61_7 := by
  unfold candidates
  have row : edge 2111 = PrimePairCache.smallRow2111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_7 : scan edge (search edge 3) 4 [311] 311 innerTail61_7 best = best := by
  rw [innerTail61_7, scan, if_neg (by decide), innerCandidates61_7_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_8
    (if 3 ≤ innerCandidates61_7.length then search edge 3 [311, 2111] innerCandidates61_7 2422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_8

def innerTail61_6 : List Nat := 1223 :: innerTail61_7
def innerCandidates61_6 : List Nat := [2141, 6131, 6761, 7523, 7559]
set_option maxRecDepth 4096 in
theorem innerCandidates61_6_checked : candidates edge 4 311 1223 innerTail61_7 best.bound = innerCandidates61_6 := by
  unfold candidates
  have row : edge 1223 = PrimePairCache.smallRow1223.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild61_6 : search edge 3 [311, 1223] innerCandidates61_6 1534 best = best := by decide

theorem innerScan61_6 : scan edge (search edge 3) 4 [311] 311 innerTail61_6 best = best := by
  rw [innerTail61_6, scan, if_neg (by decide), innerCandidates61_6_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_7
    (if 3 ≤ innerCandidates61_6.length then search edge 3 [311, 1223] innerCandidates61_6 1534 best else best) = best
  rw [if_pos (by decide), innerChild61_6]
  exact innerScan61_7

def innerTail61_5 : List Nat := 881 :: innerTail61_6
def innerCandidates61_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates61_5_checked : candidates edge 4 311 881 innerTail61_6 best.bound = innerCandidates61_5 := by
  unfold candidates
  have row : edge 881 = PrimePairCache.smallRow881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_5 : scan edge (search edge 3) 4 [311] 311 innerTail61_5 best = best := by
  rw [innerTail61_5, scan, if_neg (by decide), innerCandidates61_5_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_6
    (if 3 ≤ innerCandidates61_5.length then search edge 3 [311, 881] innerCandidates61_5 1192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_6

def innerTail61_4 : List Nat := 827 :: innerTail61_5
def innerCandidates61_4 : List Nat := [2909]
set_option maxRecDepth 4096 in
theorem innerCandidates61_4_checked : candidates edge 4 311 827 innerTail61_5 best.bound = innerCandidates61_4 := by
  unfold candidates
  have row : edge 827 = PrimePairCache.smallRow827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_4 : scan edge (search edge 3) 4 [311] 311 innerTail61_4 best = best := by
  rw [innerTail61_4, scan, if_neg (by decide), innerCandidates61_4_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_5
    (if 3 ≤ innerCandidates61_4.length then search edge 3 [311, 827] innerCandidates61_4 1138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_5

def innerTail61_3 : List Nat := 821 :: innerTail61_4
def innerCandidates61_3 : List Nat := [827, 7523]
set_option maxRecDepth 4096 in
theorem innerCandidates61_3_checked : candidates edge 4 311 821 innerTail61_4 best.bound = innerCandidates61_3 := by
  unfold candidates
  have row : edge 821 = PrimePairCache.smallRow821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_3 : scan edge (search edge 3) 4 [311] 311 innerTail61_3 best = best := by
  rw [innerTail61_3, scan, if_neg (by decide), innerCandidates61_3_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_4
    (if 3 ≤ innerCandidates61_3.length then search edge 3 [311, 821] innerCandidates61_3 1132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_4

def innerTail61_2 : List Nat := 677 :: innerTail61_3
def innerCandidates61_2 : List Nat := [827, 3863]
set_option maxRecDepth 4096 in
theorem innerCandidates61_2_checked : candidates edge 4 311 677 innerTail61_3 best.bound = innerCandidates61_2 := by
  unfold candidates
  have row : edge 677 = PrimePairCache.smallRow677.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_2 : scan edge (search edge 3) 4 [311] 311 innerTail61_2 best = best := by
  rw [innerTail61_2, scan, if_neg (by decide), innerCandidates61_2_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_3
    (if 3 ≤ innerCandidates61_2.length then search edge 3 [311, 677] innerCandidates61_2 988 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_3

def innerTail61_1 : List Nat := 653 :: innerTail61_2
def innerCandidates61_1 : List Nat := [7211]
set_option maxRecDepth 4096 in
theorem innerCandidates61_1_checked : candidates edge 4 311 653 innerTail61_2 best.bound = innerCandidates61_1 := by
  unfold candidates
  have row : edge 653 = PrimePairCache.smallRow653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_1 : scan edge (search edge 3) 4 [311] 311 innerTail61_1 best = best := by
  rw [innerTail61_1, scan, if_neg (by decide), innerCandidates61_1_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_2
    (if 3 ≤ innerCandidates61_1.length then search edge 3 [311, 653] innerCandidates61_1 964 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_2

def innerTail61_0 : List Nat := 359 :: innerTail61_1
def innerCandidates61_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates61_0_checked : candidates edge 4 311 359 innerTail61_1 best.bound = innerCandidates61_0 := by
  unfold candidates
  have row : edge 359 = PrimePairCache.smallRow359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan61_0 : scan edge (search edge 3) 4 [311] 311 innerTail61_0 best = best := by
  rw [innerTail61_0, scan, if_neg (by decide), innerCandidates61_0_checked]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_1
    (if 3 ≤ innerCandidates61_0.length then search edge 3 [311, 359] innerCandidates61_0 670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan61_1

theorem rootBranch61 : search edge 4 [311] rootCandidates61 311 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [311] 311 innerTail61_0 best = best
  exact innerScan61_0

end ProjectEulerSolutions.PrimePairUpper.Search
