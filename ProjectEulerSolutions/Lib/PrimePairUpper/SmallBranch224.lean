import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates54 : List Nat := [317, 389, 431, 461, 617, 887, 1979, 2003, 2741, 4001, 4157, 4457, 4733, 5051, 5237, 5471, 6053, 6329, 6521, 6563, 7109]
set_option maxRecDepth 4096 in
theorem rootCandidates54_checked : candidates edge 5 0 269 tail55 best.bound = rootCandidates54 := by
  unfold candidates
  have row : edge 269 = PrimePairCache.smallRow269.contains := by funext q; rfl
  rw [row]
  decide

def innerTail54_21 : List Nat := []
theorem innerScan54_21 : scan edge (search edge 3) 4 [269] 269 innerTail54_21 best = best := rfl

def innerTail54_20 : List Nat := 7109 :: innerTail54_21
def innerCandidates54_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates54_20_checked : candidates edge 4 269 7109 innerTail54_21 best.bound = innerCandidates54_20 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan54_20 : scan edge (search edge 3) 4 [269] 269 innerTail54_20 best = best := by
  rw [innerTail54_20, scan, if_neg (by decide), innerCandidates54_20_checked]
  change scan edge (search edge 3) 4 [269] 269 innerTail54_21
    (if 3 ≤ innerCandidates54_20.length then search edge 3 [269, 7109] innerCandidates54_20 7378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan54_21

def innerTail54_19 : List Nat := 6563 :: innerTail54_20
def innerCandidates54_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates54_19_checked : candidates edge 4 269 6563 innerTail54_20 best.bound = innerCandidates54_19 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan54_19 : scan edge (search edge 3) 4 [269] 269 innerTail54_19 best = best := by
  rw [innerTail54_19, scan, if_neg (by decide), innerCandidates54_19_checked]
  change scan edge (search edge 3) 4 [269] 269 innerTail54_20
    (if 3 ≤ innerCandidates54_19.length then search edge 3 [269, 6563] innerCandidates54_19 6832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan54_20

def innerTail54_18 : List Nat := 6521 :: innerTail54_19
def innerCandidates54_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates54_18_checked : candidates edge 4 269 6521 innerTail54_19 best.bound = innerCandidates54_18 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan54_18 : scan edge (search edge 3) 4 [269] 269 innerTail54_18 best = best := by
  rw [innerTail54_18, scan, if_neg (by decide), innerCandidates54_18_checked]
  change scan edge (search edge 3) 4 [269] 269 innerTail54_19
    (if 3 ≤ innerCandidates54_18.length then search edge 3 [269, 6521] innerCandidates54_18 6790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan54_19

def innerTail54_17 : List Nat := 6329 :: innerTail54_18
def innerCandidates54_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates54_17_checked : candidates edge 4 269 6329 innerTail54_18 best.bound = innerCandidates54_17 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan54_17 : scan edge (search edge 3) 4 [269] 269 innerTail54_17 best = best := by
  rw [innerTail54_17, scan, if_neg (by decide), innerCandidates54_17_checked]
  change scan edge (search edge 3) 4 [269] 269 innerTail54_18
    (if 3 ≤ innerCandidates54_17.length then search edge 3 [269, 6329] innerCandidates54_17 6598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan54_18

def innerTail54_16 : List Nat := 6053 :: innerTail54_17
def innerCandidates54_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates54_16_checked : candidates edge 4 269 6053 innerTail54_17 best.bound = innerCandidates54_16 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan54_16 : scan edge (search edge 3) 4 [269] 269 innerTail54_16 best = best := by
  rw [innerTail54_16, scan, if_neg (by decide), innerCandidates54_16_checked]
  change scan edge (search edge 3) 4 [269] 269 innerTail54_17
    (if 3 ≤ innerCandidates54_16.length then search edge 3 [269, 6053] innerCandidates54_16 6322 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan54_17

def innerTail54_15 : List Nat := 5471 :: innerTail54_16
def innerCandidates54_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates54_15_checked : candidates edge 4 269 5471 innerTail54_16 best.bound = innerCandidates54_15 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan54_15 : scan edge (search edge 3) 4 [269] 269 innerTail54_15 best = best := by
  rw [innerTail54_15, scan, if_neg (by decide), innerCandidates54_15_checked]
  change scan edge (search edge 3) 4 [269] 269 innerTail54_16
    (if 3 ≤ innerCandidates54_15.length then search edge 3 [269, 5471] innerCandidates54_15 5740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan54_16

def innerTail54_14 : List Nat := 5237 :: innerTail54_15
def innerCandidates54_14 : List Nat := [5471, 6563]
set_option maxRecDepth 4096 in
theorem innerCandidates54_14_checked : candidates edge 4 269 5237 innerTail54_15 best.bound = innerCandidates54_14 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan54_14 : scan edge (search edge 3) 4 [269] 269 innerTail54_14 best = best := by
  rw [innerTail54_14, scan, if_neg (by decide), innerCandidates54_14_checked]
  change scan edge (search edge 3) 4 [269] 269 innerTail54_15
    (if 3 ≤ innerCandidates54_14.length then search edge 3 [269, 5237] innerCandidates54_14 5506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan54_15

def innerTail54_13 : List Nat := 5051 :: innerTail54_14
def innerCandidates54_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates54_13_checked : candidates edge 4 269 5051 innerTail54_14 best.bound = innerCandidates54_13 := by
  unfold candidates
  have row : edge 5051 = SearchData.row5051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan54_13 : scan edge (search edge 3) 4 [269] 269 innerTail54_13 best = best := by
  rw [innerTail54_13, scan, if_neg (by decide), innerCandidates54_13_checked]
  change scan edge (search edge 3) 4 [269] 269 innerTail54_14
    (if 3 ≤ innerCandidates54_13.length then search edge 3 [269, 5051] innerCandidates54_13 5320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan54_14

def innerTail54_12 : List Nat := 4733 :: innerTail54_13
def innerCandidates54_12 : List Nat := [5237]
set_option maxRecDepth 4096 in
theorem innerCandidates54_12_checked : candidates edge 4 269 4733 innerTail54_13 best.bound = innerCandidates54_12 := by
  unfold candidates
  have row : edge 4733 = SearchData.row4733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan54_12 : scan edge (search edge 3) 4 [269] 269 innerTail54_12 best = best := by
  rw [innerTail54_12, scan, if_neg (by decide), innerCandidates54_12_checked]
  change scan edge (search edge 3) 4 [269] 269 innerTail54_13
    (if 3 ≤ innerCandidates54_12.length then search edge 3 [269, 4733] innerCandidates54_12 5002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan54_13

def innerTail54_11 : List Nat := 4457 :: innerTail54_12
def innerCandidates54_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates54_11_checked : candidates edge 4 269 4457 innerTail54_12 best.bound = innerCandidates54_11 := by
  unfold candidates
  have row : edge 4457 = SearchData.row4457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan54_11 : scan edge (search edge 3) 4 [269] 269 innerTail54_11 best = best := by
  rw [innerTail54_11, scan, if_neg (by decide), innerCandidates54_11_checked]
  change scan edge (search edge 3) 4 [269] 269 innerTail54_12
    (if 3 ≤ innerCandidates54_11.length then search edge 3 [269, 4457] innerCandidates54_11 4726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan54_12

def innerTail54_10 : List Nat := 4157 :: innerTail54_11
def innerCandidates54_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates54_10_checked : candidates edge 4 269 4157 innerTail54_11 best.bound = innerCandidates54_10 := by
  unfold candidates
  have row : edge 4157 = SearchData.row4157.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan54_10 : scan edge (search edge 3) 4 [269] 269 innerTail54_10 best = best := by
  rw [innerTail54_10, scan, if_neg (by decide), innerCandidates54_10_checked]
  change scan edge (search edge 3) 4 [269] 269 innerTail54_11
    (if 3 ≤ innerCandidates54_10.length then search edge 3 [269, 4157] innerCandidates54_10 4426 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan54_11

def innerTail54_9 : List Nat := 4001 :: innerTail54_10
def innerCandidates54_9 : List Nat := [6329, 6563]
set_option maxRecDepth 4096 in
theorem innerCandidates54_9_checked : candidates edge 4 269 4001 innerTail54_10 best.bound = innerCandidates54_9 := by
  unfold candidates
  have row : edge 4001 = SearchData.row4001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan54_9 : scan edge (search edge 3) 4 [269] 269 innerTail54_9 best = best := by
  rw [innerTail54_9, scan, if_neg (by decide), innerCandidates54_9_checked]
  change scan edge (search edge 3) 4 [269] 269 innerTail54_10
    (if 3 ≤ innerCandidates54_9.length then search edge 3 [269, 4001] innerCandidates54_9 4270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan54_10

def innerTail54_8 : List Nat := 2741 :: innerTail54_9
def innerCandidates54_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates54_8_checked : candidates edge 4 269 2741 innerTail54_9 best.bound = innerCandidates54_8 := by
  unfold candidates
  have row : edge 2741 = SearchData.row2741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan54_8 : scan edge (search edge 3) 4 [269] 269 innerTail54_8 best = best := by
  rw [innerTail54_8, scan, if_neg (by decide), innerCandidates54_8_checked]
  change scan edge (search edge 3) 4 [269] 269 innerTail54_9
    (if 3 ≤ innerCandidates54_8.length then search edge 3 [269, 2741] innerCandidates54_8 3010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan54_9

def innerTail54_7 : List Nat := 2003 :: innerTail54_8
def innerCandidates54_7 : List Nat := [5051, 6563]
set_option maxRecDepth 4096 in
theorem innerCandidates54_7_checked : candidates edge 4 269 2003 innerTail54_8 best.bound = innerCandidates54_7 := by
  unfold candidates
  have row : edge 2003 = PrimePairCache.smallRow2003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan54_7 : scan edge (search edge 3) 4 [269] 269 innerTail54_7 best = best := by
  rw [innerTail54_7, scan, if_neg (by decide), innerCandidates54_7_checked]
  change scan edge (search edge 3) 4 [269] 269 innerTail54_8
    (if 3 ≤ innerCandidates54_7.length then search edge 3 [269, 2003] innerCandidates54_7 2272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan54_8

def innerTail54_6 : List Nat := 1979 :: innerTail54_7
def innerCandidates54_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates54_6_checked : candidates edge 4 269 1979 innerTail54_7 best.bound = innerCandidates54_6 := by
  unfold candidates
  have row : edge 1979 = PrimePairCache.smallRow1979.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan54_6 : scan edge (search edge 3) 4 [269] 269 innerTail54_6 best = best := by
  rw [innerTail54_6, scan, if_neg (by decide), innerCandidates54_6_checked]
  change scan edge (search edge 3) 4 [269] 269 innerTail54_7
    (if 3 ≤ innerCandidates54_6.length then search edge 3 [269, 1979] innerCandidates54_6 2248 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan54_7

def innerTail54_5 : List Nat := 887 :: innerTail54_6
def innerCandidates54_5 : List Nat := [2741, 4157, 7109]
set_option maxRecDepth 4096 in
theorem innerCandidates54_5_checked : candidates edge 4 269 887 innerTail54_6 best.bound = innerCandidates54_5 := by
  unfold candidates
  have row : edge 887 = PrimePairCache.smallRow887.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild54_5 : search edge 3 [269, 887] innerCandidates54_5 1156 best = best := by decide

theorem innerScan54_5 : scan edge (search edge 3) 4 [269] 269 innerTail54_5 best = best := by
  rw [innerTail54_5, scan, if_neg (by decide), innerCandidates54_5_checked]
  change scan edge (search edge 3) 4 [269] 269 innerTail54_6
    (if 3 ≤ innerCandidates54_5.length then search edge 3 [269, 887] innerCandidates54_5 1156 best else best) = best
  rw [if_pos (by decide), innerChild54_5]
  exact innerScan54_6

def innerTail54_4 : List Nat := 617 :: innerTail54_5
def innerCandidates54_4 : List Nat := [887, 2741, 4001]
set_option maxRecDepth 4096 in
theorem innerCandidates54_4_checked : candidates edge 4 269 617 innerTail54_5 best.bound = innerCandidates54_4 := by
  unfold candidates
  have row : edge 617 = PrimePairCache.smallRow617.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild54_4 : search edge 3 [269, 617] innerCandidates54_4 886 best = best := by decide

theorem innerScan54_4 : scan edge (search edge 3) 4 [269] 269 innerTail54_4 best = best := by
  rw [innerTail54_4, scan, if_neg (by decide), innerCandidates54_4_checked]
  change scan edge (search edge 3) 4 [269] 269 innerTail54_5
    (if 3 ≤ innerCandidates54_4.length then search edge 3 [269, 617] innerCandidates54_4 886 best else best) = best
  rw [if_pos (by decide), innerChild54_4]
  exact innerScan54_5

def innerTail54_3 : List Nat := 461 :: innerTail54_4
def innerCandidates54_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates54_3_checked : candidates edge 4 269 461 innerTail54_4 best.bound = innerCandidates54_3 := by
  unfold candidates
  have row : edge 461 = PrimePairCache.smallRow461.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan54_3 : scan edge (search edge 3) 4 [269] 269 innerTail54_3 best = best := by
  rw [innerTail54_3, scan, if_neg (by decide), innerCandidates54_3_checked]
  change scan edge (search edge 3) 4 [269] 269 innerTail54_4
    (if 3 ≤ innerCandidates54_3.length then search edge 3 [269, 461] innerCandidates54_3 730 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan54_4

def innerTail54_2 : List Nat := 431 :: innerTail54_3
def innerCandidates54_2 : List Nat := [6329]
set_option maxRecDepth 4096 in
theorem innerCandidates54_2_checked : candidates edge 4 269 431 innerTail54_3 best.bound = innerCandidates54_2 := by
  unfold candidates
  have row : edge 431 = PrimePairCache.smallRow431.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan54_2 : scan edge (search edge 3) 4 [269] 269 innerTail54_2 best = best := by
  rw [innerTail54_2, scan, if_neg (by decide), innerCandidates54_2_checked]
  change scan edge (search edge 3) 4 [269] 269 innerTail54_3
    (if 3 ≤ innerCandidates54_2.length then search edge 3 [269, 431] innerCandidates54_2 700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan54_3

def innerTail54_1 : List Nat := 389 :: innerTail54_2
def innerCandidates54_1 : List Nat := [4733]
set_option maxRecDepth 4096 in
theorem innerCandidates54_1_checked : candidates edge 4 269 389 innerTail54_2 best.bound = innerCandidates54_1 := by
  unfold candidates
  have row : edge 389 = PrimePairCache.smallRow389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan54_1 : scan edge (search edge 3) 4 [269] 269 innerTail54_1 best = best := by
  rw [innerTail54_1, scan, if_neg (by decide), innerCandidates54_1_checked]
  change scan edge (search edge 3) 4 [269] 269 innerTail54_2
    (if 3 ≤ innerCandidates54_1.length then search edge 3 [269, 389] innerCandidates54_1 658 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan54_2

def innerTail54_0 : List Nat := 317 :: innerTail54_1
def innerCandidates54_0 : List Nat := [6053]
set_option maxRecDepth 4096 in
theorem innerCandidates54_0_checked : candidates edge 4 269 317 innerTail54_1 best.bound = innerCandidates54_0 := by
  unfold candidates
  have row : edge 317 = PrimePairCache.smallRow317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan54_0 : scan edge (search edge 3) 4 [269] 269 innerTail54_0 best = best := by
  rw [innerTail54_0, scan, if_neg (by decide), innerCandidates54_0_checked]
  change scan edge (search edge 3) 4 [269] 269 innerTail54_1
    (if 3 ≤ innerCandidates54_0.length then search edge 3 [269, 317] innerCandidates54_0 586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan54_1

theorem rootBranch54 : search edge 4 [269] rootCandidates54 269 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [269] 269 innerTail54_0 best = best
  exact innerScan54_0

def rootCandidates53 : List Nat := [293, 443, 647, 761, 821, 941, 1049, 1091, 2099, 2129, 2339, 2477, 2633, 2789, 2999, 3359, 3467, 3557, 3923, 4523, 5189, 5693, 5987, 6197, 6701, 7013, 7043, 7253, 7451, 7937, 8087]
set_option maxRecDepth 4096 in
theorem rootCandidates53_checked : candidates edge 5 0 263 tail54 best.bound = rootCandidates53 := by
  unfold candidates
  have row : edge 263 = PrimePairCache.smallRow263.contains := by funext q; rfl
  rw [row]
  decide

def innerTail53_31 : List Nat := []
theorem innerScan53_31 : scan edge (search edge 3) 4 [263] 263 innerTail53_31 best = best := rfl

def innerTail53_30 : List Nat := 8087 :: innerTail53_31
def innerCandidates53_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates53_30_checked : candidates edge 4 263 8087 innerTail53_31 best.bound = innerCandidates53_30 := by
  unfold candidates
  have row : edge 8087 = SearchData.row8087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_30 : scan edge (search edge 3) 4 [263] 263 innerTail53_30 best = best := by
  rw [innerTail53_30, scan, if_neg (by decide), innerCandidates53_30_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_31
    (if 3 ≤ innerCandidates53_30.length then search edge 3 [263, 8087] innerCandidates53_30 8350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_31

def innerTail53_29 : List Nat := 7937 :: innerTail53_30
def innerCandidates53_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates53_29_checked : candidates edge 4 263 7937 innerTail53_30 best.bound = innerCandidates53_29 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_29 : scan edge (search edge 3) 4 [263] 263 innerTail53_29 best = best := by
  rw [innerTail53_29, scan, if_neg (by decide), innerCandidates53_29_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_30
    (if 3 ≤ innerCandidates53_29.length then search edge 3 [263, 7937] innerCandidates53_29 8200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_30

def innerTail53_28 : List Nat := 7451 :: innerTail53_29
def innerCandidates53_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates53_28_checked : candidates edge 4 263 7451 innerTail53_29 best.bound = innerCandidates53_28 := by
  unfold candidates
  have row : edge 7451 = SearchData.row7451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_28 : scan edge (search edge 3) 4 [263] 263 innerTail53_28 best = best := by
  rw [innerTail53_28, scan, if_neg (by decide), innerCandidates53_28_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_29
    (if 3 ≤ innerCandidates53_28.length then search edge 3 [263, 7451] innerCandidates53_28 7714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_29

def innerTail53_27 : List Nat := 7253 :: innerTail53_28
def innerCandidates53_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates53_27_checked : candidates edge 4 263 7253 innerTail53_28 best.bound = innerCandidates53_27 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_27 : scan edge (search edge 3) 4 [263] 263 innerTail53_27 best = best := by
  rw [innerTail53_27, scan, if_neg (by decide), innerCandidates53_27_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_28
    (if 3 ≤ innerCandidates53_27.length then search edge 3 [263, 7253] innerCandidates53_27 7516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_28

def innerTail53_26 : List Nat := 7043 :: innerTail53_27
def innerCandidates53_26 : List Nat := [7253]
set_option maxRecDepth 4096 in
theorem innerCandidates53_26_checked : candidates edge 4 263 7043 innerTail53_27 best.bound = innerCandidates53_26 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_26 : scan edge (search edge 3) 4 [263] 263 innerTail53_26 best = best := by
  rw [innerTail53_26, scan, if_neg (by decide), innerCandidates53_26_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_27
    (if 3 ≤ innerCandidates53_26.length then search edge 3 [263, 7043] innerCandidates53_26 7306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_27

def innerTail53_25 : List Nat := 7013 :: innerTail53_26
def innerCandidates53_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates53_25_checked : candidates edge 4 263 7013 innerTail53_26 best.bound = innerCandidates53_25 := by
  unfold candidates
  have row : edge 7013 = SearchData.row7013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_25 : scan edge (search edge 3) 4 [263] 263 innerTail53_25 best = best := by
  rw [innerTail53_25, scan, if_neg (by decide), innerCandidates53_25_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_26
    (if 3 ≤ innerCandidates53_25.length then search edge 3 [263, 7013] innerCandidates53_25 7276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_26

def innerTail53_24 : List Nat := 6701 :: innerTail53_25
def innerCandidates53_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates53_24_checked : candidates edge 4 263 6701 innerTail53_25 best.bound = innerCandidates53_24 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_24 : scan edge (search edge 3) 4 [263] 263 innerTail53_24 best = best := by
  rw [innerTail53_24, scan, if_neg (by decide), innerCandidates53_24_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_25
    (if 3 ≤ innerCandidates53_24.length then search edge 3 [263, 6701] innerCandidates53_24 6964 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_25

def innerTail53_23 : List Nat := 6197 :: innerTail53_24
def innerCandidates53_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates53_23_checked : candidates edge 4 263 6197 innerTail53_24 best.bound = innerCandidates53_23 := by
  unfold candidates
  have row : edge 6197 = SearchData.row6197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_23 : scan edge (search edge 3) 4 [263] 263 innerTail53_23 best = best := by
  rw [innerTail53_23, scan, if_neg (by decide), innerCandidates53_23_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_24
    (if 3 ≤ innerCandidates53_23.length then search edge 3 [263, 6197] innerCandidates53_23 6460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_24

def innerTail53_22 : List Nat := 5987 :: innerTail53_23
def innerCandidates53_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates53_22_checked : candidates edge 4 263 5987 innerTail53_23 best.bound = innerCandidates53_22 := by
  unfold candidates
  have row : edge 5987 = SearchData.row5987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_22 : scan edge (search edge 3) 4 [263] 263 innerTail53_22 best = best := by
  rw [innerTail53_22, scan, if_neg (by decide), innerCandidates53_22_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_23
    (if 3 ≤ innerCandidates53_22.length then search edge 3 [263, 5987] innerCandidates53_22 6250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_23

def innerTail53_21 : List Nat := 5693 :: innerTail53_22
def innerCandidates53_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates53_21_checked : candidates edge 4 263 5693 innerTail53_22 best.bound = innerCandidates53_21 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_21 : scan edge (search edge 3) 4 [263] 263 innerTail53_21 best = best := by
  rw [innerTail53_21, scan, if_neg (by decide), innerCandidates53_21_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_22
    (if 3 ≤ innerCandidates53_21.length then search edge 3 [263, 5693] innerCandidates53_21 5956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_22

def innerTail53_20 : List Nat := 5189 :: innerTail53_21
def innerCandidates53_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates53_20_checked : candidates edge 4 263 5189 innerTail53_21 best.bound = innerCandidates53_20 := by
  unfold candidates
  have row : edge 5189 = SearchData.row5189.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_20 : scan edge (search edge 3) 4 [263] 263 innerTail53_20 best = best := by
  rw [innerTail53_20, scan, if_neg (by decide), innerCandidates53_20_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_21
    (if 3 ≤ innerCandidates53_20.length then search edge 3 [263, 5189] innerCandidates53_20 5452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_21

def innerTail53_19 : List Nat := 4523 :: innerTail53_20
def innerCandidates53_19 : List Nat := [7937]
set_option maxRecDepth 4096 in
theorem innerCandidates53_19_checked : candidates edge 4 263 4523 innerTail53_20 best.bound = innerCandidates53_19 := by
  unfold candidates
  have row : edge 4523 = SearchData.row4523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_19 : scan edge (search edge 3) 4 [263] 263 innerTail53_19 best = best := by
  rw [innerTail53_19, scan, if_neg (by decide), innerCandidates53_19_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_20
    (if 3 ≤ innerCandidates53_19.length then search edge 3 [263, 4523] innerCandidates53_19 4786 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_20

def innerTail53_18 : List Nat := 3923 :: innerTail53_19
def innerCandidates53_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates53_18_checked : candidates edge 4 263 3923 innerTail53_19 best.bound = innerCandidates53_18 := by
  unfold candidates
  have row : edge 3923 = SearchData.row3923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_18 : scan edge (search edge 3) 4 [263] 263 innerTail53_18 best = best := by
  rw [innerTail53_18, scan, if_neg (by decide), innerCandidates53_18_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_19
    (if 3 ≤ innerCandidates53_18.length then search edge 3 [263, 3923] innerCandidates53_18 4186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_19

def innerTail53_17 : List Nat := 3557 :: innerTail53_18
def innerCandidates53_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates53_17_checked : candidates edge 4 263 3557 innerTail53_18 best.bound = innerCandidates53_17 := by
  unfold candidates
  have row : edge 3557 = SearchData.row3557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_17 : scan edge (search edge 3) 4 [263] 263 innerTail53_17 best = best := by
  rw [innerTail53_17, scan, if_neg (by decide), innerCandidates53_17_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_18
    (if 3 ≤ innerCandidates53_17.length then search edge 3 [263, 3557] innerCandidates53_17 3820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_18

def innerTail53_16 : List Nat := 3467 :: innerTail53_17
def innerCandidates53_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates53_16_checked : candidates edge 4 263 3467 innerTail53_17 best.bound = innerCandidates53_16 := by
  unfold candidates
  have row : edge 3467 = SearchData.row3467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_16 : scan edge (search edge 3) 4 [263] 263 innerTail53_16 best = best := by
  rw [innerTail53_16, scan, if_neg (by decide), innerCandidates53_16_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_17
    (if 3 ≤ innerCandidates53_16.length then search edge 3 [263, 3467] innerCandidates53_16 3730 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_17

def innerTail53_15 : List Nat := 3359 :: innerTail53_16
def innerCandidates53_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates53_15_checked : candidates edge 4 263 3359 innerTail53_16 best.bound = innerCandidates53_15 := by
  unfold candidates
  have row : edge 3359 = SearchData.row3359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_15 : scan edge (search edge 3) 4 [263] 263 innerTail53_15 best = best := by
  rw [innerTail53_15, scan, if_neg (by decide), innerCandidates53_15_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_16
    (if 3 ≤ innerCandidates53_15.length then search edge 3 [263, 3359] innerCandidates53_15 3622 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_16

def innerTail53_14 : List Nat := 2999 :: innerTail53_15
def innerCandidates53_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates53_14_checked : candidates edge 4 263 2999 innerTail53_15 best.bound = innerCandidates53_14 := by
  unfold candidates
  have row : edge 2999 = SearchData.row2999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_14 : scan edge (search edge 3) 4 [263] 263 innerTail53_14 best = best := by
  rw [innerTail53_14, scan, if_neg (by decide), innerCandidates53_14_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_15
    (if 3 ≤ innerCandidates53_14.length then search edge 3 [263, 2999] innerCandidates53_14 3262 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_15

def innerTail53_13 : List Nat := 2789 :: innerTail53_14
def innerCandidates53_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates53_13_checked : candidates edge 4 263 2789 innerTail53_14 best.bound = innerCandidates53_13 := by
  unfold candidates
  have row : edge 2789 = SearchData.row2789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_13 : scan edge (search edge 3) 4 [263] 263 innerTail53_13 best = best := by
  rw [innerTail53_13, scan, if_neg (by decide), innerCandidates53_13_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_14
    (if 3 ≤ innerCandidates53_13.length then search edge 3 [263, 2789] innerCandidates53_13 3052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_14

def innerTail53_12 : List Nat := 2633 :: innerTail53_13
def innerCandidates53_12 : List Nat := [5693]
set_option maxRecDepth 4096 in
theorem innerCandidates53_12_checked : candidates edge 4 263 2633 innerTail53_13 best.bound = innerCandidates53_12 := by
  unfold candidates
  have row : edge 2633 = PrimePairCache.smallRow2633.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_12 : scan edge (search edge 3) 4 [263] 263 innerTail53_12 best = best := by
  rw [innerTail53_12, scan, if_neg (by decide), innerCandidates53_12_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_13
    (if 3 ≤ innerCandidates53_12.length then search edge 3 [263, 2633] innerCandidates53_12 2896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_13

def innerTail53_11 : List Nat := 2477 :: innerTail53_12
def innerCandidates53_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates53_11_checked : candidates edge 4 263 2477 innerTail53_12 best.bound = innerCandidates53_11 := by
  unfold candidates
  have row : edge 2477 = PrimePairCache.smallRow2477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_11 : scan edge (search edge 3) 4 [263] 263 innerTail53_11 best = best := by
  rw [innerTail53_11, scan, if_neg (by decide), innerCandidates53_11_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_12
    (if 3 ≤ innerCandidates53_11.length then search edge 3 [263, 2477] innerCandidates53_11 2740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_12

def innerTail53_10 : List Nat := 2339 :: innerTail53_11
def innerCandidates53_10 : List Nat := [7013]
set_option maxRecDepth 4096 in
theorem innerCandidates53_10_checked : candidates edge 4 263 2339 innerTail53_11 best.bound = innerCandidates53_10 := by
  unfold candidates
  have row : edge 2339 = PrimePairCache.smallRow2339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_10 : scan edge (search edge 3) 4 [263] 263 innerTail53_10 best = best := by
  rw [innerTail53_10, scan, if_neg (by decide), innerCandidates53_10_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_11
    (if 3 ≤ innerCandidates53_10.length then search edge 3 [263, 2339] innerCandidates53_10 2602 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_11

def innerTail53_9 : List Nat := 2129 :: innerTail53_10
def innerCandidates53_9 : List Nat := [7937]
set_option maxRecDepth 4096 in
theorem innerCandidates53_9_checked : candidates edge 4 263 2129 innerTail53_10 best.bound = innerCandidates53_9 := by
  unfold candidates
  have row : edge 2129 = PrimePairCache.smallRow2129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_9 : scan edge (search edge 3) 4 [263] 263 innerTail53_9 best = best := by
  rw [innerTail53_9, scan, if_neg (by decide), innerCandidates53_9_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_10
    (if 3 ≤ innerCandidates53_9.length then search edge 3 [263, 2129] innerCandidates53_9 2392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_10

def innerTail53_8 : List Nat := 2099 :: innerTail53_9
def innerCandidates53_8 : List Nat := [2129]
set_option maxRecDepth 4096 in
theorem innerCandidates53_8_checked : candidates edge 4 263 2099 innerTail53_9 best.bound = innerCandidates53_8 := by
  unfold candidates
  have row : edge 2099 = PrimePairCache.smallRow2099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_8 : scan edge (search edge 3) 4 [263] 263 innerTail53_8 best = best := by
  rw [innerTail53_8, scan, if_neg (by decide), innerCandidates53_8_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_9
    (if 3 ≤ innerCandidates53_8.length then search edge 3 [263, 2099] innerCandidates53_8 2362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_9

def innerTail53_7 : List Nat := 1091 :: innerTail53_8
def innerCandidates53_7 : List Nat := [2633]
set_option maxRecDepth 4096 in
theorem innerCandidates53_7_checked : candidates edge 4 263 1091 innerTail53_8 best.bound = innerCandidates53_7 := by
  unfold candidates
  have row : edge 1091 = PrimePairCache.smallRow1091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_7 : scan edge (search edge 3) 4 [263] 263 innerTail53_7 best = best := by
  rw [innerTail53_7, scan, if_neg (by decide), innerCandidates53_7_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_8
    (if 3 ≤ innerCandidates53_7.length then search edge 3 [263, 1091] innerCandidates53_7 1354 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_8

def innerTail53_6 : List Nat := 1049 :: innerTail53_7
def innerCandidates53_6 : List Nat := [2633, 7013]
set_option maxRecDepth 4096 in
theorem innerCandidates53_6_checked : candidates edge 4 263 1049 innerTail53_7 best.bound = innerCandidates53_6 := by
  unfold candidates
  have row : edge 1049 = PrimePairCache.smallRow1049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_6 : scan edge (search edge 3) 4 [263] 263 innerTail53_6 best = best := by
  rw [innerTail53_6, scan, if_neg (by decide), innerCandidates53_6_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_7
    (if 3 ≤ innerCandidates53_6.length then search edge 3 [263, 1049] innerCandidates53_6 1312 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_7

def innerTail53_5 : List Nat := 941 :: innerTail53_6
def innerCandidates53_5 : List Nat := [1049, 2099]
set_option maxRecDepth 4096 in
theorem innerCandidates53_5_checked : candidates edge 4 263 941 innerTail53_6 best.bound = innerCandidates53_5 := by
  unfold candidates
  have row : edge 941 = PrimePairCache.smallRow941.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_5 : scan edge (search edge 3) 4 [263] 263 innerTail53_5 best = best := by
  rw [innerTail53_5, scan, if_neg (by decide), innerCandidates53_5_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_6
    (if 3 ≤ innerCandidates53_5.length then search edge 3 [263, 941] innerCandidates53_5 1204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_6

def innerTail53_4 : List Nat := 821 :: innerTail53_5
def innerCandidates53_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates53_4_checked : candidates edge 4 263 821 innerTail53_5 best.bound = innerCandidates53_4 := by
  unfold candidates
  have row : edge 821 = PrimePairCache.smallRow821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_4 : scan edge (search edge 3) 4 [263] 263 innerTail53_4 best = best := by
  rw [innerTail53_4, scan, if_neg (by decide), innerCandidates53_4_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_5
    (if 3 ≤ innerCandidates53_4.length then search edge 3 [263, 821] innerCandidates53_4 1084 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_5

def innerTail53_3 : List Nat := 761 :: innerTail53_4
def innerCandidates53_3 : List Nat := [3923]
set_option maxRecDepth 4096 in
theorem innerCandidates53_3_checked : candidates edge 4 263 761 innerTail53_4 best.bound = innerCandidates53_3 := by
  unfold candidates
  have row : edge 761 = PrimePairCache.smallRow761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_3 : scan edge (search edge 3) 4 [263] 263 innerTail53_3 best = best := by
  rw [innerTail53_3, scan, if_neg (by decide), innerCandidates53_3_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_4
    (if 3 ≤ innerCandidates53_3.length then search edge 3 [263, 761] innerCandidates53_3 1024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_4

def innerTail53_2 : List Nat := 647 :: innerTail53_3
def innerCandidates53_2 : List Nat := [821, 7253]
set_option maxRecDepth 4096 in
theorem innerCandidates53_2_checked : candidates edge 4 263 647 innerTail53_3 best.bound = innerCandidates53_2 := by
  unfold candidates
  have row : edge 647 = PrimePairCache.smallRow647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_2 : scan edge (search edge 3) 4 [263] 263 innerTail53_2 best = best := by
  rw [innerTail53_2, scan, if_neg (by decide), innerCandidates53_2_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_3
    (if 3 ≤ innerCandidates53_2.length then search edge 3 [263, 647] innerCandidates53_2 910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_3

def innerTail53_1 : List Nat := 443 :: innerTail53_2
def innerCandidates53_1 : List Nat := [761]
set_option maxRecDepth 4096 in
theorem innerCandidates53_1_checked : candidates edge 4 263 443 innerTail53_2 best.bound = innerCandidates53_1 := by
  unfold candidates
  have row : edge 443 = PrimePairCache.smallRow443.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_1 : scan edge (search edge 3) 4 [263] 263 innerTail53_1 best = best := by
  rw [innerTail53_1, scan, if_neg (by decide), innerCandidates53_1_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_2
    (if 3 ≤ innerCandidates53_1.length then search edge 3 [263, 443] innerCandidates53_1 706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_2

def innerTail53_0 : List Nat := 293 :: innerTail53_1
def innerCandidates53_0 : List Nat := [5189]
set_option maxRecDepth 4096 in
theorem innerCandidates53_0_checked : candidates edge 4 263 293 innerTail53_1 best.bound = innerCandidates53_0 := by
  unfold candidates
  have row : edge 293 = PrimePairCache.smallRow293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan53_0 : scan edge (search edge 3) 4 [263] 263 innerTail53_0 best = best := by
  rw [innerTail53_0, scan, if_neg (by decide), innerCandidates53_0_checked]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_1
    (if 3 ≤ innerCandidates53_0.length then search edge 3 [263, 293] innerCandidates53_0 556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan53_1

theorem rootBranch53 : search edge 4 [263] rootCandidates53 263 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [263] 263 innerTail53_0 best = best
  exact innerScan53_0

end ProjectEulerSolutions.PrimePairUpper.Search
