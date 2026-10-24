import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates86 : List Nat := [479, 569, 653, 677, 983, 1181, 1307, 1487, 1571, 1697, 2339, 2549, 3449, 3881, 4217, 4271, 4637, 4937, 5783, 6473, 6977, 7187, 7451]
set_option maxRecDepth 4096 in
theorem rootCandidates86_checked : candidates edge 5 0 461 tail87 best.bound = rootCandidates86 := by
  unfold candidates
  have row : edge 461 = PrimePairCache.smallRow461.contains := by funext q; rfl
  rw [row]
  decide

def innerTail86_23 : List Nat := []
theorem innerScan86_23 : scan edge (search edge 3) 4 [461] 461 innerTail86_23 best = best := rfl

def innerTail86_22 : List Nat := 7451 :: innerTail86_23
def innerCandidates86_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates86_22_checked : candidates edge 4 461 7451 innerTail86_23 best.bound = innerCandidates86_22 := by
  unfold candidates
  have row : edge 7451 = SearchData.row7451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_22 : scan edge (search edge 3) 4 [461] 461 innerTail86_22 best = best := by
  rw [innerTail86_22, scan, if_neg (by decide), innerCandidates86_22_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_23
    (if 3 ≤ innerCandidates86_22.length then search edge 3 [461, 7451] innerCandidates86_22 7912 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_23

def innerTail86_21 : List Nat := 7187 :: innerTail86_22
def innerCandidates86_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates86_21_checked : candidates edge 4 461 7187 innerTail86_22 best.bound = innerCandidates86_21 := by
  unfold candidates
  have row : edge 7187 = SearchData.row7187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_21 : scan edge (search edge 3) 4 [461] 461 innerTail86_21 best = best := by
  rw [innerTail86_21, scan, if_neg (by decide), innerCandidates86_21_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_22
    (if 3 ≤ innerCandidates86_21.length then search edge 3 [461, 7187] innerCandidates86_21 7648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_22

def innerTail86_20 : List Nat := 6977 :: innerTail86_21
def innerCandidates86_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates86_20_checked : candidates edge 4 461 6977 innerTail86_21 best.bound = innerCandidates86_20 := by
  unfold candidates
  have row : edge 6977 = SearchData.row6977.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_20 : scan edge (search edge 3) 4 [461] 461 innerTail86_20 best = best := by
  rw [innerTail86_20, scan, if_neg (by decide), innerCandidates86_20_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_21
    (if 3 ≤ innerCandidates86_20.length then search edge 3 [461, 6977] innerCandidates86_20 7438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_21

def innerTail86_19 : List Nat := 6473 :: innerTail86_20
def innerCandidates86_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates86_19_checked : candidates edge 4 461 6473 innerTail86_20 best.bound = innerCandidates86_19 := by
  unfold candidates
  have row : edge 6473 = SearchData.row6473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_19 : scan edge (search edge 3) 4 [461] 461 innerTail86_19 best = best := by
  rw [innerTail86_19, scan, if_neg (by decide), innerCandidates86_19_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_20
    (if 3 ≤ innerCandidates86_19.length then search edge 3 [461, 6473] innerCandidates86_19 6934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_20

def innerTail86_18 : List Nat := 5783 :: innerTail86_19
def innerCandidates86_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates86_18_checked : candidates edge 4 461 5783 innerTail86_19 best.bound = innerCandidates86_18 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_18 : scan edge (search edge 3) 4 [461] 461 innerTail86_18 best = best := by
  rw [innerTail86_18, scan, if_neg (by decide), innerCandidates86_18_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_19
    (if 3 ≤ innerCandidates86_18.length then search edge 3 [461, 5783] innerCandidates86_18 6244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_19

def innerTail86_17 : List Nat := 4937 :: innerTail86_18
def innerCandidates86_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates86_17_checked : candidates edge 4 461 4937 innerTail86_18 best.bound = innerCandidates86_17 := by
  unfold candidates
  have row : edge 4937 = SearchData.row4937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_17 : scan edge (search edge 3) 4 [461] 461 innerTail86_17 best = best := by
  rw [innerTail86_17, scan, if_neg (by decide), innerCandidates86_17_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_18
    (if 3 ≤ innerCandidates86_17.length then search edge 3 [461, 4937] innerCandidates86_17 5398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_18

def innerTail86_16 : List Nat := 4637 :: innerTail86_17
def innerCandidates86_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates86_16_checked : candidates edge 4 461 4637 innerTail86_17 best.bound = innerCandidates86_16 := by
  unfold candidates
  have row : edge 4637 = SearchData.row4637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_16 : scan edge (search edge 3) 4 [461] 461 innerTail86_16 best = best := by
  rw [innerTail86_16, scan, if_neg (by decide), innerCandidates86_16_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_17
    (if 3 ≤ innerCandidates86_16.length then search edge 3 [461, 4637] innerCandidates86_16 5098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_17

def innerTail86_15 : List Nat := 4271 :: innerTail86_16
def innerCandidates86_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates86_15_checked : candidates edge 4 461 4271 innerTail86_16 best.bound = innerCandidates86_15 := by
  unfold candidates
  have row : edge 4271 = SearchData.row4271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_15 : scan edge (search edge 3) 4 [461] 461 innerTail86_15 best = best := by
  rw [innerTail86_15, scan, if_neg (by decide), innerCandidates86_15_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_16
    (if 3 ≤ innerCandidates86_15.length then search edge 3 [461, 4271] innerCandidates86_15 4732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_16

def innerTail86_14 : List Nat := 4217 :: innerTail86_15
def innerCandidates86_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates86_14_checked : candidates edge 4 461 4217 innerTail86_15 best.bound = innerCandidates86_14 := by
  unfold candidates
  have row : edge 4217 = SearchData.row4217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_14 : scan edge (search edge 3) 4 [461] 461 innerTail86_14 best = best := by
  rw [innerTail86_14, scan, if_neg (by decide), innerCandidates86_14_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_15
    (if 3 ≤ innerCandidates86_14.length then search edge 3 [461, 4217] innerCandidates86_14 4678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_15

def innerTail86_13 : List Nat := 3881 :: innerTail86_14
def innerCandidates86_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates86_13_checked : candidates edge 4 461 3881 innerTail86_14 best.bound = innerCandidates86_13 := by
  unfold candidates
  have row : edge 3881 = SearchData.row3881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_13 : scan edge (search edge 3) 4 [461] 461 innerTail86_13 best = best := by
  rw [innerTail86_13, scan, if_neg (by decide), innerCandidates86_13_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_14
    (if 3 ≤ innerCandidates86_13.length then search edge 3 [461, 3881] innerCandidates86_13 4342 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_14

def innerTail86_12 : List Nat := 3449 :: innerTail86_13
def innerCandidates86_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates86_12_checked : candidates edge 4 461 3449 innerTail86_13 best.bound = innerCandidates86_12 := by
  unfold candidates
  have row : edge 3449 = SearchData.row3449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_12 : scan edge (search edge 3) 4 [461] 461 innerTail86_12 best = best := by
  rw [innerTail86_12, scan, if_neg (by decide), innerCandidates86_12_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_13
    (if 3 ≤ innerCandidates86_12.length then search edge 3 [461, 3449] innerCandidates86_12 3910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_13

def innerTail86_11 : List Nat := 2549 :: innerTail86_12
def innerCandidates86_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates86_11_checked : candidates edge 4 461 2549 innerTail86_12 best.bound = innerCandidates86_11 := by
  unfold candidates
  have row : edge 2549 = PrimePairCache.smallRow2549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_11 : scan edge (search edge 3) 4 [461] 461 innerTail86_11 best = best := by
  rw [innerTail86_11, scan, if_neg (by decide), innerCandidates86_11_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_12
    (if 3 ≤ innerCandidates86_11.length then search edge 3 [461, 2549] innerCandidates86_11 3010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_12

def innerTail86_10 : List Nat := 2339 :: innerTail86_11
def innerCandidates86_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates86_10_checked : candidates edge 4 461 2339 innerTail86_11 best.bound = innerCandidates86_10 := by
  unfold candidates
  have row : edge 2339 = PrimePairCache.smallRow2339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_10 : scan edge (search edge 3) 4 [461] 461 innerTail86_10 best = best := by
  rw [innerTail86_10, scan, if_neg (by decide), innerCandidates86_10_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_11
    (if 3 ≤ innerCandidates86_10.length then search edge 3 [461, 2339] innerCandidates86_10 2800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_11

def innerTail86_9 : List Nat := 1697 :: innerTail86_10
def innerCandidates86_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates86_9_checked : candidates edge 4 461 1697 innerTail86_10 best.bound = innerCandidates86_9 := by
  unfold candidates
  have row : edge 1697 = PrimePairCache.smallRow1697.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_9 : scan edge (search edge 3) 4 [461] 461 innerTail86_9 best = best := by
  rw [innerTail86_9, scan, if_neg (by decide), innerCandidates86_9_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_10
    (if 3 ≤ innerCandidates86_9.length then search edge 3 [461, 1697] innerCandidates86_9 2158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_10

def innerTail86_8 : List Nat := 1571 :: innerTail86_9
def innerCandidates86_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates86_8_checked : candidates edge 4 461 1571 innerTail86_9 best.bound = innerCandidates86_8 := by
  unfold candidates
  have row : edge 1571 = PrimePairCache.smallRow1571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_8 : scan edge (search edge 3) 4 [461] 461 innerTail86_8 best = best := by
  rw [innerTail86_8, scan, if_neg (by decide), innerCandidates86_8_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_9
    (if 3 ≤ innerCandidates86_8.length then search edge 3 [461, 1571] innerCandidates86_8 2032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_9

def innerTail86_7 : List Nat := 1487 :: innerTail86_8
def innerCandidates86_7 : List Nat := [6473]
set_option maxRecDepth 4096 in
theorem innerCandidates86_7_checked : candidates edge 4 461 1487 innerTail86_8 best.bound = innerCandidates86_7 := by
  unfold candidates
  have row : edge 1487 = PrimePairCache.smallRow1487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_7 : scan edge (search edge 3) 4 [461] 461 innerTail86_7 best = best := by
  rw [innerTail86_7, scan, if_neg (by decide), innerCandidates86_7_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_8
    (if 3 ≤ innerCandidates86_7.length then search edge 3 [461, 1487] innerCandidates86_7 1948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_8

def innerTail86_6 : List Nat := 1307 :: innerTail86_7
def innerCandidates86_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates86_6_checked : candidates edge 4 461 1307 innerTail86_7 best.bound = innerCandidates86_6 := by
  unfold candidates
  have row : edge 1307 = PrimePairCache.smallRow1307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_6 : scan edge (search edge 3) 4 [461] 461 innerTail86_6 best = best := by
  rw [innerTail86_6, scan, if_neg (by decide), innerCandidates86_6_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_7
    (if 3 ≤ innerCandidates86_6.length then search edge 3 [461, 1307] innerCandidates86_6 1768 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_7

def innerTail86_5 : List Nat := 1181 :: innerTail86_6
def innerCandidates86_5 : List Nat := [6977]
set_option maxRecDepth 4096 in
theorem innerCandidates86_5_checked : candidates edge 4 461 1181 innerTail86_6 best.bound = innerCandidates86_5 := by
  unfold candidates
  have row : edge 1181 = PrimePairCache.smallRow1181.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_5 : scan edge (search edge 3) 4 [461] 461 innerTail86_5 best = best := by
  rw [innerTail86_5, scan, if_neg (by decide), innerCandidates86_5_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_6
    (if 3 ≤ innerCandidates86_5.length then search edge 3 [461, 1181] innerCandidates86_5 1642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_6

def innerTail86_4 : List Nat := 983 :: innerTail86_5
def innerCandidates86_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates86_4_checked : candidates edge 4 461 983 innerTail86_5 best.bound = innerCandidates86_4 := by
  unfold candidates
  have row : edge 983 = PrimePairCache.smallRow983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_4 : scan edge (search edge 3) 4 [461] 461 innerTail86_4 best = best := by
  rw [innerTail86_4, scan, if_neg (by decide), innerCandidates86_4_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_5
    (if 3 ≤ innerCandidates86_4.length then search edge 3 [461, 983] innerCandidates86_4 1444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_5

def innerTail86_3 : List Nat := 677 :: innerTail86_4
def innerCandidates86_3 : List Nat := [7451]
set_option maxRecDepth 4096 in
theorem innerCandidates86_3_checked : candidates edge 4 461 677 innerTail86_4 best.bound = innerCandidates86_3 := by
  unfold candidates
  have row : edge 677 = PrimePairCache.smallRow677.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_3 : scan edge (search edge 3) 4 [461] 461 innerTail86_3 best = best := by
  rw [innerTail86_3, scan, if_neg (by decide), innerCandidates86_3_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_4
    (if 3 ≤ innerCandidates86_3.length then search edge 3 [461, 677] innerCandidates86_3 1138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_4

def innerTail86_2 : List Nat := 653 :: innerTail86_3
def innerCandidates86_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates86_2_checked : candidates edge 4 461 653 innerTail86_3 best.bound = innerCandidates86_2 := by
  unfold candidates
  have row : edge 653 = PrimePairCache.smallRow653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_2 : scan edge (search edge 3) 4 [461] 461 innerTail86_2 best = best := by
  rw [innerTail86_2, scan, if_neg (by decide), innerCandidates86_2_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_3
    (if 3 ≤ innerCandidates86_2.length then search edge 3 [461, 653] innerCandidates86_2 1114 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_3

def innerTail86_1 : List Nat := 569 :: innerTail86_2
def innerCandidates86_1 : List Nat := [1487]
set_option maxRecDepth 4096 in
theorem innerCandidates86_1_checked : candidates edge 4 461 569 innerTail86_2 best.bound = innerCandidates86_1 := by
  unfold candidates
  have row : edge 569 = PrimePairCache.smallRow569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_1 : scan edge (search edge 3) 4 [461] 461 innerTail86_1 best = best := by
  rw [innerTail86_1, scan, if_neg (by decide), innerCandidates86_1_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_2
    (if 3 ≤ innerCandidates86_1.length then search edge 3 [461, 569] innerCandidates86_1 1030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_2

def innerTail86_0 : List Nat := 479 :: innerTail86_1
def innerCandidates86_0 : List Nat := [569, 1307]
set_option maxRecDepth 4096 in
theorem innerCandidates86_0_checked : candidates edge 4 461 479 innerTail86_1 best.bound = innerCandidates86_0 := by
  unfold candidates
  have row : edge 479 = PrimePairCache.smallRow479.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan86_0 : scan edge (search edge 3) 4 [461] 461 innerTail86_0 best = best := by
  rw [innerTail86_0, scan, if_neg (by decide), innerCandidates86_0_checked]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_1
    (if 3 ≤ innerCandidates86_0.length then search edge 3 [461, 479] innerCandidates86_0 940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan86_1

theorem rootBranch86 : search edge 4 [461] rootCandidates86 461 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [461] 461 innerTail86_0 best = best
  exact innerScan86_0

def rootCandidates85 : List Nat := [643, 673, 757, 829, 1033, 1597, 2131, 2383, 2551, 2707, 2917, 3571, 3631, 3931, 4441, 4987, 5107, 5281, 5503, 5647, 6997, 7039, 7297, 7411, 7669]
set_option maxRecDepth 4096 in
theorem rootCandidates85_checked : candidates edge 5 0 457 tail86 best.bound = rootCandidates85 := by
  unfold candidates
  have row : edge 457 = PrimePairCache.smallRow457.contains := by funext q; rfl
  rw [row]
  decide

def innerTail85_25 : List Nat := []
theorem innerScan85_25 : scan edge (search edge 3) 4 [457] 457 innerTail85_25 best = best := rfl

def innerTail85_24 : List Nat := 7669 :: innerTail85_25
def innerCandidates85_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates85_24_checked : candidates edge 4 457 7669 innerTail85_25 best.bound = innerCandidates85_24 := by
  unfold candidates
  have row : edge 7669 = SearchData.row7669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_24 : scan edge (search edge 3) 4 [457] 457 innerTail85_24 best = best := by
  rw [innerTail85_24, scan, if_neg (by decide), innerCandidates85_24_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_25
    (if 3 ≤ innerCandidates85_24.length then search edge 3 [457, 7669] innerCandidates85_24 8126 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_25

def innerTail85_23 : List Nat := 7411 :: innerTail85_24
def innerCandidates85_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates85_23_checked : candidates edge 4 457 7411 innerTail85_24 best.bound = innerCandidates85_23 := by
  unfold candidates
  have row : edge 7411 = SearchData.row7411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_23 : scan edge (search edge 3) 4 [457] 457 innerTail85_23 best = best := by
  rw [innerTail85_23, scan, if_neg (by decide), innerCandidates85_23_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_24
    (if 3 ≤ innerCandidates85_23.length then search edge 3 [457, 7411] innerCandidates85_23 7868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_24

def innerTail85_22 : List Nat := 7297 :: innerTail85_23
def innerCandidates85_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates85_22_checked : candidates edge 4 457 7297 innerTail85_23 best.bound = innerCandidates85_22 := by
  unfold candidates
  have row : edge 7297 = SearchData.row7297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_22 : scan edge (search edge 3) 4 [457] 457 innerTail85_22 best = best := by
  rw [innerTail85_22, scan, if_neg (by decide), innerCandidates85_22_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_23
    (if 3 ≤ innerCandidates85_22.length then search edge 3 [457, 7297] innerCandidates85_22 7754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_23

def innerTail85_21 : List Nat := 7039 :: innerTail85_22
def innerCandidates85_21 : List Nat := [7669]
set_option maxRecDepth 4096 in
theorem innerCandidates85_21_checked : candidates edge 4 457 7039 innerTail85_22 best.bound = innerCandidates85_21 := by
  unfold candidates
  have row : edge 7039 = SearchData.row7039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_21 : scan edge (search edge 3) 4 [457] 457 innerTail85_21 best = best := by
  rw [innerTail85_21, scan, if_neg (by decide), innerCandidates85_21_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_22
    (if 3 ≤ innerCandidates85_21.length then search edge 3 [457, 7039] innerCandidates85_21 7496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_22

def innerTail85_20 : List Nat := 6997 :: innerTail85_21
def innerCandidates85_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates85_20_checked : candidates edge 4 457 6997 innerTail85_21 best.bound = innerCandidates85_20 := by
  unfold candidates
  have row : edge 6997 = SearchData.row6997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_20 : scan edge (search edge 3) 4 [457] 457 innerTail85_20 best = best := by
  rw [innerTail85_20, scan, if_neg (by decide), innerCandidates85_20_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_21
    (if 3 ≤ innerCandidates85_20.length then search edge 3 [457, 6997] innerCandidates85_20 7454 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_21

def innerTail85_19 : List Nat := 5647 :: innerTail85_20
def innerCandidates85_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates85_19_checked : candidates edge 4 457 5647 innerTail85_20 best.bound = innerCandidates85_19 := by
  unfold candidates
  have row : edge 5647 = SearchData.row5647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_19 : scan edge (search edge 3) 4 [457] 457 innerTail85_19 best = best := by
  rw [innerTail85_19, scan, if_neg (by decide), innerCandidates85_19_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_20
    (if 3 ≤ innerCandidates85_19.length then search edge 3 [457, 5647] innerCandidates85_19 6104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_20

def innerTail85_18 : List Nat := 5503 :: innerTail85_19
def innerCandidates85_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates85_18_checked : candidates edge 4 457 5503 innerTail85_19 best.bound = innerCandidates85_18 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_18 : scan edge (search edge 3) 4 [457] 457 innerTail85_18 best = best := by
  rw [innerTail85_18, scan, if_neg (by decide), innerCandidates85_18_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_19
    (if 3 ≤ innerCandidates85_18.length then search edge 3 [457, 5503] innerCandidates85_18 5960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_19

def innerTail85_17 : List Nat := 5281 :: innerTail85_18
def innerCandidates85_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates85_17_checked : candidates edge 4 457 5281 innerTail85_18 best.bound = innerCandidates85_17 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_17 : scan edge (search edge 3) 4 [457] 457 innerTail85_17 best = best := by
  rw [innerTail85_17, scan, if_neg (by decide), innerCandidates85_17_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_18
    (if 3 ≤ innerCandidates85_17.length then search edge 3 [457, 5281] innerCandidates85_17 5738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_18

def innerTail85_16 : List Nat := 5107 :: innerTail85_17
def innerCandidates85_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates85_16_checked : candidates edge 4 457 5107 innerTail85_17 best.bound = innerCandidates85_16 := by
  unfold candidates
  have row : edge 5107 = SearchData.row5107.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_16 : scan edge (search edge 3) 4 [457] 457 innerTail85_16 best = best := by
  rw [innerTail85_16, scan, if_neg (by decide), innerCandidates85_16_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_17
    (if 3 ≤ innerCandidates85_16.length then search edge 3 [457, 5107] innerCandidates85_16 5564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_17

def innerTail85_15 : List Nat := 4987 :: innerTail85_16
def innerCandidates85_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates85_15_checked : candidates edge 4 457 4987 innerTail85_16 best.bound = innerCandidates85_15 := by
  unfold candidates
  have row : edge 4987 = SearchData.row4987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_15 : scan edge (search edge 3) 4 [457] 457 innerTail85_15 best = best := by
  rw [innerTail85_15, scan, if_neg (by decide), innerCandidates85_15_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_16
    (if 3 ≤ innerCandidates85_15.length then search edge 3 [457, 4987] innerCandidates85_15 5444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_16

def innerTail85_14 : List Nat := 4441 :: innerTail85_15
def innerCandidates85_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates85_14_checked : candidates edge 4 457 4441 innerTail85_15 best.bound = innerCandidates85_14 := by
  unfold candidates
  have row : edge 4441 = SearchData.row4441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_14 : scan edge (search edge 3) 4 [457] 457 innerTail85_14 best = best := by
  rw [innerTail85_14, scan, if_neg (by decide), innerCandidates85_14_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_15
    (if 3 ≤ innerCandidates85_14.length then search edge 3 [457, 4441] innerCandidates85_14 4898 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_15

def innerTail85_13 : List Nat := 3931 :: innerTail85_14
def innerCandidates85_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates85_13_checked : candidates edge 4 457 3931 innerTail85_14 best.bound = innerCandidates85_13 := by
  unfold candidates
  have row : edge 3931 = SearchData.row3931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_13 : scan edge (search edge 3) 4 [457] 457 innerTail85_13 best = best := by
  rw [innerTail85_13, scan, if_neg (by decide), innerCandidates85_13_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_14
    (if 3 ≤ innerCandidates85_13.length then search edge 3 [457, 3931] innerCandidates85_13 4388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_14

def innerTail85_12 : List Nat := 3631 :: innerTail85_13
def innerCandidates85_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates85_12_checked : candidates edge 4 457 3631 innerTail85_13 best.bound = innerCandidates85_12 := by
  unfold candidates
  have row : edge 3631 = SearchData.row3631.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_12 : scan edge (search edge 3) 4 [457] 457 innerTail85_12 best = best := by
  rw [innerTail85_12, scan, if_neg (by decide), innerCandidates85_12_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_13
    (if 3 ≤ innerCandidates85_12.length then search edge 3 [457, 3631] innerCandidates85_12 4088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_13

def innerTail85_11 : List Nat := 3571 :: innerTail85_12
def innerCandidates85_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates85_11_checked : candidates edge 4 457 3571 innerTail85_12 best.bound = innerCandidates85_11 := by
  unfold candidates
  have row : edge 3571 = SearchData.row3571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_11 : scan edge (search edge 3) 4 [457] 457 innerTail85_11 best = best := by
  rw [innerTail85_11, scan, if_neg (by decide), innerCandidates85_11_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_12
    (if 3 ≤ innerCandidates85_11.length then search edge 3 [457, 3571] innerCandidates85_11 4028 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_12

def innerTail85_10 : List Nat := 2917 :: innerTail85_11
def innerCandidates85_10 : List Nat := [3631]
set_option maxRecDepth 4096 in
theorem innerCandidates85_10_checked : candidates edge 4 457 2917 innerTail85_11 best.bound = innerCandidates85_10 := by
  unfold candidates
  have row : edge 2917 = SearchData.row2917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_10 : scan edge (search edge 3) 4 [457] 457 innerTail85_10 best = best := by
  rw [innerTail85_10, scan, if_neg (by decide), innerCandidates85_10_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_11
    (if 3 ≤ innerCandidates85_10.length then search edge 3 [457, 2917] innerCandidates85_10 3374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_11

def innerTail85_9 : List Nat := 2707 :: innerTail85_10
def innerCandidates85_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates85_9_checked : candidates edge 4 457 2707 innerTail85_10 best.bound = innerCandidates85_9 := by
  unfold candidates
  have row : edge 2707 = SearchData.row2707.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_9 : scan edge (search edge 3) 4 [457] 457 innerTail85_9 best = best := by
  rw [innerTail85_9, scan, if_neg (by decide), innerCandidates85_9_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_10
    (if 3 ≤ innerCandidates85_9.length then search edge 3 [457, 2707] innerCandidates85_9 3164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_10

def innerTail85_8 : List Nat := 2551 :: innerTail85_9
def innerCandidates85_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates85_8_checked : candidates edge 4 457 2551 innerTail85_9 best.bound = innerCandidates85_8 := by
  unfold candidates
  have row : edge 2551 = PrimePairCache.smallRow2551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_8 : scan edge (search edge 3) 4 [457] 457 innerTail85_8 best = best := by
  rw [innerTail85_8, scan, if_neg (by decide), innerCandidates85_8_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_9
    (if 3 ≤ innerCandidates85_8.length then search edge 3 [457, 2551] innerCandidates85_8 3008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_9

def innerTail85_7 : List Nat := 2383 :: innerTail85_8
def innerCandidates85_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates85_7_checked : candidates edge 4 457 2383 innerTail85_8 best.bound = innerCandidates85_7 := by
  unfold candidates
  have row : edge 2383 = PrimePairCache.smallRow2383.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_7 : scan edge (search edge 3) 4 [457] 457 innerTail85_7 best = best := by
  rw [innerTail85_7, scan, if_neg (by decide), innerCandidates85_7_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_8
    (if 3 ≤ innerCandidates85_7.length then search edge 3 [457, 2383] innerCandidates85_7 2840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_8

def innerTail85_6 : List Nat := 2131 :: innerTail85_7
def innerCandidates85_6 : List Nat := [7669]
set_option maxRecDepth 4096 in
theorem innerCandidates85_6_checked : candidates edge 4 457 2131 innerTail85_7 best.bound = innerCandidates85_6 := by
  unfold candidates
  have row : edge 2131 = PrimePairCache.smallRow2131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_6 : scan edge (search edge 3) 4 [457] 457 innerTail85_6 best = best := by
  rw [innerTail85_6, scan, if_neg (by decide), innerCandidates85_6_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_7
    (if 3 ≤ innerCandidates85_6.length then search edge 3 [457, 2131] innerCandidates85_6 2588 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_7

def innerTail85_5 : List Nat := 1597 :: innerTail85_6
def innerCandidates85_5 : List Nat := [5503]
set_option maxRecDepth 4096 in
theorem innerCandidates85_5_checked : candidates edge 4 457 1597 innerTail85_6 best.bound = innerCandidates85_5 := by
  unfold candidates
  have row : edge 1597 = PrimePairCache.smallRow1597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_5 : scan edge (search edge 3) 4 [457] 457 innerTail85_5 best = best := by
  rw [innerTail85_5, scan, if_neg (by decide), innerCandidates85_5_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_6
    (if 3 ≤ innerCandidates85_5.length then search edge 3 [457, 1597] innerCandidates85_5 2054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_6

def innerTail85_4 : List Nat := 1033 :: innerTail85_5
def innerCandidates85_4 : List Nat := [2131]
set_option maxRecDepth 4096 in
theorem innerCandidates85_4_checked : candidates edge 4 457 1033 innerTail85_5 best.bound = innerCandidates85_4 := by
  unfold candidates
  have row : edge 1033 = PrimePairCache.smallRow1033.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_4 : scan edge (search edge 3) 4 [457] 457 innerTail85_4 best = best := by
  rw [innerTail85_4, scan, if_neg (by decide), innerCandidates85_4_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_5
    (if 3 ≤ innerCandidates85_4.length then search edge 3 [457, 1033] innerCandidates85_4 1490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_5

def innerTail85_3 : List Nat := 829 :: innerTail85_4
def innerCandidates85_3 : List Nat := [1033, 5107]
set_option maxRecDepth 4096 in
theorem innerCandidates85_3_checked : candidates edge 4 457 829 innerTail85_4 best.bound = innerCandidates85_3 := by
  unfold candidates
  have row : edge 829 = PrimePairCache.smallRow829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_3 : scan edge (search edge 3) 4 [457] 457 innerTail85_3 best = best := by
  rw [innerTail85_3, scan, if_neg (by decide), innerCandidates85_3_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_4
    (if 3 ≤ innerCandidates85_3.length then search edge 3 [457, 829] innerCandidates85_3 1286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_4

def innerTail85_2 : List Nat := 757 :: innerTail85_3
def innerCandidates85_2 : List Nat := [829, 3931]
set_option maxRecDepth 4096 in
theorem innerCandidates85_2_checked : candidates edge 4 457 757 innerTail85_3 best.bound = innerCandidates85_2 := by
  unfold candidates
  have row : edge 757 = PrimePairCache.smallRow757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_2 : scan edge (search edge 3) 4 [457] 457 innerTail85_2 best = best := by
  rw [innerTail85_2, scan, if_neg (by decide), innerCandidates85_2_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_3
    (if 3 ≤ innerCandidates85_2.length then search edge 3 [457, 757] innerCandidates85_2 1214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_3

def innerTail85_1 : List Nat := 673 :: innerTail85_2
def innerCandidates85_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates85_1_checked : candidates edge 4 457 673 innerTail85_2 best.bound = innerCandidates85_1 := by
  unfold candidates
  have row : edge 673 = PrimePairCache.smallRow673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_1 : scan edge (search edge 3) 4 [457] 457 innerTail85_1 best = best := by
  rw [innerTail85_1, scan, if_neg (by decide), innerCandidates85_1_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_2
    (if 3 ≤ innerCandidates85_1.length then search edge 3 [457, 673] innerCandidates85_1 1130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_2

def innerTail85_0 : List Nat := 643 :: innerTail85_1
def innerCandidates85_0 : List Nat := [2383, 2917]
set_option maxRecDepth 4096 in
theorem innerCandidates85_0_checked : candidates edge 4 457 643 innerTail85_1 best.bound = innerCandidates85_0 := by
  unfold candidates
  have row : edge 643 = PrimePairCache.smallRow643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan85_0 : scan edge (search edge 3) 4 [457] 457 innerTail85_0 best = best := by
  rw [innerTail85_0, scan, if_neg (by decide), innerCandidates85_0_checked]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_1
    (if 3 ≤ innerCandidates85_0.length then search edge 3 [457, 643] innerCandidates85_0 1100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan85_1

theorem rootBranch85 : search edge 4 [457] rootCandidates85 457 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [457] 457 innerTail85_0 best = best
  exact innerScan85_0

end ProjectEulerSolutions.PrimePairUpper.Search
