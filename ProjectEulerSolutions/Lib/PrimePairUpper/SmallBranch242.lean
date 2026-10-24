import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates27 : List Nat := [131, 149, 167, 227, 233, 383, 647, 761, 983, 1217, 1259, 1973, 2309, 2861, 3191, 3359, 3533, 3659, 3947, 4421, 4517, 5333, 5711, 6389, 6659, 6719, 6869, 7457, 8237]
set_option maxRecDepth 4096 in
theorem rootCandidates27_checked : candidates edge 5 0 113 tail28 best.bound = rootCandidates27 := by
  unfold candidates
  have row : edge 113 = PrimePairCache.smallRow113.contains := by funext q; rfl
  rw [row]
  decide

def innerTail27_29 : List Nat := []
theorem innerScan27_29 : scan edge (search edge 3) 4 [113] 113 innerTail27_29 best = best := rfl

def innerTail27_28 : List Nat := 8237 :: innerTail27_29
def innerCandidates27_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates27_28_checked : candidates edge 4 113 8237 innerTail27_29 best.bound = innerCandidates27_28 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_28 : scan edge (search edge 3) 4 [113] 113 innerTail27_28 best = best := by
  rw [innerTail27_28, scan, if_neg (by decide), innerCandidates27_28_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_29
    (if 3 ≤ innerCandidates27_28.length then search edge 3 [113, 8237] innerCandidates27_28 8350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_29

def innerTail27_27 : List Nat := 7457 :: innerTail27_28
def innerCandidates27_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates27_27_checked : candidates edge 4 113 7457 innerTail27_28 best.bound = innerCandidates27_27 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_27 : scan edge (search edge 3) 4 [113] 113 innerTail27_27 best = best := by
  rw [innerTail27_27, scan, if_neg (by decide), innerCandidates27_27_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_28
    (if 3 ≤ innerCandidates27_27.length then search edge 3 [113, 7457] innerCandidates27_27 7570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_28

def innerTail27_26 : List Nat := 6869 :: innerTail27_27
def innerCandidates27_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates27_26_checked : candidates edge 4 113 6869 innerTail27_27 best.bound = innerCandidates27_26 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_26 : scan edge (search edge 3) 4 [113] 113 innerTail27_26 best = best := by
  rw [innerTail27_26, scan, if_neg (by decide), innerCandidates27_26_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_27
    (if 3 ≤ innerCandidates27_26.length then search edge 3 [113, 6869] innerCandidates27_26 6982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_27

def innerTail27_25 : List Nat := 6719 :: innerTail27_26
def innerCandidates27_25 : List Nat := [8237]
set_option maxRecDepth 4096 in
theorem innerCandidates27_25_checked : candidates edge 4 113 6719 innerTail27_26 best.bound = innerCandidates27_25 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_25 : scan edge (search edge 3) 4 [113] 113 innerTail27_25 best = best := by
  rw [innerTail27_25, scan, if_neg (by decide), innerCandidates27_25_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_26
    (if 3 ≤ innerCandidates27_25.length then search edge 3 [113, 6719] innerCandidates27_25 6832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_26

def innerTail27_24 : List Nat := 6659 :: innerTail27_25
def innerCandidates27_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates27_24_checked : candidates edge 4 113 6659 innerTail27_25 best.bound = innerCandidates27_24 := by
  unfold candidates
  have row : edge 6659 = SearchData.row6659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_24 : scan edge (search edge 3) 4 [113] 113 innerTail27_24 best = best := by
  rw [innerTail27_24, scan, if_neg (by decide), innerCandidates27_24_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_25
    (if 3 ≤ innerCandidates27_24.length then search edge 3 [113, 6659] innerCandidates27_24 6772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_25

def innerTail27_23 : List Nat := 6389 :: innerTail27_24
def innerCandidates27_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates27_23_checked : candidates edge 4 113 6389 innerTail27_24 best.bound = innerCandidates27_23 := by
  unfold candidates
  have row : edge 6389 = SearchData.row6389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_23 : scan edge (search edge 3) 4 [113] 113 innerTail27_23 best = best := by
  rw [innerTail27_23, scan, if_neg (by decide), innerCandidates27_23_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_24
    (if 3 ≤ innerCandidates27_23.length then search edge 3 [113, 6389] innerCandidates27_23 6502 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_24

def innerTail27_22 : List Nat := 5711 :: innerTail27_23
def innerCandidates27_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates27_22_checked : candidates edge 4 113 5711 innerTail27_23 best.bound = innerCandidates27_22 := by
  unfold candidates
  have row : edge 5711 = SearchData.row5711.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_22 : scan edge (search edge 3) 4 [113] 113 innerTail27_22 best = best := by
  rw [innerTail27_22, scan, if_neg (by decide), innerCandidates27_22_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_23
    (if 3 ≤ innerCandidates27_22.length then search edge 3 [113, 5711] innerCandidates27_22 5824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_23

def innerTail27_21 : List Nat := 5333 :: innerTail27_22
def innerCandidates27_21 : List Nat := [6719]
set_option maxRecDepth 4096 in
theorem innerCandidates27_21_checked : candidates edge 4 113 5333 innerTail27_22 best.bound = innerCandidates27_21 := by
  unfold candidates
  have row : edge 5333 = SearchData.row5333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_21 : scan edge (search edge 3) 4 [113] 113 innerTail27_21 best = best := by
  rw [innerTail27_21, scan, if_neg (by decide), innerCandidates27_21_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_22
    (if 3 ≤ innerCandidates27_21.length then search edge 3 [113, 5333] innerCandidates27_21 5446 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_22

def innerTail27_20 : List Nat := 4517 :: innerTail27_21
def innerCandidates27_20 : List Nat := [6869]
set_option maxRecDepth 4096 in
theorem innerCandidates27_20_checked : candidates edge 4 113 4517 innerTail27_21 best.bound = innerCandidates27_20 := by
  unfold candidates
  have row : edge 4517 = SearchData.row4517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_20 : scan edge (search edge 3) 4 [113] 113 innerTail27_20 best = best := by
  rw [innerTail27_20, scan, if_neg (by decide), innerCandidates27_20_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_21
    (if 3 ≤ innerCandidates27_20.length then search edge 3 [113, 4517] innerCandidates27_20 4630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_21

def innerTail27_19 : List Nat := 4421 :: innerTail27_20
def innerCandidates27_19 : List Nat := [6869]
set_option maxRecDepth 4096 in
theorem innerCandidates27_19_checked : candidates edge 4 113 4421 innerTail27_20 best.bound = innerCandidates27_19 := by
  unfold candidates
  have row : edge 4421 = SearchData.row4421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_19 : scan edge (search edge 3) 4 [113] 113 innerTail27_19 best = best := by
  rw [innerTail27_19, scan, if_neg (by decide), innerCandidates27_19_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_20
    (if 3 ≤ innerCandidates27_19.length then search edge 3 [113, 4421] innerCandidates27_19 4534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_20

def innerTail27_18 : List Nat := 3947 :: innerTail27_19
def innerCandidates27_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates27_18_checked : candidates edge 4 113 3947 innerTail27_19 best.bound = innerCandidates27_18 := by
  unfold candidates
  have row : edge 3947 = SearchData.row3947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_18 : scan edge (search edge 3) 4 [113] 113 innerTail27_18 best = best := by
  rw [innerTail27_18, scan, if_neg (by decide), innerCandidates27_18_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_19
    (if 3 ≤ innerCandidates27_18.length then search edge 3 [113, 3947] innerCandidates27_18 4060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_19

def innerTail27_17 : List Nat := 3659 :: innerTail27_18
def innerCandidates27_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates27_17_checked : candidates edge 4 113 3659 innerTail27_18 best.bound = innerCandidates27_17 := by
  unfold candidates
  have row : edge 3659 = SearchData.row3659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_17 : scan edge (search edge 3) 4 [113] 113 innerTail27_17 best = best := by
  rw [innerTail27_17, scan, if_neg (by decide), innerCandidates27_17_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_18
    (if 3 ≤ innerCandidates27_17.length then search edge 3 [113, 3659] innerCandidates27_17 3772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_18

def innerTail27_16 : List Nat := 3533 :: innerTail27_17
def innerCandidates27_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates27_16_checked : candidates edge 4 113 3533 innerTail27_17 best.bound = innerCandidates27_16 := by
  unfold candidates
  have row : edge 3533 = SearchData.row3533.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_16 : scan edge (search edge 3) 4 [113] 113 innerTail27_16 best = best := by
  rw [innerTail27_16, scan, if_neg (by decide), innerCandidates27_16_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_17
    (if 3 ≤ innerCandidates27_16.length then search edge 3 [113, 3533] innerCandidates27_16 3646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_17

def innerTail27_15 : List Nat := 3359 :: innerTail27_16
def innerCandidates27_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates27_15_checked : candidates edge 4 113 3359 innerTail27_16 best.bound = innerCandidates27_15 := by
  unfold candidates
  have row : edge 3359 = SearchData.row3359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_15 : scan edge (search edge 3) 4 [113] 113 innerTail27_15 best = best := by
  rw [innerTail27_15, scan, if_neg (by decide), innerCandidates27_15_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_16
    (if 3 ≤ innerCandidates27_15.length then search edge 3 [113, 3359] innerCandidates27_15 3472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_16

def innerTail27_14 : List Nat := 3191 :: innerTail27_15
def innerCandidates27_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates27_14_checked : candidates edge 4 113 3191 innerTail27_15 best.bound = innerCandidates27_14 := by
  unfold candidates
  have row : edge 3191 = SearchData.row3191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_14 : scan edge (search edge 3) 4 [113] 113 innerTail27_14 best = best := by
  rw [innerTail27_14, scan, if_neg (by decide), innerCandidates27_14_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_15
    (if 3 ≤ innerCandidates27_14.length then search edge 3 [113, 3191] innerCandidates27_14 3304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_15

def innerTail27_13 : List Nat := 2861 :: innerTail27_14
def innerCandidates27_13 : List Nat := [6719]
set_option maxRecDepth 4096 in
theorem innerCandidates27_13_checked : candidates edge 4 113 2861 innerTail27_14 best.bound = innerCandidates27_13 := by
  unfold candidates
  have row : edge 2861 = SearchData.row2861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_13 : scan edge (search edge 3) 4 [113] 113 innerTail27_13 best = best := by
  rw [innerTail27_13, scan, if_neg (by decide), innerCandidates27_13_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_14
    (if 3 ≤ innerCandidates27_13.length then search edge 3 [113, 2861] innerCandidates27_13 2974 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_14

def innerTail27_12 : List Nat := 2309 :: innerTail27_13
def innerCandidates27_12 : List Nat := [8237]
set_option maxRecDepth 4096 in
theorem innerCandidates27_12_checked : candidates edge 4 113 2309 innerTail27_13 best.bound = innerCandidates27_12 := by
  unfold candidates
  have row : edge 2309 = PrimePairCache.smallRow2309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_12 : scan edge (search edge 3) 4 [113] 113 innerTail27_12 best = best := by
  rw [innerTail27_12, scan, if_neg (by decide), innerCandidates27_12_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_13
    (if 3 ≤ innerCandidates27_12.length then search edge 3 [113, 2309] innerCandidates27_12 2422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_13

def innerTail27_11 : List Nat := 1973 :: innerTail27_12
def innerCandidates27_11 : List Nat := [5711]
set_option maxRecDepth 4096 in
theorem innerCandidates27_11_checked : candidates edge 4 113 1973 innerTail27_12 best.bound = innerCandidates27_11 := by
  unfold candidates
  have row : edge 1973 = PrimePairCache.smallRow1973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_11 : scan edge (search edge 3) 4 [113] 113 innerTail27_11 best = best := by
  rw [innerTail27_11, scan, if_neg (by decide), innerCandidates27_11_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_12
    (if 3 ≤ innerCandidates27_11.length then search edge 3 [113, 1973] innerCandidates27_11 2086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_12

def innerTail27_10 : List Nat := 1259 :: innerTail27_11
def innerCandidates27_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates27_10_checked : candidates edge 4 113 1259 innerTail27_11 best.bound = innerCandidates27_10 := by
  unfold candidates
  have row : edge 1259 = PrimePairCache.smallRow1259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_10 : scan edge (search edge 3) 4 [113] 113 innerTail27_10 best = best := by
  rw [innerTail27_10, scan, if_neg (by decide), innerCandidates27_10_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_11
    (if 3 ≤ innerCandidates27_10.length then search edge 3 [113, 1259] innerCandidates27_10 1372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_11

def innerTail27_9 : List Nat := 1217 :: innerTail27_10
def innerCandidates27_9 : List Nat := [2309]
set_option maxRecDepth 4096 in
theorem innerCandidates27_9_checked : candidates edge 4 113 1217 innerTail27_10 best.bound = innerCandidates27_9 := by
  unfold candidates
  have row : edge 1217 = PrimePairCache.smallRow1217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_9 : scan edge (search edge 3) 4 [113] 113 innerTail27_9 best = best := by
  rw [innerTail27_9, scan, if_neg (by decide), innerCandidates27_9_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_10
    (if 3 ≤ innerCandidates27_9.length then search edge 3 [113, 1217] innerCandidates27_9 1330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_10

def innerTail27_8 : List Nat := 983 :: innerTail27_9
def innerCandidates27_8 : List Nat := [1259, 6389, 6719]
set_option maxRecDepth 4096 in
theorem innerCandidates27_8_checked : candidates edge 4 113 983 innerTail27_9 best.bound = innerCandidates27_8 := by
  unfold candidates
  have row : edge 983 = PrimePairCache.smallRow983.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild27_8 : search edge 3 [113, 983] innerCandidates27_8 1096 best = best := by decide

theorem innerScan27_8 : scan edge (search edge 3) 4 [113] 113 innerTail27_8 best = best := by
  rw [innerTail27_8, scan, if_neg (by decide), innerCandidates27_8_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_9
    (if 3 ≤ innerCandidates27_8.length then search edge 3 [113, 983] innerCandidates27_8 1096 best else best) = best
  rw [if_pos (by decide), innerChild27_8]
  exact innerScan27_9

def innerTail27_7 : List Nat := 761 :: innerTail27_8
def innerCandidates27_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates27_7_checked : candidates edge 4 113 761 innerTail27_8 best.bound = innerCandidates27_7 := by
  unfold candidates
  have row : edge 761 = PrimePairCache.smallRow761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_7 : scan edge (search edge 3) 4 [113] 113 innerTail27_7 best = best := by
  rw [innerTail27_7, scan, if_neg (by decide), innerCandidates27_7_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_8
    (if 3 ≤ innerCandidates27_7.length then search edge 3 [113, 761] innerCandidates27_7 874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_8

def innerTail27_6 : List Nat := 647 :: innerTail27_7
def innerCandidates27_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates27_6_checked : candidates edge 4 113 647 innerTail27_7 best.bound = innerCandidates27_6 := by
  unfold candidates
  have row : edge 647 = PrimePairCache.smallRow647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_6 : scan edge (search edge 3) 4 [113] 113 innerTail27_6 best = best := by
  rw [innerTail27_6, scan, if_neg (by decide), innerCandidates27_6_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_7
    (if 3 ≤ innerCandidates27_6.length then search edge 3 [113, 647] innerCandidates27_6 760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_7

def innerTail27_5 : List Nat := 383 :: innerTail27_6
def innerCandidates27_5 : List Nat := [3359]
set_option maxRecDepth 4096 in
theorem innerCandidates27_5_checked : candidates edge 4 113 383 innerTail27_6 best.bound = innerCandidates27_5 := by
  unfold candidates
  have row : edge 383 = PrimePairCache.smallRow383.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_5 : scan edge (search edge 3) 4 [113] 113 innerTail27_5 best = best := by
  rw [innerTail27_5, scan, if_neg (by decide), innerCandidates27_5_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_6
    (if 3 ≤ innerCandidates27_5.length then search edge 3 [113, 383] innerCandidates27_5 496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_6

def innerTail27_4 : List Nat := 233 :: innerTail27_5
def innerCandidates27_4 : List Nat := [983]
set_option maxRecDepth 4096 in
theorem innerCandidates27_4_checked : candidates edge 4 113 233 innerTail27_5 best.bound = innerCandidates27_4 := by
  unfold candidates
  have row : edge 233 = PrimePairCache.smallRow233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_4 : scan edge (search edge 3) 4 [113] 113 innerTail27_4 best = best := by
  rw [innerTail27_4, scan, if_neg (by decide), innerCandidates27_4_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_5
    (if 3 ≤ innerCandidates27_4.length then search edge 3 [113, 233] innerCandidates27_4 346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_5

def innerTail27_3 : List Nat := 227 :: innerTail27_4
def innerCandidates27_3 : List Nat := [4421, 5333]
set_option maxRecDepth 4096 in
theorem innerCandidates27_3_checked : candidates edge 4 113 227 innerTail27_4 best.bound = innerCandidates27_3 := by
  unfold candidates
  have row : edge 227 = PrimePairCache.smallRow227.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_3 : scan edge (search edge 3) 4 [113] 113 innerTail27_3 best = best := by
  rw [innerTail27_3, scan, if_neg (by decide), innerCandidates27_3_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_4
    (if 3 ≤ innerCandidates27_3.length then search edge 3 [113, 227] innerCandidates27_3 340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_4

def innerTail27_2 : List Nat := 167 :: innerTail27_3
def innerCandidates27_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates27_2_checked : candidates edge 4 113 167 innerTail27_3 best.bound = innerCandidates27_2 := by
  unfold candidates
  have row : edge 167 = PrimePairCache.smallRow167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_2 : scan edge (search edge 3) 4 [113] 113 innerTail27_2 best = best := by
  rw [innerTail27_2, scan, if_neg (by decide), innerCandidates27_2_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_3
    (if 3 ≤ innerCandidates27_2.length then search edge 3 [113, 167] innerCandidates27_2 280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_3

def innerTail27_1 : List Nat := 149 :: innerTail27_2
def innerCandidates27_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates27_1_checked : candidates edge 4 113 149 innerTail27_2 best.bound = innerCandidates27_1 := by
  unfold candidates
  have row : edge 149 = PrimePairCache.smallRow149.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_1 : scan edge (search edge 3) 4 [113] 113 innerTail27_1 best = best := by
  rw [innerTail27_1, scan, if_neg (by decide), innerCandidates27_1_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_2
    (if 3 ≤ innerCandidates27_1.length then search edge 3 [113, 149] innerCandidates27_1 262 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_2

def innerTail27_0 : List Nat := 131 :: innerTail27_1
def innerCandidates27_0 : List Nat := [1217]
set_option maxRecDepth 4096 in
theorem innerCandidates27_0_checked : candidates edge 4 113 131 innerTail27_1 best.bound = innerCandidates27_0 := by
  unfold candidates
  have row : edge 131 = PrimePairCache.smallRow131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan27_0 : scan edge (search edge 3) 4 [113] 113 innerTail27_0 best = best := by
  rw [innerTail27_0, scan, if_neg (by decide), innerCandidates27_0_checked]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_1
    (if 3 ≤ innerCandidates27_0.length then search edge 3 [113, 131] innerCandidates27_0 244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan27_1

theorem rootBranch27 : search edge 4 [113] rootCandidates27 113 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [113] 113 innerTail27_0 best = best
  exact innerScan27_0

def rootCandidates26 : List Nat := [139, 199, 313, 661, 673, 859, 883, 919, 1063, 1381, 1399, 1459, 1609, 1777, 2389, 2803, 3331, 3529, 3943, 4129, 4441, 4567, 4759, 4801, 4957, 5503, 5791, 6673, 7297, 7321, 7351, 7933, 8269]
set_option maxRecDepth 4096 in
theorem rootCandidates26_checked : candidates edge 5 0 109 tail27 best.bound = rootCandidates26 := by
  unfold candidates
  have row : edge 109 = PrimePairCache.smallRow109.contains := by funext q; rfl
  rw [row]
  decide

def innerTail26_33 : List Nat := []
theorem innerScan26_33 : scan edge (search edge 3) 4 [109] 109 innerTail26_33 best = best := rfl

def innerTail26_32 : List Nat := 8269 :: innerTail26_33
def innerCandidates26_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates26_32_checked : candidates edge 4 109 8269 innerTail26_33 best.bound = innerCandidates26_32 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_32 : scan edge (search edge 3) 4 [109] 109 innerTail26_32 best = best := by
  rw [innerTail26_32, scan, if_neg (by decide), innerCandidates26_32_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_33
    (if 3 ≤ innerCandidates26_32.length then search edge 3 [109, 8269] innerCandidates26_32 8378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_33

def innerTail26_31 : List Nat := 7933 :: innerTail26_32
def innerCandidates26_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates26_31_checked : candidates edge 4 109 7933 innerTail26_32 best.bound = innerCandidates26_31 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_31 : scan edge (search edge 3) 4 [109] 109 innerTail26_31 best = best := by
  rw [innerTail26_31, scan, if_neg (by decide), innerCandidates26_31_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_32
    (if 3 ≤ innerCandidates26_31.length then search edge 3 [109, 7933] innerCandidates26_31 8042 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_32

def innerTail26_30 : List Nat := 7351 :: innerTail26_31
def innerCandidates26_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates26_30_checked : candidates edge 4 109 7351 innerTail26_31 best.bound = innerCandidates26_30 := by
  unfold candidates
  have row : edge 7351 = SearchData.row7351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_30 : scan edge (search edge 3) 4 [109] 109 innerTail26_30 best = best := by
  rw [innerTail26_30, scan, if_neg (by decide), innerCandidates26_30_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_31
    (if 3 ≤ innerCandidates26_30.length then search edge 3 [109, 7351] innerCandidates26_30 7460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_31

def innerTail26_29 : List Nat := 7321 :: innerTail26_30
def innerCandidates26_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates26_29_checked : candidates edge 4 109 7321 innerTail26_30 best.bound = innerCandidates26_29 := by
  unfold candidates
  have row : edge 7321 = SearchData.row7321.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_29 : scan edge (search edge 3) 4 [109] 109 innerTail26_29 best = best := by
  rw [innerTail26_29, scan, if_neg (by decide), innerCandidates26_29_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_30
    (if 3 ≤ innerCandidates26_29.length then search edge 3 [109, 7321] innerCandidates26_29 7430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_30

def innerTail26_28 : List Nat := 7297 :: innerTail26_29
def innerCandidates26_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates26_28_checked : candidates edge 4 109 7297 innerTail26_29 best.bound = innerCandidates26_28 := by
  unfold candidates
  have row : edge 7297 = SearchData.row7297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_28 : scan edge (search edge 3) 4 [109] 109 innerTail26_28 best = best := by
  rw [innerTail26_28, scan, if_neg (by decide), innerCandidates26_28_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_29
    (if 3 ≤ innerCandidates26_28.length then search edge 3 [109, 7297] innerCandidates26_28 7406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_29

def innerTail26_27 : List Nat := 6673 :: innerTail26_28
def innerCandidates26_27 : List Nat := [7351]
set_option maxRecDepth 4096 in
theorem innerCandidates26_27_checked : candidates edge 4 109 6673 innerTail26_28 best.bound = innerCandidates26_27 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_27 : scan edge (search edge 3) 4 [109] 109 innerTail26_27 best = best := by
  rw [innerTail26_27, scan, if_neg (by decide), innerCandidates26_27_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_28
    (if 3 ≤ innerCandidates26_27.length then search edge 3 [109, 6673] innerCandidates26_27 6782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_28

def innerTail26_26 : List Nat := 5791 :: innerTail26_27
def innerCandidates26_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates26_26_checked : candidates edge 4 109 5791 innerTail26_27 best.bound = innerCandidates26_26 := by
  unfold candidates
  have row : edge 5791 = SearchData.row5791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_26 : scan edge (search edge 3) 4 [109] 109 innerTail26_26 best = best := by
  rw [innerTail26_26, scan, if_neg (by decide), innerCandidates26_26_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_27
    (if 3 ≤ innerCandidates26_26.length then search edge 3 [109, 5791] innerCandidates26_26 5900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_27

def innerTail26_25 : List Nat := 5503 :: innerTail26_26
def innerCandidates26_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates26_25_checked : candidates edge 4 109 5503 innerTail26_26 best.bound = innerCandidates26_25 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_25 : scan edge (search edge 3) 4 [109] 109 innerTail26_25 best = best := by
  rw [innerTail26_25, scan, if_neg (by decide), innerCandidates26_25_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_26
    (if 3 ≤ innerCandidates26_25.length then search edge 3 [109, 5503] innerCandidates26_25 5612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_26

def innerTail26_24 : List Nat := 4957 :: innerTail26_25
def innerCandidates26_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates26_24_checked : candidates edge 4 109 4957 innerTail26_25 best.bound = innerCandidates26_24 := by
  unfold candidates
  have row : edge 4957 = SearchData.row4957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_24 : scan edge (search edge 3) 4 [109] 109 innerTail26_24 best = best := by
  rw [innerTail26_24, scan, if_neg (by decide), innerCandidates26_24_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_25
    (if 3 ≤ innerCandidates26_24.length then search edge 3 [109, 4957] innerCandidates26_24 5066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_25

def innerTail26_23 : List Nat := 4801 :: innerTail26_24
def innerCandidates26_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates26_23_checked : candidates edge 4 109 4801 innerTail26_24 best.bound = innerCandidates26_23 := by
  unfold candidates
  have row : edge 4801 = SearchData.row4801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_23 : scan edge (search edge 3) 4 [109] 109 innerTail26_23 best = best := by
  rw [innerTail26_23, scan, if_neg (by decide), innerCandidates26_23_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_24
    (if 3 ≤ innerCandidates26_23.length then search edge 3 [109, 4801] innerCandidates26_23 4910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_24

def innerTail26_22 : List Nat := 4759 :: innerTail26_23
def innerCandidates26_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates26_22_checked : candidates edge 4 109 4759 innerTail26_23 best.bound = innerCandidates26_22 := by
  unfold candidates
  have row : edge 4759 = SearchData.row4759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_22 : scan edge (search edge 3) 4 [109] 109 innerTail26_22 best = best := by
  rw [innerTail26_22, scan, if_neg (by decide), innerCandidates26_22_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_23
    (if 3 ≤ innerCandidates26_22.length then search edge 3 [109, 4759] innerCandidates26_22 4868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_23

def innerTail26_21 : List Nat := 4567 :: innerTail26_22
def innerCandidates26_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates26_21_checked : candidates edge 4 109 4567 innerTail26_22 best.bound = innerCandidates26_21 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_21 : scan edge (search edge 3) 4 [109] 109 innerTail26_21 best = best := by
  rw [innerTail26_21, scan, if_neg (by decide), innerCandidates26_21_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_22
    (if 3 ≤ innerCandidates26_21.length then search edge 3 [109, 4567] innerCandidates26_21 4676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_22

def innerTail26_20 : List Nat := 4441 :: innerTail26_21
def innerCandidates26_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates26_20_checked : candidates edge 4 109 4441 innerTail26_21 best.bound = innerCandidates26_20 := by
  unfold candidates
  have row : edge 4441 = SearchData.row4441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_20 : scan edge (search edge 3) 4 [109] 109 innerTail26_20 best = best := by
  rw [innerTail26_20, scan, if_neg (by decide), innerCandidates26_20_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_21
    (if 3 ≤ innerCandidates26_20.length then search edge 3 [109, 4441] innerCandidates26_20 4550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_21

def innerTail26_19 : List Nat := 4129 :: innerTail26_20
def innerCandidates26_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates26_19_checked : candidates edge 4 109 4129 innerTail26_20 best.bound = innerCandidates26_19 := by
  unfold candidates
  have row : edge 4129 = SearchData.row4129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_19 : scan edge (search edge 3) 4 [109] 109 innerTail26_19 best = best := by
  rw [innerTail26_19, scan, if_neg (by decide), innerCandidates26_19_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_20
    (if 3 ≤ innerCandidates26_19.length then search edge 3 [109, 4129] innerCandidates26_19 4238 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_20

def innerTail26_18 : List Nat := 3943 :: innerTail26_19
def innerCandidates26_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates26_18_checked : candidates edge 4 109 3943 innerTail26_19 best.bound = innerCandidates26_18 := by
  unfold candidates
  have row : edge 3943 = SearchData.row3943.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_18 : scan edge (search edge 3) 4 [109] 109 innerTail26_18 best = best := by
  rw [innerTail26_18, scan, if_neg (by decide), innerCandidates26_18_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_19
    (if 3 ≤ innerCandidates26_18.length then search edge 3 [109, 3943] innerCandidates26_18 4052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_19

def innerTail26_17 : List Nat := 3529 :: innerTail26_18
def innerCandidates26_17 : List Nat := [3943]
set_option maxRecDepth 4096 in
theorem innerCandidates26_17_checked : candidates edge 4 109 3529 innerTail26_18 best.bound = innerCandidates26_17 := by
  unfold candidates
  have row : edge 3529 = SearchData.row3529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_17 : scan edge (search edge 3) 4 [109] 109 innerTail26_17 best = best := by
  rw [innerTail26_17, scan, if_neg (by decide), innerCandidates26_17_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_18
    (if 3 ≤ innerCandidates26_17.length then search edge 3 [109, 3529] innerCandidates26_17 3638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_18

def innerTail26_16 : List Nat := 3331 :: innerTail26_17
def innerCandidates26_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates26_16_checked : candidates edge 4 109 3331 innerTail26_17 best.bound = innerCandidates26_16 := by
  unfold candidates
  have row : edge 3331 = SearchData.row3331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_16 : scan edge (search edge 3) 4 [109] 109 innerTail26_16 best = best := by
  rw [innerTail26_16, scan, if_neg (by decide), innerCandidates26_16_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_17
    (if 3 ≤ innerCandidates26_16.length then search edge 3 [109, 3331] innerCandidates26_16 3440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_17

def innerTail26_15 : List Nat := 2803 :: innerTail26_16
def innerCandidates26_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates26_15_checked : candidates edge 4 109 2803 innerTail26_16 best.bound = innerCandidates26_15 := by
  unfold candidates
  have row : edge 2803 = SearchData.row2803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_15 : scan edge (search edge 3) 4 [109] 109 innerTail26_15 best = best := by
  rw [innerTail26_15, scan, if_neg (by decide), innerCandidates26_15_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_16
    (if 3 ≤ innerCandidates26_15.length then search edge 3 [109, 2803] innerCandidates26_15 2912 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_16

def innerTail26_14 : List Nat := 2389 :: innerTail26_15
def innerCandidates26_14 : List Nat := [4957, 7351]
set_option maxRecDepth 4096 in
theorem innerCandidates26_14_checked : candidates edge 4 109 2389 innerTail26_15 best.bound = innerCandidates26_14 := by
  unfold candidates
  have row : edge 2389 = PrimePairCache.smallRow2389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_14 : scan edge (search edge 3) 4 [109] 109 innerTail26_14 best = best := by
  rw [innerTail26_14, scan, if_neg (by decide), innerCandidates26_14_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_15
    (if 3 ≤ innerCandidates26_14.length then search edge 3 [109, 2389] innerCandidates26_14 2498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_15

def innerTail26_13 : List Nat := 1777 :: innerTail26_14
def innerCandidates26_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates26_13_checked : candidates edge 4 109 1777 innerTail26_14 best.bound = innerCandidates26_13 := by
  unfold candidates
  have row : edge 1777 = PrimePairCache.smallRow1777.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_13 : scan edge (search edge 3) 4 [109] 109 innerTail26_13 best = best := by
  rw [innerTail26_13, scan, if_neg (by decide), innerCandidates26_13_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_14
    (if 3 ≤ innerCandidates26_13.length then search edge 3 [109, 1777] innerCandidates26_13 1886 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_14

def innerTail26_12 : List Nat := 1609 :: innerTail26_13
def innerCandidates26_12 : List Nat := [2803, 4129]
set_option maxRecDepth 4096 in
theorem innerCandidates26_12_checked : candidates edge 4 109 1609 innerTail26_13 best.bound = innerCandidates26_12 := by
  unfold candidates
  have row : edge 1609 = PrimePairCache.smallRow1609.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_12 : scan edge (search edge 3) 4 [109] 109 innerTail26_12 best = best := by
  rw [innerTail26_12, scan, if_neg (by decide), innerCandidates26_12_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_13
    (if 3 ≤ innerCandidates26_12.length then search edge 3 [109, 1609] innerCandidates26_12 1718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_13

def innerTail26_11 : List Nat := 1459 :: innerTail26_12
def innerCandidates26_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates26_11_checked : candidates edge 4 109 1459 innerTail26_12 best.bound = innerCandidates26_11 := by
  unfold candidates
  have row : edge 1459 = PrimePairCache.smallRow1459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_11 : scan edge (search edge 3) 4 [109] 109 innerTail26_11 best = best := by
  rw [innerTail26_11, scan, if_neg (by decide), innerCandidates26_11_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_12
    (if 3 ≤ innerCandidates26_11.length then search edge 3 [109, 1459] innerCandidates26_11 1568 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_12

def innerTail26_10 : List Nat := 1399 :: innerTail26_11
def innerCandidates26_10 : List Nat := [4801]
set_option maxRecDepth 4096 in
theorem innerCandidates26_10_checked : candidates edge 4 109 1399 innerTail26_11 best.bound = innerCandidates26_10 := by
  unfold candidates
  have row : edge 1399 = PrimePairCache.smallRow1399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_10 : scan edge (search edge 3) 4 [109] 109 innerTail26_10 best = best := by
  rw [innerTail26_10, scan, if_neg (by decide), innerCandidates26_10_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_11
    (if 3 ≤ innerCandidates26_10.length then search edge 3 [109, 1399] innerCandidates26_10 1508 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_11

def innerTail26_9 : List Nat := 1381 :: innerTail26_10
def innerCandidates26_9 : List Nat := [7933]
set_option maxRecDepth 4096 in
theorem innerCandidates26_9_checked : candidates edge 4 109 1381 innerTail26_10 best.bound = innerCandidates26_9 := by
  unfold candidates
  have row : edge 1381 = PrimePairCache.smallRow1381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_9 : scan edge (search edge 3) 4 [109] 109 innerTail26_9 best = best := by
  rw [innerTail26_9, scan, if_neg (by decide), innerCandidates26_9_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_10
    (if 3 ≤ innerCandidates26_9.length then search edge 3 [109, 1381] innerCandidates26_9 1490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_10

def innerTail26_8 : List Nat := 1063 :: innerTail26_9
def innerCandidates26_8 : List Nat := [1609]
set_option maxRecDepth 4096 in
theorem innerCandidates26_8_checked : candidates edge 4 109 1063 innerTail26_9 best.bound = innerCandidates26_8 := by
  unfold candidates
  have row : edge 1063 = PrimePairCache.smallRow1063.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_8 : scan edge (search edge 3) 4 [109] 109 innerTail26_8 best = best := by
  rw [innerTail26_8, scan, if_neg (by decide), innerCandidates26_8_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_9
    (if 3 ≤ innerCandidates26_8.length then search edge 3 [109, 1063] innerCandidates26_8 1172 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_9

def innerTail26_7 : List Nat := 919 :: innerTail26_8
def innerCandidates26_7 : List Nat := [4441, 4567]
set_option maxRecDepth 4096 in
theorem innerCandidates26_7_checked : candidates edge 4 109 919 innerTail26_8 best.bound = innerCandidates26_7 := by
  unfold candidates
  have row : edge 919 = PrimePairCache.smallRow919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_7 : scan edge (search edge 3) 4 [109] 109 innerTail26_7 best = best := by
  rw [innerTail26_7, scan, if_neg (by decide), innerCandidates26_7_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_8
    (if 3 ≤ innerCandidates26_7.length then search edge 3 [109, 919] innerCandidates26_7 1028 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_8

def innerTail26_6 : List Nat := 883 :: innerTail26_7
def innerCandidates26_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates26_6_checked : candidates edge 4 109 883 innerTail26_7 best.bound = innerCandidates26_6 := by
  unfold candidates
  have row : edge 883 = PrimePairCache.smallRow883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_6 : scan edge (search edge 3) 4 [109] 109 innerTail26_6 best = best := by
  rw [innerTail26_6, scan, if_neg (by decide), innerCandidates26_6_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_7
    (if 3 ≤ innerCandidates26_6.length then search edge 3 [109, 883] innerCandidates26_6 992 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_7

def innerTail26_5 : List Nat := 859 :: innerTail26_6
def innerCandidates26_5 : List Nat := [919, 1381, 3943]
set_option maxRecDepth 4096 in
theorem innerCandidates26_5_checked : candidates edge 4 109 859 innerTail26_6 best.bound = innerCandidates26_5 := by
  unfold candidates
  have row : edge 859 = PrimePairCache.smallRow859.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild26_5 : search edge 3 [109, 859] innerCandidates26_5 968 best = best := by decide

theorem innerScan26_5 : scan edge (search edge 3) 4 [109] 109 innerTail26_5 best = best := by
  rw [innerTail26_5, scan, if_neg (by decide), innerCandidates26_5_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_6
    (if 3 ≤ innerCandidates26_5.length then search edge 3 [109, 859] innerCandidates26_5 968 best else best) = best
  rw [if_pos (by decide), innerChild26_5]
  exact innerScan26_6

def innerTail26_4 : List Nat := 673 :: innerTail26_5
def innerCandidates26_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates26_4_checked : candidates edge 4 109 673 innerTail26_5 best.bound = innerCandidates26_4 := by
  unfold candidates
  have row : edge 673 = PrimePairCache.smallRow673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_4 : scan edge (search edge 3) 4 [109] 109 innerTail26_4 best = best := by
  rw [innerTail26_4, scan, if_neg (by decide), innerCandidates26_4_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_5
    (if 3 ≤ innerCandidates26_4.length then search edge 3 [109, 673] innerCandidates26_4 782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_5

def innerTail26_3 : List Nat := 661 :: innerTail26_4
def innerCandidates26_3 : List Nat := [883, 1777, 3331, 4759, 5503]
set_option maxRecDepth 4096 in
theorem innerCandidates26_3_checked : candidates edge 4 109 661 innerTail26_4 best.bound = innerCandidates26_3 := by
  unfold candidates
  have row : edge 661 = PrimePairCache.smallRow661.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild26_3 : search edge 3 [109, 661] innerCandidates26_3 770 best = best := by decide

theorem innerScan26_3 : scan edge (search edge 3) 4 [109] 109 innerTail26_3 best = best := by
  rw [innerTail26_3, scan, if_neg (by decide), innerCandidates26_3_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_4
    (if 3 ≤ innerCandidates26_3.length then search edge 3 [109, 661] innerCandidates26_3 770 best else best) = best
  rw [if_pos (by decide), innerChild26_3]
  exact innerScan26_4

def innerTail26_2 : List Nat := 313 :: innerTail26_3
def innerCandidates26_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates26_2_checked : candidates edge 4 109 313 innerTail26_3 best.bound = innerCandidates26_2 := by
  unfold candidates
  have row : edge 313 = PrimePairCache.smallRow313.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan26_2 : scan edge (search edge 3) 4 [109] 109 innerTail26_2 best = best := by
  rw [innerTail26_2, scan, if_neg (by decide), innerCandidates26_2_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_3
    (if 3 ≤ innerCandidates26_2.length then search edge 3 [109, 313] innerCandidates26_2 422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan26_3

def innerTail26_1 : List Nat := 199 :: innerTail26_2
def innerCandidates26_1 : List Nat := [673, 1609, 3529]
set_option maxRecDepth 4096 in
theorem innerCandidates26_1_checked : candidates edge 4 109 199 innerTail26_2 best.bound = innerCandidates26_1 := by
  unfold candidates
  have row : edge 199 = PrimePairCache.smallRow199.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild26_1 : search edge 3 [109, 199] innerCandidates26_1 308 best = best := by decide

theorem innerScan26_1 : scan edge (search edge 3) 4 [109] 109 innerTail26_1 best = best := by
  rw [innerTail26_1, scan, if_neg (by decide), innerCandidates26_1_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_2
    (if 3 ≤ innerCandidates26_1.length then search edge 3 [109, 199] innerCandidates26_1 308 best else best) = best
  rw [if_pos (by decide), innerChild26_1]
  exact innerScan26_2

def innerTail26_0 : List Nat := 139 :: innerTail26_1
def innerCandidates26_0 : List Nat := [661, 2803, 7933, 8269]
set_option maxRecDepth 4096 in
theorem innerCandidates26_0_checked : candidates edge 4 109 139 innerTail26_1 best.bound = innerCandidates26_0 := by
  unfold candidates
  have row : edge 139 = PrimePairCache.smallRow139.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild26_0 : search edge 3 [109, 139] innerCandidates26_0 248 best = best := by decide

theorem innerScan26_0 : scan edge (search edge 3) 4 [109] 109 innerTail26_0 best = best := by
  rw [innerTail26_0, scan, if_neg (by decide), innerCandidates26_0_checked]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_1
    (if 3 ≤ innerCandidates26_0.length then search edge 3 [109, 139] innerCandidates26_0 248 best else best) = best
  rw [if_pos (by decide), innerChild26_0]
  exact innerScan26_1

theorem rootBranch26 : search edge 4 [109] rootCandidates26 109 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [109] 109 innerTail26_0 best = best
  exact innerScan26_0

end ProjectEulerSolutions.PrimePairUpper.Search
