import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates39 : List Nat := [193, 199, 283, 397, 421, 499, 607, 619, 751, 757, 787, 1657, 1759, 2269, 2311, 2683, 2917, 3391, 3673, 4051, 5101, 5323, 5839, 6007, 6337, 7549, 7687]
set_option maxRecDepth 4096 in
theorem rootCandidates39_checked : candidates edge 5 0 181 tail40 best.bound = rootCandidates39 := by
  unfold candidates
  have row : edge 181 = PrimePairCache.smallRow181.contains := by funext q; rfl
  rw [row]
  decide

def innerTail39_27 : List Nat := []
theorem innerScan39_27 : scan edge (search edge 3) 4 [181] 181 innerTail39_27 best = best := rfl

def innerTail39_26 : List Nat := 7687 :: innerTail39_27
def innerCandidates39_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates39_26_checked : candidates edge 4 181 7687 innerTail39_27 best.bound = innerCandidates39_26 := by
  unfold candidates
  have row : edge 7687 = SearchData.row7687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_26 : scan edge (search edge 3) 4 [181] 181 innerTail39_26 best = best := by
  rw [innerTail39_26, scan, if_neg (by decide), innerCandidates39_26_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_27
    (if 3 ≤ innerCandidates39_26.length then search edge 3 [181, 7687] innerCandidates39_26 7868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_27

def innerTail39_25 : List Nat := 7549 :: innerTail39_26
def innerCandidates39_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates39_25_checked : candidates edge 4 181 7549 innerTail39_26 best.bound = innerCandidates39_25 := by
  unfold candidates
  have row : edge 7549 = SearchData.row7549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_25 : scan edge (search edge 3) 4 [181] 181 innerTail39_25 best = best := by
  rw [innerTail39_25, scan, if_neg (by decide), innerCandidates39_25_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_26
    (if 3 ≤ innerCandidates39_25.length then search edge 3 [181, 7549] innerCandidates39_25 7730 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_26

def innerTail39_24 : List Nat := 6337 :: innerTail39_25
def innerCandidates39_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates39_24_checked : candidates edge 4 181 6337 innerTail39_25 best.bound = innerCandidates39_24 := by
  unfold candidates
  have row : edge 6337 = SearchData.row6337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_24 : scan edge (search edge 3) 4 [181] 181 innerTail39_24 best = best := by
  rw [innerTail39_24, scan, if_neg (by decide), innerCandidates39_24_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_25
    (if 3 ≤ innerCandidates39_24.length then search edge 3 [181, 6337] innerCandidates39_24 6518 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_25

def innerTail39_23 : List Nat := 6007 :: innerTail39_24
def innerCandidates39_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates39_23_checked : candidates edge 4 181 6007 innerTail39_24 best.bound = innerCandidates39_23 := by
  unfold candidates
  have row : edge 6007 = SearchData.row6007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_23 : scan edge (search edge 3) 4 [181] 181 innerTail39_23 best = best := by
  rw [innerTail39_23, scan, if_neg (by decide), innerCandidates39_23_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_24
    (if 3 ≤ innerCandidates39_23.length then search edge 3 [181, 6007] innerCandidates39_23 6188 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_24

def innerTail39_22 : List Nat := 5839 :: innerTail39_23
def innerCandidates39_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates39_22_checked : candidates edge 4 181 5839 innerTail39_23 best.bound = innerCandidates39_22 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_22 : scan edge (search edge 3) 4 [181] 181 innerTail39_22 best = best := by
  rw [innerTail39_22, scan, if_neg (by decide), innerCandidates39_22_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_23
    (if 3 ≤ innerCandidates39_22.length then search edge 3 [181, 5839] innerCandidates39_22 6020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_23

def innerTail39_21 : List Nat := 5323 :: innerTail39_22
def innerCandidates39_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates39_21_checked : candidates edge 4 181 5323 innerTail39_22 best.bound = innerCandidates39_21 := by
  unfold candidates
  have row : edge 5323 = SearchData.row5323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_21 : scan edge (search edge 3) 4 [181] 181 innerTail39_21 best = best := by
  rw [innerTail39_21, scan, if_neg (by decide), innerCandidates39_21_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_22
    (if 3 ≤ innerCandidates39_21.length then search edge 3 [181, 5323] innerCandidates39_21 5504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_22

def innerTail39_20 : List Nat := 5101 :: innerTail39_21
def innerCandidates39_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates39_20_checked : candidates edge 4 181 5101 innerTail39_21 best.bound = innerCandidates39_20 := by
  unfold candidates
  have row : edge 5101 = SearchData.row5101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_20 : scan edge (search edge 3) 4 [181] 181 innerTail39_20 best = best := by
  rw [innerTail39_20, scan, if_neg (by decide), innerCandidates39_20_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_21
    (if 3 ≤ innerCandidates39_20.length then search edge 3 [181, 5101] innerCandidates39_20 5282 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_21

def innerTail39_19 : List Nat := 4051 :: innerTail39_20
def innerCandidates39_19 : List Nat := [7549]
set_option maxRecDepth 4096 in
theorem innerCandidates39_19_checked : candidates edge 4 181 4051 innerTail39_20 best.bound = innerCandidates39_19 := by
  unfold candidates
  have row : edge 4051 = SearchData.row4051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_19 : scan edge (search edge 3) 4 [181] 181 innerTail39_19 best = best := by
  rw [innerTail39_19, scan, if_neg (by decide), innerCandidates39_19_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_20
    (if 3 ≤ innerCandidates39_19.length then search edge 3 [181, 4051] innerCandidates39_19 4232 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_20

def innerTail39_18 : List Nat := 3673 :: innerTail39_19
def innerCandidates39_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates39_18_checked : candidates edge 4 181 3673 innerTail39_19 best.bound = innerCandidates39_18 := by
  unfold candidates
  have row : edge 3673 = SearchData.row3673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_18 : scan edge (search edge 3) 4 [181] 181 innerTail39_18 best = best := by
  rw [innerTail39_18, scan, if_neg (by decide), innerCandidates39_18_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_19
    (if 3 ≤ innerCandidates39_18.length then search edge 3 [181, 3673] innerCandidates39_18 3854 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_19

def innerTail39_17 : List Nat := 3391 :: innerTail39_18
def innerCandidates39_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates39_17_checked : candidates edge 4 181 3391 innerTail39_18 best.bound = innerCandidates39_17 := by
  unfold candidates
  have row : edge 3391 = SearchData.row3391.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_17 : scan edge (search edge 3) 4 [181] 181 innerTail39_17 best = best := by
  rw [innerTail39_17, scan, if_neg (by decide), innerCandidates39_17_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_18
    (if 3 ≤ innerCandidates39_17.length then search edge 3 [181, 3391] innerCandidates39_17 3572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_18

def innerTail39_16 : List Nat := 2917 :: innerTail39_17
def innerCandidates39_16 : List Nat := [7549]
set_option maxRecDepth 4096 in
theorem innerCandidates39_16_checked : candidates edge 4 181 2917 innerTail39_17 best.bound = innerCandidates39_16 := by
  unfold candidates
  have row : edge 2917 = SearchData.row2917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_16 : scan edge (search edge 3) 4 [181] 181 innerTail39_16 best = best := by
  rw [innerTail39_16, scan, if_neg (by decide), innerCandidates39_16_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_17
    (if 3 ≤ innerCandidates39_16.length then search edge 3 [181, 2917] innerCandidates39_16 3098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_17

def innerTail39_15 : List Nat := 2683 :: innerTail39_16
def innerCandidates39_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates39_15_checked : candidates edge 4 181 2683 innerTail39_16 best.bound = innerCandidates39_15 := by
  unfold candidates
  have row : edge 2683 = SearchData.row2683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_15 : scan edge (search edge 3) 4 [181] 181 innerTail39_15 best = best := by
  rw [innerTail39_15, scan, if_neg (by decide), innerCandidates39_15_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_16
    (if 3 ≤ innerCandidates39_15.length then search edge 3 [181, 2683] innerCandidates39_15 2864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_16

def innerTail39_14 : List Nat := 2311 :: innerTail39_15
def innerCandidates39_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates39_14_checked : candidates edge 4 181 2311 innerTail39_15 best.bound = innerCandidates39_14 := by
  unfold candidates
  have row : edge 2311 = PrimePairCache.smallRow2311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_14 : scan edge (search edge 3) 4 [181] 181 innerTail39_14 best = best := by
  rw [innerTail39_14, scan, if_neg (by decide), innerCandidates39_14_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_15
    (if 3 ≤ innerCandidates39_14.length then search edge 3 [181, 2311] innerCandidates39_14 2492 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_15

def innerTail39_13 : List Nat := 2269 :: innerTail39_14
def innerCandidates39_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates39_13_checked : candidates edge 4 181 2269 innerTail39_14 best.bound = innerCandidates39_13 := by
  unfold candidates
  have row : edge 2269 = PrimePairCache.smallRow2269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_13 : scan edge (search edge 3) 4 [181] 181 innerTail39_13 best = best := by
  rw [innerTail39_13, scan, if_neg (by decide), innerCandidates39_13_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_14
    (if 3 ≤ innerCandidates39_13.length then search edge 3 [181, 2269] innerCandidates39_13 2450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_14

def innerTail39_12 : List Nat := 1759 :: innerTail39_13
def innerCandidates39_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates39_12_checked : candidates edge 4 181 1759 innerTail39_13 best.bound = innerCandidates39_12 := by
  unfold candidates
  have row : edge 1759 = PrimePairCache.smallRow1759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_12 : scan edge (search edge 3) 4 [181] 181 innerTail39_12 best = best := by
  rw [innerTail39_12, scan, if_neg (by decide), innerCandidates39_12_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_13
    (if 3 ≤ innerCandidates39_12.length then search edge 3 [181, 1759] innerCandidates39_12 1940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_13

def innerTail39_11 : List Nat := 1657 :: innerTail39_12
def innerCandidates39_11 : List Nat := [2683, 2917, 5101]
set_option maxRecDepth 4096 in
theorem innerCandidates39_11_checked : candidates edge 4 181 1657 innerTail39_12 best.bound = innerCandidates39_11 := by
  unfold candidates
  have row : edge 1657 = PrimePairCache.smallRow1657.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild39_11 : search edge 3 [181, 1657] innerCandidates39_11 1838 best = best := by decide

theorem innerScan39_11 : scan edge (search edge 3) 4 [181] 181 innerTail39_11 best = best := by
  rw [innerTail39_11, scan, if_neg (by decide), innerCandidates39_11_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_12
    (if 3 ≤ innerCandidates39_11.length then search edge 3 [181, 1657] innerCandidates39_11 1838 best else best) = best
  rw [if_pos (by decide), innerChild39_11]
  exact innerScan39_12

def innerTail39_10 : List Nat := 787 :: innerTail39_11
def innerCandidates39_10 : List Nat := [6007]
set_option maxRecDepth 4096 in
theorem innerCandidates39_10_checked : candidates edge 4 181 787 innerTail39_11 best.bound = innerCandidates39_10 := by
  unfold candidates
  have row : edge 787 = PrimePairCache.smallRow787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_10 : scan edge (search edge 3) 4 [181] 181 innerTail39_10 best = best := by
  rw [innerTail39_10, scan, if_neg (by decide), innerCandidates39_10_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_11
    (if 3 ≤ innerCandidates39_10.length then search edge 3 [181, 787] innerCandidates39_10 968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_11

def innerTail39_9 : List Nat := 757 :: innerTail39_10
def innerCandidates39_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates39_9_checked : candidates edge 4 181 757 innerTail39_10 best.bound = innerCandidates39_9 := by
  unfold candidates
  have row : edge 757 = PrimePairCache.smallRow757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_9 : scan edge (search edge 3) 4 [181] 181 innerTail39_9 best = best := by
  rw [innerTail39_9, scan, if_neg (by decide), innerCandidates39_9_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_10
    (if 3 ≤ innerCandidates39_9.length then search edge 3 [181, 757] innerCandidates39_9 938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_10

def innerTail39_8 : List Nat := 751 :: innerTail39_9
def innerCandidates39_8 : List Nat := [787]
set_option maxRecDepth 4096 in
theorem innerCandidates39_8_checked : candidates edge 4 181 751 innerTail39_9 best.bound = innerCandidates39_8 := by
  unfold candidates
  have row : edge 751 = PrimePairCache.smallRow751.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_8 : scan edge (search edge 3) 4 [181] 181 innerTail39_8 best = best := by
  rw [innerTail39_8, scan, if_neg (by decide), innerCandidates39_8_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_9
    (if 3 ≤ innerCandidates39_8.length then search edge 3 [181, 751] innerCandidates39_8 932 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_9

def innerTail39_7 : List Nat := 619 :: innerTail39_8
def innerCandidates39_7 : List Nat := [7687]
set_option maxRecDepth 4096 in
theorem innerCandidates39_7_checked : candidates edge 4 181 619 innerTail39_8 best.bound = innerCandidates39_7 := by
  unfold candidates
  have row : edge 619 = PrimePairCache.smallRow619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_7 : scan edge (search edge 3) 4 [181] 181 innerTail39_7 best = best := by
  rw [innerTail39_7, scan, if_neg (by decide), innerCandidates39_7_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_8
    (if 3 ≤ innerCandidates39_7.length then search edge 3 [181, 619] innerCandidates39_7 800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_8

def innerTail39_6 : List Nat := 607 :: innerTail39_7
def innerCandidates39_6 : List Nat := [619, 6337]
set_option maxRecDepth 4096 in
theorem innerCandidates39_6_checked : candidates edge 4 181 607 innerTail39_7 best.bound = innerCandidates39_6 := by
  unfold candidates
  have row : edge 607 = PrimePairCache.smallRow607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_6 : scan edge (search edge 3) 4 [181] 181 innerTail39_6 best = best := by
  rw [innerTail39_6, scan, if_neg (by decide), innerCandidates39_6_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_7
    (if 3 ≤ innerCandidates39_6.length then search edge 3 [181, 607] innerCandidates39_6 788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_7

def innerTail39_5 : List Nat := 499 :: innerTail39_6
def innerCandidates39_5 : List Nat := [4051, 7687]
set_option maxRecDepth 4096 in
theorem innerCandidates39_5_checked : candidates edge 4 181 499 innerTail39_6 best.bound = innerCandidates39_5 := by
  unfold candidates
  have row : edge 499 = PrimePairCache.smallRow499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_5 : scan edge (search edge 3) 4 [181] 181 innerTail39_5 best = best := by
  rw [innerTail39_5, scan, if_neg (by decide), innerCandidates39_5_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_6
    (if 3 ≤ innerCandidates39_5.length then search edge 3 [181, 499] innerCandidates39_5 680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_6

def innerTail39_4 : List Nat := 421 :: innerTail39_5
def innerCandidates39_4 : List Nat := [607, 1657]
set_option maxRecDepth 4096 in
theorem innerCandidates39_4_checked : candidates edge 4 181 421 innerTail39_5 best.bound = innerCandidates39_4 := by
  unfold candidates
  have row : edge 421 = PrimePairCache.smallRow421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_4 : scan edge (search edge 3) 4 [181] 181 innerTail39_4 best = best := by
  rw [innerTail39_4, scan, if_neg (by decide), innerCandidates39_4_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_5
    (if 3 ≤ innerCandidates39_4.length then search edge 3 [181, 421] innerCandidates39_4 602 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_5

def innerTail39_3 : List Nat := 397 :: innerTail39_4
def innerCandidates39_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates39_3_checked : candidates edge 4 181 397 innerTail39_4 best.bound = innerCandidates39_3 := by
  unfold candidates
  have row : edge 397 = PrimePairCache.smallRow397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_3 : scan edge (search edge 3) 4 [181] 181 innerTail39_3 best = best := by
  rw [innerTail39_3, scan, if_neg (by decide), innerCandidates39_3_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_4
    (if 3 ≤ innerCandidates39_3.length then search edge 3 [181, 397] innerCandidates39_3 578 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_4

def innerTail39_2 : List Nat := 283 :: innerTail39_3
def innerCandidates39_2 : List Nat := [397, 1657]
set_option maxRecDepth 4096 in
theorem innerCandidates39_2_checked : candidates edge 4 181 283 innerTail39_3 best.bound = innerCandidates39_2 := by
  unfold candidates
  have row : edge 283 = PrimePairCache.smallRow283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_2 : scan edge (search edge 3) 4 [181] 181 innerTail39_2 best = best := by
  rw [innerTail39_2, scan, if_neg (by decide), innerCandidates39_2_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_3
    (if 3 ≤ innerCandidates39_2.length then search edge 3 [181, 283] innerCandidates39_2 464 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_3

def innerTail39_1 : List Nat := 199 :: innerTail39_2
def innerCandidates39_1 : List Nat := [751, 2917]
set_option maxRecDepth 4096 in
theorem innerCandidates39_1_checked : candidates edge 4 181 199 innerTail39_2 best.bound = innerCandidates39_1 := by
  unfold candidates
  have row : edge 199 = PrimePairCache.smallRow199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_1 : scan edge (search edge 3) 4 [181] 181 innerTail39_1 best = best := by
  rw [innerTail39_1, scan, if_neg (by decide), innerCandidates39_1_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_2
    (if 3 ≤ innerCandidates39_1.length then search edge 3 [181, 199] innerCandidates39_1 380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_2

def innerTail39_0 : List Nat := 193 :: innerTail39_1
def innerCandidates39_0 : List Nat := [283, 751]
set_option maxRecDepth 4096 in
theorem innerCandidates39_0_checked : candidates edge 4 181 193 innerTail39_1 best.bound = innerCandidates39_0 := by
  unfold candidates
  have row : edge 193 = PrimePairCache.smallRow193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan39_0 : scan edge (search edge 3) 4 [181] 181 innerTail39_0 best = best := by
  rw [innerTail39_0, scan, if_neg (by decide), innerCandidates39_0_checked]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_1
    (if 3 ≤ innerCandidates39_0.length then search edge 3 [181, 193] innerCandidates39_0 374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan39_1

theorem rootBranch39 : search edge 4 [181] rootCandidates39 181 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [181] 181 innerTail39_0 best = best
  exact innerScan39_0

def rootCandidates38 : List Nat := [269, 317, 383, 593, 719, 743, 953, 1289, 1451, 1487, 1787, 1847, 2039, 2309, 2339, 2381, 2579, 3137, 3719, 4349, 4517, 4649, 4787, 4817, 4973, 5153, 5279, 5867, 7331, 7541, 7547, 7673, 7877, 8009]
set_option maxRecDepth 4096 in
theorem rootCandidates38_checked : candidates edge 5 0 179 tail39 best.bound = rootCandidates38 := by
  unfold candidates
  have row : edge 179 = PrimePairCache.smallRow179.contains := by funext q; rfl
  rw [row]
  decide

def innerTail38_34 : List Nat := []
theorem innerScan38_34 : scan edge (search edge 3) 4 [179] 179 innerTail38_34 best = best := rfl

def innerTail38_33 : List Nat := 8009 :: innerTail38_34
def innerCandidates38_33 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates38_33_checked : candidates edge 4 179 8009 innerTail38_34 best.bound = innerCandidates38_33 := by
  unfold candidates
  have row : edge 8009 = SearchData.row8009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_33 : scan edge (search edge 3) 4 [179] 179 innerTail38_33 best = best := by
  rw [innerTail38_33, scan, if_neg (by decide), innerCandidates38_33_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_34
    (if 3 ≤ innerCandidates38_33.length then search edge 3 [179, 8009] innerCandidates38_33 8188 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_34

def innerTail38_32 : List Nat := 7877 :: innerTail38_33
def innerCandidates38_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates38_32_checked : candidates edge 4 179 7877 innerTail38_33 best.bound = innerCandidates38_32 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_32 : scan edge (search edge 3) 4 [179] 179 innerTail38_32 best = best := by
  rw [innerTail38_32, scan, if_neg (by decide), innerCandidates38_32_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_33
    (if 3 ≤ innerCandidates38_32.length then search edge 3 [179, 7877] innerCandidates38_32 8056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_33

def innerTail38_31 : List Nat := 7673 :: innerTail38_32
def innerCandidates38_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates38_31_checked : candidates edge 4 179 7673 innerTail38_32 best.bound = innerCandidates38_31 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_31 : scan edge (search edge 3) 4 [179] 179 innerTail38_31 best = best := by
  rw [innerTail38_31, scan, if_neg (by decide), innerCandidates38_31_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_32
    (if 3 ≤ innerCandidates38_31.length then search edge 3 [179, 7673] innerCandidates38_31 7852 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_32

def innerTail38_30 : List Nat := 7547 :: innerTail38_31
def innerCandidates38_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates38_30_checked : candidates edge 4 179 7547 innerTail38_31 best.bound = innerCandidates38_30 := by
  unfold candidates
  have row : edge 7547 = SearchData.row7547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_30 : scan edge (search edge 3) 4 [179] 179 innerTail38_30 best = best := by
  rw [innerTail38_30, scan, if_neg (by decide), innerCandidates38_30_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_31
    (if 3 ≤ innerCandidates38_30.length then search edge 3 [179, 7547] innerCandidates38_30 7726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_31

def innerTail38_29 : List Nat := 7541 :: innerTail38_30
def innerCandidates38_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates38_29_checked : candidates edge 4 179 7541 innerTail38_30 best.bound = innerCandidates38_29 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_29 : scan edge (search edge 3) 4 [179] 179 innerTail38_29 best = best := by
  rw [innerTail38_29, scan, if_neg (by decide), innerCandidates38_29_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_30
    (if 3 ≤ innerCandidates38_29.length then search edge 3 [179, 7541] innerCandidates38_29 7720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_30

def innerTail38_28 : List Nat := 7331 :: innerTail38_29
def innerCandidates38_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates38_28_checked : candidates edge 4 179 7331 innerTail38_29 best.bound = innerCandidates38_28 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_28 : scan edge (search edge 3) 4 [179] 179 innerTail38_28 best = best := by
  rw [innerTail38_28, scan, if_neg (by decide), innerCandidates38_28_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_29
    (if 3 ≤ innerCandidates38_28.length then search edge 3 [179, 7331] innerCandidates38_28 7510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_29

def innerTail38_27 : List Nat := 5867 :: innerTail38_28
def innerCandidates38_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates38_27_checked : candidates edge 4 179 5867 innerTail38_28 best.bound = innerCandidates38_27 := by
  unfold candidates
  have row : edge 5867 = SearchData.row5867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_27 : scan edge (search edge 3) 4 [179] 179 innerTail38_27 best = best := by
  rw [innerTail38_27, scan, if_neg (by decide), innerCandidates38_27_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_28
    (if 3 ≤ innerCandidates38_27.length then search edge 3 [179, 5867] innerCandidates38_27 6046 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_28

def innerTail38_26 : List Nat := 5279 :: innerTail38_27
def innerCandidates38_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates38_26_checked : candidates edge 4 179 5279 innerTail38_27 best.bound = innerCandidates38_26 := by
  unfold candidates
  have row : edge 5279 = SearchData.row5279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_26 : scan edge (search edge 3) 4 [179] 179 innerTail38_26 best = best := by
  rw [innerTail38_26, scan, if_neg (by decide), innerCandidates38_26_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_27
    (if 3 ≤ innerCandidates38_26.length then search edge 3 [179, 5279] innerCandidates38_26 5458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_27

def innerTail38_25 : List Nat := 5153 :: innerTail38_26
def innerCandidates38_25 : List Nat := [5867]
set_option maxRecDepth 4096 in
theorem innerCandidates38_25_checked : candidates edge 4 179 5153 innerTail38_26 best.bound = innerCandidates38_25 := by
  unfold candidates
  have row : edge 5153 = SearchData.row5153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_25 : scan edge (search edge 3) 4 [179] 179 innerTail38_25 best = best := by
  rw [innerTail38_25, scan, if_neg (by decide), innerCandidates38_25_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_26
    (if 3 ≤ innerCandidates38_25.length then search edge 3 [179, 5153] innerCandidates38_25 5332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_26

def innerTail38_24 : List Nat := 4973 :: innerTail38_25
def innerCandidates38_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates38_24_checked : candidates edge 4 179 4973 innerTail38_25 best.bound = innerCandidates38_24 := by
  unfold candidates
  have row : edge 4973 = SearchData.row4973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_24 : scan edge (search edge 3) 4 [179] 179 innerTail38_24 best = best := by
  rw [innerTail38_24, scan, if_neg (by decide), innerCandidates38_24_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_25
    (if 3 ≤ innerCandidates38_24.length then search edge 3 [179, 4973] innerCandidates38_24 5152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_25

def innerTail38_23 : List Nat := 4817 :: innerTail38_24
def innerCandidates38_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates38_23_checked : candidates edge 4 179 4817 innerTail38_24 best.bound = innerCandidates38_23 := by
  unfold candidates
  have row : edge 4817 = SearchData.row4817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_23 : scan edge (search edge 3) 4 [179] 179 innerTail38_23 best = best := by
  rw [innerTail38_23, scan, if_neg (by decide), innerCandidates38_23_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_24
    (if 3 ≤ innerCandidates38_23.length then search edge 3 [179, 4817] innerCandidates38_23 4996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_24

def innerTail38_22 : List Nat := 4787 :: innerTail38_23
def innerCandidates38_22 : List Nat := [8009]
set_option maxRecDepth 4096 in
theorem innerCandidates38_22_checked : candidates edge 4 179 4787 innerTail38_23 best.bound = innerCandidates38_22 := by
  unfold candidates
  have row : edge 4787 = SearchData.row4787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_22 : scan edge (search edge 3) 4 [179] 179 innerTail38_22 best = best := by
  rw [innerTail38_22, scan, if_neg (by decide), innerCandidates38_22_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_23
    (if 3 ≤ innerCandidates38_22.length then search edge 3 [179, 4787] innerCandidates38_22 4966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_23

def innerTail38_21 : List Nat := 4649 :: innerTail38_22
def innerCandidates38_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates38_21_checked : candidates edge 4 179 4649 innerTail38_22 best.bound = innerCandidates38_21 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_21 : scan edge (search edge 3) 4 [179] 179 innerTail38_21 best = best := by
  rw [innerTail38_21, scan, if_neg (by decide), innerCandidates38_21_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_22
    (if 3 ≤ innerCandidates38_21.length then search edge 3 [179, 4649] innerCandidates38_21 4828 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_22

def innerTail38_20 : List Nat := 4517 :: innerTail38_21
def innerCandidates38_20 : List Nat := [5279]
set_option maxRecDepth 4096 in
theorem innerCandidates38_20_checked : candidates edge 4 179 4517 innerTail38_21 best.bound = innerCandidates38_20 := by
  unfold candidates
  have row : edge 4517 = SearchData.row4517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_20 : scan edge (search edge 3) 4 [179] 179 innerTail38_20 best = best := by
  rw [innerTail38_20, scan, if_neg (by decide), innerCandidates38_20_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_21
    (if 3 ≤ innerCandidates38_20.length then search edge 3 [179, 4517] innerCandidates38_20 4696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_21

def innerTail38_19 : List Nat := 4349 :: innerTail38_20
def innerCandidates38_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates38_19_checked : candidates edge 4 179 4349 innerTail38_20 best.bound = innerCandidates38_19 := by
  unfold candidates
  have row : edge 4349 = SearchData.row4349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_19 : scan edge (search edge 3) 4 [179] 179 innerTail38_19 best = best := by
  rw [innerTail38_19, scan, if_neg (by decide), innerCandidates38_19_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_20
    (if 3 ≤ innerCandidates38_19.length then search edge 3 [179, 4349] innerCandidates38_19 4528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_20

def innerTail38_18 : List Nat := 3719 :: innerTail38_19
def innerCandidates38_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates38_18_checked : candidates edge 4 179 3719 innerTail38_19 best.bound = innerCandidates38_18 := by
  unfold candidates
  have row : edge 3719 = SearchData.row3719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_18 : scan edge (search edge 3) 4 [179] 179 innerTail38_18 best = best := by
  rw [innerTail38_18, scan, if_neg (by decide), innerCandidates38_18_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_19
    (if 3 ≤ innerCandidates38_18.length then search edge 3 [179, 3719] innerCandidates38_18 3898 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_19

def innerTail38_17 : List Nat := 3137 :: innerTail38_18
def innerCandidates38_17 : List Nat := [4349, 7331]
set_option maxRecDepth 4096 in
theorem innerCandidates38_17_checked : candidates edge 4 179 3137 innerTail38_18 best.bound = innerCandidates38_17 := by
  unfold candidates
  have row : edge 3137 = SearchData.row3137.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_17 : scan edge (search edge 3) 4 [179] 179 innerTail38_17 best = best := by
  rw [innerTail38_17, scan, if_neg (by decide), innerCandidates38_17_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_18
    (if 3 ≤ innerCandidates38_17.length then search edge 3 [179, 3137] innerCandidates38_17 3316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_18

def innerTail38_16 : List Nat := 2579 :: innerTail38_17
def innerCandidates38_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates38_16_checked : candidates edge 4 179 2579 innerTail38_17 best.bound = innerCandidates38_16 := by
  unfold candidates
  have row : edge 2579 = PrimePairCache.smallRow2579.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_16 : scan edge (search edge 3) 4 [179] 179 innerTail38_16 best = best := by
  rw [innerTail38_16, scan, if_neg (by decide), innerCandidates38_16_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_17
    (if 3 ≤ innerCandidates38_16.length then search edge 3 [179, 2579] innerCandidates38_16 2758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_17

def innerTail38_15 : List Nat := 2381 :: innerTail38_16
def innerCandidates38_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates38_15_checked : candidates edge 4 179 2381 innerTail38_16 best.bound = innerCandidates38_15 := by
  unfold candidates
  have row : edge 2381 = PrimePairCache.smallRow2381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_15 : scan edge (search edge 3) 4 [179] 179 innerTail38_15 best = best := by
  rw [innerTail38_15, scan, if_neg (by decide), innerCandidates38_15_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_16
    (if 3 ≤ innerCandidates38_15.length then search edge 3 [179, 2381] innerCandidates38_15 2560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_16

def innerTail38_14 : List Nat := 2339 :: innerTail38_15
def innerCandidates38_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates38_14_checked : candidates edge 4 179 2339 innerTail38_15 best.bound = innerCandidates38_14 := by
  unfold candidates
  have row : edge 2339 = PrimePairCache.smallRow2339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_14 : scan edge (search edge 3) 4 [179] 179 innerTail38_14 best = best := by
  rw [innerTail38_14, scan, if_neg (by decide), innerCandidates38_14_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_15
    (if 3 ≤ innerCandidates38_14.length then search edge 3 [179, 2339] innerCandidates38_14 2518 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_15

def innerTail38_13 : List Nat := 2309 :: innerTail38_14
def innerCandidates38_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates38_13_checked : candidates edge 4 179 2309 innerTail38_14 best.bound = innerCandidates38_13 := by
  unfold candidates
  have row : edge 2309 = PrimePairCache.smallRow2309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_13 : scan edge (search edge 3) 4 [179] 179 innerTail38_13 best = best := by
  rw [innerTail38_13, scan, if_neg (by decide), innerCandidates38_13_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_14
    (if 3 ≤ innerCandidates38_13.length then search edge 3 [179, 2309] innerCandidates38_13 2488 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_14

def innerTail38_12 : List Nat := 2039 :: innerTail38_13
def innerCandidates38_12 : List Nat := [7673]
set_option maxRecDepth 4096 in
theorem innerCandidates38_12_checked : candidates edge 4 179 2039 innerTail38_13 best.bound = innerCandidates38_12 := by
  unfold candidates
  have row : edge 2039 = PrimePairCache.smallRow2039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_12 : scan edge (search edge 3) 4 [179] 179 innerTail38_12 best = best := by
  rw [innerTail38_12, scan, if_neg (by decide), innerCandidates38_12_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_13
    (if 3 ≤ innerCandidates38_12.length then search edge 3 [179, 2039] innerCandidates38_12 2218 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_13

def innerTail38_11 : List Nat := 1847 :: innerTail38_12
def innerCandidates38_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates38_11_checked : candidates edge 4 179 1847 innerTail38_12 best.bound = innerCandidates38_11 := by
  unfold candidates
  have row : edge 1847 = PrimePairCache.smallRow1847.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_11 : scan edge (search edge 3) 4 [179] 179 innerTail38_11 best = best := by
  rw [innerTail38_11, scan, if_neg (by decide), innerCandidates38_11_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_12
    (if 3 ≤ innerCandidates38_11.length then search edge 3 [179, 1847] innerCandidates38_11 2026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_12

def innerTail38_10 : List Nat := 1787 :: innerTail38_11
def innerCandidates38_10 : List Nat := [4349, 7547]
set_option maxRecDepth 4096 in
theorem innerCandidates38_10_checked : candidates edge 4 179 1787 innerTail38_11 best.bound = innerCandidates38_10 := by
  unfold candidates
  have row : edge 1787 = PrimePairCache.smallRow1787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_10 : scan edge (search edge 3) 4 [179] 179 innerTail38_10 best = best := by
  rw [innerTail38_10, scan, if_neg (by decide), innerCandidates38_10_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_11
    (if 3 ≤ innerCandidates38_10.length then search edge 3 [179, 1787] innerCandidates38_10 1966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_11

def innerTail38_9 : List Nat := 1487 :: innerTail38_10
def innerCandidates38_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates38_9_checked : candidates edge 4 179 1487 innerTail38_10 best.bound = innerCandidates38_9 := by
  unfold candidates
  have row : edge 1487 = PrimePairCache.smallRow1487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_9 : scan edge (search edge 3) 4 [179] 179 innerTail38_9 best = best := by
  rw [innerTail38_9, scan, if_neg (by decide), innerCandidates38_9_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_10
    (if 3 ≤ innerCandidates38_9.length then search edge 3 [179, 1487] innerCandidates38_9 1666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_10

def innerTail38_8 : List Nat := 1451 :: innerTail38_9
def innerCandidates38_8 : List Nat := [2039]
set_option maxRecDepth 4096 in
theorem innerCandidates38_8_checked : candidates edge 4 179 1451 innerTail38_9 best.bound = innerCandidates38_8 := by
  unfold candidates
  have row : edge 1451 = PrimePairCache.smallRow1451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_8 : scan edge (search edge 3) 4 [179] 179 innerTail38_8 best = best := by
  rw [innerTail38_8, scan, if_neg (by decide), innerCandidates38_8_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_9
    (if 3 ≤ innerCandidates38_8.length then search edge 3 [179, 1451] innerCandidates38_8 1630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_9

def innerTail38_7 : List Nat := 1289 :: innerTail38_8
def innerCandidates38_7 : List Nat := [2039, 7541]
set_option maxRecDepth 4096 in
theorem innerCandidates38_7_checked : candidates edge 4 179 1289 innerTail38_8 best.bound = innerCandidates38_7 := by
  unfold candidates
  have row : edge 1289 = PrimePairCache.smallRow1289.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_7 : scan edge (search edge 3) 4 [179] 179 innerTail38_7 best = best := by
  rw [innerTail38_7, scan, if_neg (by decide), innerCandidates38_7_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_8
    (if 3 ≤ innerCandidates38_7.length then search edge 3 [179, 1289] innerCandidates38_7 1468 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_8

def innerTail38_6 : List Nat := 953 :: innerTail38_7
def innerCandidates38_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates38_6_checked : candidates edge 4 179 953 innerTail38_7 best.bound = innerCandidates38_6 := by
  unfold candidates
  have row : edge 953 = PrimePairCache.smallRow953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_6 : scan edge (search edge 3) 4 [179] 179 innerTail38_6 best = best := by
  rw [innerTail38_6, scan, if_neg (by decide), innerCandidates38_6_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_7
    (if 3 ≤ innerCandidates38_6.length then search edge 3 [179, 953] innerCandidates38_6 1132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_7

def innerTail38_5 : List Nat := 743 :: innerTail38_6
def innerCandidates38_5 : List Nat := [2309]
set_option maxRecDepth 4096 in
theorem innerCandidates38_5_checked : candidates edge 4 179 743 innerTail38_6 best.bound = innerCandidates38_5 := by
  unfold candidates
  have row : edge 743 = PrimePairCache.smallRow743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_5 : scan edge (search edge 3) 4 [179] 179 innerTail38_5 best = best := by
  rw [innerTail38_5, scan, if_neg (by decide), innerCandidates38_5_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_6
    (if 3 ≤ innerCandidates38_5.length then search edge 3 [179, 743] innerCandidates38_5 922 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_6

def innerTail38_4 : List Nat := 719 :: innerTail38_5
def innerCandidates38_4 : List Nat := [2579, 4973]
set_option maxRecDepth 4096 in
theorem innerCandidates38_4_checked : candidates edge 4 179 719 innerTail38_5 best.bound = innerCandidates38_4 := by
  unfold candidates
  have row : edge 719 = PrimePairCache.smallRow719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_4 : scan edge (search edge 3) 4 [179] 179 innerTail38_4 best = best := by
  rw [innerTail38_4, scan, if_neg (by decide), innerCandidates38_4_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_5
    (if 3 ≤ innerCandidates38_4.length then search edge 3 [179, 719] innerCandidates38_4 898 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_5

def innerTail38_3 : List Nat := 593 :: innerTail38_4
def innerCandidates38_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates38_3_checked : candidates edge 4 179 593 innerTail38_4 best.bound = innerCandidates38_3 := by
  unfold candidates
  have row : edge 593 = PrimePairCache.smallRow593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_3 : scan edge (search edge 3) 4 [179] 179 innerTail38_3 best = best := by
  rw [innerTail38_3, scan, if_neg (by decide), innerCandidates38_3_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_4
    (if 3 ≤ innerCandidates38_3.length then search edge 3 [179, 593] innerCandidates38_3 772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_4

def innerTail38_2 : List Nat := 383 :: innerTail38_3
def innerCandidates38_2 : List Nat := [7331, 7877]
set_option maxRecDepth 4096 in
theorem innerCandidates38_2_checked : candidates edge 4 179 383 innerTail38_3 best.bound = innerCandidates38_2 := by
  unfold candidates
  have row : edge 383 = PrimePairCache.smallRow383.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_2 : scan edge (search edge 3) 4 [179] 179 innerTail38_2 best = best := by
  rw [innerTail38_2, scan, if_neg (by decide), innerCandidates38_2_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_3
    (if 3 ≤ innerCandidates38_2.length then search edge 3 [179, 383] innerCandidates38_2 562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_3

def innerTail38_1 : List Nat := 317 :: innerTail38_2
def innerCandidates38_1 : List Nat := [4973]
set_option maxRecDepth 4096 in
theorem innerCandidates38_1_checked : candidates edge 4 179 317 innerTail38_2 best.bound = innerCandidates38_1 := by
  unfold candidates
  have row : edge 317 = PrimePairCache.smallRow317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_1 : scan edge (search edge 3) 4 [179] 179 innerTail38_1 best = best := by
  rw [innerTail38_1, scan, if_neg (by decide), innerCandidates38_1_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_2
    (if 3 ≤ innerCandidates38_1.length then search edge 3 [179, 317] innerCandidates38_1 496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_2

def innerTail38_0 : List Nat := 269 :: innerTail38_1
def innerCandidates38_0 : List Nat := [317]
set_option maxRecDepth 4096 in
theorem innerCandidates38_0_checked : candidates edge 4 179 269 innerTail38_1 best.bound = innerCandidates38_0 := by
  unfold candidates
  have row : edge 269 = PrimePairCache.smallRow269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan38_0 : scan edge (search edge 3) 4 [179] 179 innerTail38_0 best = best := by
  rw [innerTail38_0, scan, if_neg (by decide), innerCandidates38_0_checked]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_1
    (if 3 ≤ innerCandidates38_0.length then search edge 3 [179, 269] innerCandidates38_0 448 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan38_1

theorem rootBranch38 : search edge 4 [179] rootCandidates38 179 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [179] 179 innerTail38_0 best = best
  exact innerScan38_0

end ProjectEulerSolutions.PrimePairUpper.Search
