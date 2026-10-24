import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates191 : List Nat := [1361, 1367, 1427, 1451, 1601, 1847, 2621, 2687, 2729, 3083, 3209, 3251, 3797, 4013, 4889, 4931, 5171, 5501, 5717, 6449, 6719, 6911, 6977, 7433, 7577, 8069, 8237]
set_option maxRecDepth 4096 in
theorem rootCandidates191_checked : candidates edge 5 0 1181 tail192 best.bound = rootCandidates191 := by
  unfold candidates
  have row : edge 1181 = PrimePairCache.smallRow1181.contains := by funext q; rfl
  rw [row]
  decide

def innerTail191_27 : List Nat := []
theorem innerScan191_27 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_27 best = best := rfl

def innerTail191_26 : List Nat := 8237 :: innerTail191_27
def innerCandidates191_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates191_26_checked : candidates edge 4 1181 8237 innerTail191_27 best.bound = innerCandidates191_26 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_26 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_26 best = best := by
  rw [innerTail191_26, scan, if_neg (by decide), innerCandidates191_26_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_27
    (if 3 ≤ innerCandidates191_26.length then search edge 3 [1181, 8237] innerCandidates191_26 9418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_27

def innerTail191_25 : List Nat := 8069 :: innerTail191_26
def innerCandidates191_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates191_25_checked : candidates edge 4 1181 8069 innerTail191_26 best.bound = innerCandidates191_25 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_25 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_25 best = best := by
  rw [innerTail191_25, scan, if_neg (by decide), innerCandidates191_25_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_26
    (if 3 ≤ innerCandidates191_25.length then search edge 3 [1181, 8069] innerCandidates191_25 9250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_26

def innerTail191_24 : List Nat := 7577 :: innerTail191_25
def innerCandidates191_24 : List Nat := [8069]
set_option maxRecDepth 4096 in
theorem innerCandidates191_24_checked : candidates edge 4 1181 7577 innerTail191_25 best.bound = innerCandidates191_24 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_24 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_24 best = best := by
  rw [innerTail191_24, scan, if_neg (by decide), innerCandidates191_24_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_25
    (if 3 ≤ innerCandidates191_24.length then search edge 3 [1181, 7577] innerCandidates191_24 8758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_25

def innerTail191_23 : List Nat := 7433 :: innerTail191_24
def innerCandidates191_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates191_23_checked : candidates edge 4 1181 7433 innerTail191_24 best.bound = innerCandidates191_23 := by
  unfold candidates
  have row : edge 7433 = SearchData.row7433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_23 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_23 best = best := by
  rw [innerTail191_23, scan, if_neg (by decide), innerCandidates191_23_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_24
    (if 3 ≤ innerCandidates191_23.length then search edge 3 [1181, 7433] innerCandidates191_23 8614 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_24

def innerTail191_22 : List Nat := 6977 :: innerTail191_23
def innerCandidates191_22 : List Nat := [7577]
set_option maxRecDepth 4096 in
theorem innerCandidates191_22_checked : candidates edge 4 1181 6977 innerTail191_23 best.bound = innerCandidates191_22 := by
  unfold candidates
  have row : edge 6977 = SearchData.row6977.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_22 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_22 best = best := by
  rw [innerTail191_22, scan, if_neg (by decide), innerCandidates191_22_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_23
    (if 3 ≤ innerCandidates191_22.length then search edge 3 [1181, 6977] innerCandidates191_22 8158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_23

def innerTail191_21 : List Nat := 6911 :: innerTail191_22
def innerCandidates191_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates191_21_checked : candidates edge 4 1181 6911 innerTail191_22 best.bound = innerCandidates191_21 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_21 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_21 best = best := by
  rw [innerTail191_21, scan, if_neg (by decide), innerCandidates191_21_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_22
    (if 3 ≤ innerCandidates191_21.length then search edge 3 [1181, 6911] innerCandidates191_21 8092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_22

def innerTail191_20 : List Nat := 6719 :: innerTail191_21
def innerCandidates191_20 : List Nat := [8237]
set_option maxRecDepth 4096 in
theorem innerCandidates191_20_checked : candidates edge 4 1181 6719 innerTail191_21 best.bound = innerCandidates191_20 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_20 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_20 best = best := by
  rw [innerTail191_20, scan, if_neg (by decide), innerCandidates191_20_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_21
    (if 3 ≤ innerCandidates191_20.length then search edge 3 [1181, 6719] innerCandidates191_20 7900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_21

def innerTail191_19 : List Nat := 6449 :: innerTail191_20
def innerCandidates191_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates191_19_checked : candidates edge 4 1181 6449 innerTail191_20 best.bound = innerCandidates191_19 := by
  unfold candidates
  have row : edge 6449 = SearchData.row6449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_19 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_19 best = best := by
  rw [innerTail191_19, scan, if_neg (by decide), innerCandidates191_19_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_20
    (if 3 ≤ innerCandidates191_19.length then search edge 3 [1181, 6449] innerCandidates191_19 7630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_20

def innerTail191_18 : List Nat := 5717 :: innerTail191_19
def innerCandidates191_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates191_18_checked : candidates edge 4 1181 5717 innerTail191_19 best.bound = innerCandidates191_18 := by
  unfold candidates
  have row : edge 5717 = SearchData.row5717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_18 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_18 best = best := by
  rw [innerTail191_18, scan, if_neg (by decide), innerCandidates191_18_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_19
    (if 3 ≤ innerCandidates191_18.length then search edge 3 [1181, 5717] innerCandidates191_18 6898 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_19

def innerTail191_17 : List Nat := 5501 :: innerTail191_18
def innerCandidates191_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates191_17_checked : candidates edge 4 1181 5501 innerTail191_18 best.bound = innerCandidates191_17 := by
  unfold candidates
  have row : edge 5501 = SearchData.row5501.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_17 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_17 best = best := by
  rw [innerTail191_17, scan, if_neg (by decide), innerCandidates191_17_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_18
    (if 3 ≤ innerCandidates191_17.length then search edge 3 [1181, 5501] innerCandidates191_17 6682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_18

def innerTail191_16 : List Nat := 5171 :: innerTail191_17
def innerCandidates191_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates191_16_checked : candidates edge 4 1181 5171 innerTail191_17 best.bound = innerCandidates191_16 := by
  unfold candidates
  have row : edge 5171 = SearchData.row5171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_16 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_16 best = best := by
  rw [innerTail191_16, scan, if_neg (by decide), innerCandidates191_16_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_17
    (if 3 ≤ innerCandidates191_16.length then search edge 3 [1181, 5171] innerCandidates191_16 6352 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_17

def innerTail191_15 : List Nat := 4931 :: innerTail191_16
def innerCandidates191_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates191_15_checked : candidates edge 4 1181 4931 innerTail191_16 best.bound = innerCandidates191_15 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_15 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_15 best = best := by
  rw [innerTail191_15, scan, if_neg (by decide), innerCandidates191_15_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_16
    (if 3 ≤ innerCandidates191_15.length then search edge 3 [1181, 4931] innerCandidates191_15 6112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_16

def innerTail191_14 : List Nat := 4889 :: innerTail191_15
def innerCandidates191_14 : List Nat := [8237]
set_option maxRecDepth 4096 in
theorem innerCandidates191_14_checked : candidates edge 4 1181 4889 innerTail191_15 best.bound = innerCandidates191_14 := by
  unfold candidates
  have row : edge 4889 = SearchData.row4889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_14 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_14 best = best := by
  rw [innerTail191_14, scan, if_neg (by decide), innerCandidates191_14_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_15
    (if 3 ≤ innerCandidates191_14.length then search edge 3 [1181, 4889] innerCandidates191_14 6070 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_15

def innerTail191_13 : List Nat := 4013 :: innerTail191_14
def innerCandidates191_13 : List Nat := [8069]
set_option maxRecDepth 4096 in
theorem innerCandidates191_13_checked : candidates edge 4 1181 4013 innerTail191_14 best.bound = innerCandidates191_13 := by
  unfold candidates
  have row : edge 4013 = SearchData.row4013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_13 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_13 best = best := by
  rw [innerTail191_13, scan, if_neg (by decide), innerCandidates191_13_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_14
    (if 3 ≤ innerCandidates191_13.length then search edge 3 [1181, 4013] innerCandidates191_13 5194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_14

def innerTail191_12 : List Nat := 3797 :: innerTail191_13
def innerCandidates191_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates191_12_checked : candidates edge 4 1181 3797 innerTail191_13 best.bound = innerCandidates191_12 := by
  unfold candidates
  have row : edge 3797 = SearchData.row3797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_12 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_12 best = best := by
  rw [innerTail191_12, scan, if_neg (by decide), innerCandidates191_12_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_13
    (if 3 ≤ innerCandidates191_12.length then search edge 3 [1181, 3797] innerCandidates191_12 4978 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_13

def innerTail191_11 : List Nat := 3251 :: innerTail191_12
def innerCandidates191_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates191_11_checked : candidates edge 4 1181 3251 innerTail191_12 best.bound = innerCandidates191_11 := by
  unfold candidates
  have row : edge 3251 = SearchData.row3251.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_11 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_11 best = best := by
  rw [innerTail191_11, scan, if_neg (by decide), innerCandidates191_11_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_12
    (if 3 ≤ innerCandidates191_11.length then search edge 3 [1181, 3251] innerCandidates191_11 4432 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_12

def innerTail191_10 : List Nat := 3209 :: innerTail191_11
def innerCandidates191_10 : List Nat := [5171, 5717, 7433]
set_option maxRecDepth 4096 in
theorem innerCandidates191_10_checked : candidates edge 4 1181 3209 innerTail191_11 best.bound = innerCandidates191_10 := by
  unfold candidates
  have row : edge 3209 = SearchData.row3209.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild191_10 : search edge 3 [1181, 3209] innerCandidates191_10 4390 best = best := by decide

theorem innerScan191_10 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_10 best = best := by
  rw [innerTail191_10, scan, if_neg (by decide), innerCandidates191_10_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_11
    (if 3 ≤ innerCandidates191_10.length then search edge 3 [1181, 3209] innerCandidates191_10 4390 best else best) = best
  rw [if_pos (by decide), innerChild191_10]
  exact innerScan191_11

def innerTail191_9 : List Nat := 3083 :: innerTail191_10
def innerCandidates191_9 : List Nat := [3797, 4931]
set_option maxRecDepth 4096 in
theorem innerCandidates191_9_checked : candidates edge 4 1181 3083 innerTail191_10 best.bound = innerCandidates191_9 := by
  unfold candidates
  have row : edge 3083 = SearchData.row3083.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_9 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_9 best = best := by
  rw [innerTail191_9, scan, if_neg (by decide), innerCandidates191_9_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_10
    (if 3 ≤ innerCandidates191_9.length then search edge 3 [1181, 3083] innerCandidates191_9 4264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_10

def innerTail191_8 : List Nat := 2729 :: innerTail191_9
def innerCandidates191_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates191_8_checked : candidates edge 4 1181 2729 innerTail191_9 best.bound = innerCandidates191_8 := by
  unfold candidates
  have row : edge 2729 = SearchData.row2729.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_8 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_8 best = best := by
  rw [innerTail191_8, scan, if_neg (by decide), innerCandidates191_8_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_9
    (if 3 ≤ innerCandidates191_8.length then search edge 3 [1181, 2729] innerCandidates191_8 3910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_9

def innerTail191_7 : List Nat := 2687 :: innerTail191_8
def innerCandidates191_7 : List Nat := [8069]
set_option maxRecDepth 4096 in
theorem innerCandidates191_7_checked : candidates edge 4 1181 2687 innerTail191_8 best.bound = innerCandidates191_7 := by
  unfold candidates
  have row : edge 2687 = SearchData.row2687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_7 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_7 best = best := by
  rw [innerTail191_7, scan, if_neg (by decide), innerCandidates191_7_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_8
    (if 3 ≤ innerCandidates191_7.length then search edge 3 [1181, 2687] innerCandidates191_7 3868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_8

def innerTail191_6 : List Nat := 2621 :: innerTail191_7
def innerCandidates191_6 : List Nat := [6977]
set_option maxRecDepth 4096 in
theorem innerCandidates191_6_checked : candidates edge 4 1181 2621 innerTail191_7 best.bound = innerCandidates191_6 := by
  unfold candidates
  have row : edge 2621 = PrimePairCache.smallRow2621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_6 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_6 best = best := by
  rw [innerTail191_6, scan, if_neg (by decide), innerCandidates191_6_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_7
    (if 3 ≤ innerCandidates191_6.length then search edge 3 [1181, 2621] innerCandidates191_6 3802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_7

def innerTail191_5 : List Nat := 1847 :: innerTail191_6
def innerCandidates191_5 : List Nat := [2687, 8069]
set_option maxRecDepth 4096 in
theorem innerCandidates191_5_checked : candidates edge 4 1181 1847 innerTail191_6 best.bound = innerCandidates191_5 := by
  unfold candidates
  have row : edge 1847 = PrimePairCache.smallRow1847.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_5 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_5 best = best := by
  rw [innerTail191_5, scan, if_neg (by decide), innerCandidates191_5_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_6
    (if 3 ≤ innerCandidates191_5.length then search edge 3 [1181, 1847] innerCandidates191_5 3028 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_6

def innerTail191_4 : List Nat := 1601 :: innerTail191_5
def innerCandidates191_4 : List Nat := [1847]
set_option maxRecDepth 4096 in
theorem innerCandidates191_4_checked : candidates edge 4 1181 1601 innerTail191_5 best.bound = innerCandidates191_4 := by
  unfold candidates
  have row : edge 1601 = PrimePairCache.smallRow1601.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_4 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_4 best = best := by
  rw [innerTail191_4, scan, if_neg (by decide), innerCandidates191_4_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_5
    (if 3 ≤ innerCandidates191_4.length then search edge 3 [1181, 1601] innerCandidates191_4 2782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_5

def innerTail191_3 : List Nat := 1451 :: innerTail191_4
def innerCandidates191_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates191_3_checked : candidates edge 4 1181 1451 innerTail191_4 best.bound = innerCandidates191_3 := by
  unfold candidates
  have row : edge 1451 = PrimePairCache.smallRow1451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_3 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_3 best = best := by
  rw [innerTail191_3, scan, if_neg (by decide), innerCandidates191_3_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_4
    (if 3 ≤ innerCandidates191_3.length then search edge 3 [1181, 1451] innerCandidates191_3 2632 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_4

def innerTail191_2 : List Nat := 1427 :: innerTail191_3
def innerCandidates191_2 : List Nat := [5501]
set_option maxRecDepth 4096 in
theorem innerCandidates191_2_checked : candidates edge 4 1181 1427 innerTail191_3 best.bound = innerCandidates191_2 := by
  unfold candidates
  have row : edge 1427 = PrimePairCache.smallRow1427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_2 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_2 best = best := by
  rw [innerTail191_2, scan, if_neg (by decide), innerCandidates191_2_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_3
    (if 3 ≤ innerCandidates191_2.length then search edge 3 [1181, 1427] innerCandidates191_2 2608 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_3

def innerTail191_1 : List Nat := 1367 :: innerTail191_2
def innerCandidates191_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates191_1_checked : candidates edge 4 1181 1367 innerTail191_2 best.bound = innerCandidates191_1 := by
  unfold candidates
  have row : edge 1367 = PrimePairCache.smallRow1367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_1 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_1 best = best := by
  rw [innerTail191_1, scan, if_neg (by decide), innerCandidates191_1_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_2
    (if 3 ≤ innerCandidates191_1.length then search edge 3 [1181, 1367] innerCandidates191_1 2548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_2

def innerTail191_0 : List Nat := 1361 :: innerTail191_1
def innerCandidates191_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates191_0_checked : candidates edge 4 1181 1361 innerTail191_1 best.bound = innerCandidates191_0 := by
  unfold candidates
  have row : edge 1361 = PrimePairCache.smallRow1361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan191_0 : scan edge (search edge 3) 4 [1181] 1181 innerTail191_0 best = best := by
  rw [innerTail191_0, scan, if_neg (by decide), innerCandidates191_0_checked]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_1
    (if 3 ≤ innerCandidates191_0.length then search edge 3 [1181, 1361] innerCandidates191_0 2542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan191_1

theorem rootBranch191 : search edge 4 [1181] rootCandidates191 1181 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1181] 1181 innerTail191_0 best = best
  exact innerScan191_0

def rootCandidates190 : List Nat := [1303, 1699, 2011, 2053, 2137, 2161, 3607, 4663, 4933, 4951, 5881, 6007, 6337, 7873, 7963]
set_option maxRecDepth 4096 in
theorem rootCandidates190_checked : candidates edge 5 0 1171 tail191 best.bound = rootCandidates190 := by
  unfold candidates
  have row : edge 1171 = PrimePairCache.smallRow1171.contains := by funext q; rfl
  rw [row]
  decide

def innerTail190_15 : List Nat := []
theorem innerScan190_15 : scan edge (search edge 3) 4 [1171] 1171 innerTail190_15 best = best := rfl

def innerTail190_14 : List Nat := 7963 :: innerTail190_15
def innerCandidates190_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates190_14_checked : candidates edge 4 1171 7963 innerTail190_15 best.bound = innerCandidates190_14 := by
  unfold candidates
  have row : edge 7963 = SearchData.row7963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan190_14 : scan edge (search edge 3) 4 [1171] 1171 innerTail190_14 best = best := by
  rw [innerTail190_14, scan, if_neg (by decide), innerCandidates190_14_checked]
  change scan edge (search edge 3) 4 [1171] 1171 innerTail190_15
    (if 3 ≤ innerCandidates190_14.length then search edge 3 [1171, 7963] innerCandidates190_14 9134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan190_15

def innerTail190_13 : List Nat := 7873 :: innerTail190_14
def innerCandidates190_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates190_13_checked : candidates edge 4 1171 7873 innerTail190_14 best.bound = innerCandidates190_13 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan190_13 : scan edge (search edge 3) 4 [1171] 1171 innerTail190_13 best = best := by
  rw [innerTail190_13, scan, if_neg (by decide), innerCandidates190_13_checked]
  change scan edge (search edge 3) 4 [1171] 1171 innerTail190_14
    (if 3 ≤ innerCandidates190_13.length then search edge 3 [1171, 7873] innerCandidates190_13 9044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan190_14

def innerTail190_12 : List Nat := 6337 :: innerTail190_13
def innerCandidates190_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates190_12_checked : candidates edge 4 1171 6337 innerTail190_13 best.bound = innerCandidates190_12 := by
  unfold candidates
  have row : edge 6337 = SearchData.row6337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan190_12 : scan edge (search edge 3) 4 [1171] 1171 innerTail190_12 best = best := by
  rw [innerTail190_12, scan, if_neg (by decide), innerCandidates190_12_checked]
  change scan edge (search edge 3) 4 [1171] 1171 innerTail190_13
    (if 3 ≤ innerCandidates190_12.length then search edge 3 [1171, 6337] innerCandidates190_12 7508 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan190_13

def innerTail190_11 : List Nat := 6007 :: innerTail190_12
def innerCandidates190_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates190_11_checked : candidates edge 4 1171 6007 innerTail190_12 best.bound = innerCandidates190_11 := by
  unfold candidates
  have row : edge 6007 = SearchData.row6007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan190_11 : scan edge (search edge 3) 4 [1171] 1171 innerTail190_11 best = best := by
  rw [innerTail190_11, scan, if_neg (by decide), innerCandidates190_11_checked]
  change scan edge (search edge 3) 4 [1171] 1171 innerTail190_12
    (if 3 ≤ innerCandidates190_11.length then search edge 3 [1171, 6007] innerCandidates190_11 7178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan190_12

def innerTail190_10 : List Nat := 5881 :: innerTail190_11
def innerCandidates190_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates190_10_checked : candidates edge 4 1171 5881 innerTail190_11 best.bound = innerCandidates190_10 := by
  unfold candidates
  have row : edge 5881 = SearchData.row5881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan190_10 : scan edge (search edge 3) 4 [1171] 1171 innerTail190_10 best = best := by
  rw [innerTail190_10, scan, if_neg (by decide), innerCandidates190_10_checked]
  change scan edge (search edge 3) 4 [1171] 1171 innerTail190_11
    (if 3 ≤ innerCandidates190_10.length then search edge 3 [1171, 5881] innerCandidates190_10 7052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan190_11

def innerTail190_9 : List Nat := 4951 :: innerTail190_10
def innerCandidates190_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates190_9_checked : candidates edge 4 1171 4951 innerTail190_10 best.bound = innerCandidates190_9 := by
  unfold candidates
  have row : edge 4951 = SearchData.row4951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan190_9 : scan edge (search edge 3) 4 [1171] 1171 innerTail190_9 best = best := by
  rw [innerTail190_9, scan, if_neg (by decide), innerCandidates190_9_checked]
  change scan edge (search edge 3) 4 [1171] 1171 innerTail190_10
    (if 3 ≤ innerCandidates190_9.length then search edge 3 [1171, 4951] innerCandidates190_9 6122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan190_10

def innerTail190_8 : List Nat := 4933 :: innerTail190_9
def innerCandidates190_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates190_8_checked : candidates edge 4 1171 4933 innerTail190_9 best.bound = innerCandidates190_8 := by
  unfold candidates
  have row : edge 4933 = SearchData.row4933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan190_8 : scan edge (search edge 3) 4 [1171] 1171 innerTail190_8 best = best := by
  rw [innerTail190_8, scan, if_neg (by decide), innerCandidates190_8_checked]
  change scan edge (search edge 3) 4 [1171] 1171 innerTail190_9
    (if 3 ≤ innerCandidates190_8.length then search edge 3 [1171, 4933] innerCandidates190_8 6104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan190_9

def innerTail190_7 : List Nat := 4663 :: innerTail190_8
def innerCandidates190_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates190_7_checked : candidates edge 4 1171 4663 innerTail190_8 best.bound = innerCandidates190_7 := by
  unfold candidates
  have row : edge 4663 = SearchData.row4663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan190_7 : scan edge (search edge 3) 4 [1171] 1171 innerTail190_7 best = best := by
  rw [innerTail190_7, scan, if_neg (by decide), innerCandidates190_7_checked]
  change scan edge (search edge 3) 4 [1171] 1171 innerTail190_8
    (if 3 ≤ innerCandidates190_7.length then search edge 3 [1171, 4663] innerCandidates190_7 5834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan190_8

def innerTail190_6 : List Nat := 3607 :: innerTail190_7
def innerCandidates190_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates190_6_checked : candidates edge 4 1171 3607 innerTail190_7 best.bound = innerCandidates190_6 := by
  unfold candidates
  have row : edge 3607 = SearchData.row3607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan190_6 : scan edge (search edge 3) 4 [1171] 1171 innerTail190_6 best = best := by
  rw [innerTail190_6, scan, if_neg (by decide), innerCandidates190_6_checked]
  change scan edge (search edge 3) 4 [1171] 1171 innerTail190_7
    (if 3 ≤ innerCandidates190_6.length then search edge 3 [1171, 3607] innerCandidates190_6 4778 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan190_7

def innerTail190_5 : List Nat := 2161 :: innerTail190_6
def innerCandidates190_5 : List Nat := [6337]
set_option maxRecDepth 4096 in
theorem innerCandidates190_5_checked : candidates edge 4 1171 2161 innerTail190_6 best.bound = innerCandidates190_5 := by
  unfold candidates
  have row : edge 2161 = PrimePairCache.smallRow2161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan190_5 : scan edge (search edge 3) 4 [1171] 1171 innerTail190_5 best = best := by
  rw [innerTail190_5, scan, if_neg (by decide), innerCandidates190_5_checked]
  change scan edge (search edge 3) 4 [1171] 1171 innerTail190_6
    (if 3 ≤ innerCandidates190_5.length then search edge 3 [1171, 2161] innerCandidates190_5 3332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan190_6

def innerTail190_4 : List Nat := 2137 :: innerTail190_5
def innerCandidates190_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates190_4_checked : candidates edge 4 1171 2137 innerTail190_5 best.bound = innerCandidates190_4 := by
  unfold candidates
  have row : edge 2137 = PrimePairCache.smallRow2137.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan190_4 : scan edge (search edge 3) 4 [1171] 1171 innerTail190_4 best = best := by
  rw [innerTail190_4, scan, if_neg (by decide), innerCandidates190_4_checked]
  change scan edge (search edge 3) 4 [1171] 1171 innerTail190_5
    (if 3 ≤ innerCandidates190_4.length then search edge 3 [1171, 2137] innerCandidates190_4 3308 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan190_5

def innerTail190_3 : List Nat := 2053 :: innerTail190_4
def innerCandidates190_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates190_3_checked : candidates edge 4 1171 2053 innerTail190_4 best.bound = innerCandidates190_3 := by
  unfold candidates
  have row : edge 2053 = PrimePairCache.smallRow2053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan190_3 : scan edge (search edge 3) 4 [1171] 1171 innerTail190_3 best = best := by
  rw [innerTail190_3, scan, if_neg (by decide), innerCandidates190_3_checked]
  change scan edge (search edge 3) 4 [1171] 1171 innerTail190_4
    (if 3 ≤ innerCandidates190_3.length then search edge 3 [1171, 2053] innerCandidates190_3 3224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan190_4

def innerTail190_2 : List Nat := 2011 :: innerTail190_3
def innerCandidates190_2 : List Nat := [7963]
set_option maxRecDepth 4096 in
theorem innerCandidates190_2_checked : candidates edge 4 1171 2011 innerTail190_3 best.bound = innerCandidates190_2 := by
  unfold candidates
  have row : edge 2011 = PrimePairCache.smallRow2011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan190_2 : scan edge (search edge 3) 4 [1171] 1171 innerTail190_2 best = best := by
  rw [innerTail190_2, scan, if_neg (by decide), innerCandidates190_2_checked]
  change scan edge (search edge 3) 4 [1171] 1171 innerTail190_3
    (if 3 ≤ innerCandidates190_2.length then search edge 3 [1171, 2011] innerCandidates190_2 3182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan190_3

def innerTail190_1 : List Nat := 1699 :: innerTail190_2
def innerCandidates190_1 : List Nat := [3607]
set_option maxRecDepth 4096 in
theorem innerCandidates190_1_checked : candidates edge 4 1171 1699 innerTail190_2 best.bound = innerCandidates190_1 := by
  unfold candidates
  have row : edge 1699 = PrimePairCache.smallRow1699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan190_1 : scan edge (search edge 3) 4 [1171] 1171 innerTail190_1 best = best := by
  rw [innerTail190_1, scan, if_neg (by decide), innerCandidates190_1_checked]
  change scan edge (search edge 3) 4 [1171] 1171 innerTail190_2
    (if 3 ≤ innerCandidates190_1.length then search edge 3 [1171, 1699] innerCandidates190_1 2870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan190_2

def innerTail190_0 : List Nat := 1303 :: innerTail190_1
def innerCandidates190_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates190_0_checked : candidates edge 4 1171 1303 innerTail190_1 best.bound = innerCandidates190_0 := by
  unfold candidates
  have row : edge 1303 = PrimePairCache.smallRow1303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan190_0 : scan edge (search edge 3) 4 [1171] 1171 innerTail190_0 best = best := by
  rw [innerTail190_0, scan, if_neg (by decide), innerCandidates190_0_checked]
  change scan edge (search edge 3) 4 [1171] 1171 innerTail190_1
    (if 3 ≤ innerCandidates190_0.length then search edge 3 [1171, 1303] innerCandidates190_0 2474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan190_1

theorem rootBranch190 : search edge 4 [1171] rootCandidates190 1171 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1171] 1171 innerTail190_0 best = best
  exact innerScan190_0

def rootCandidates189 : List Nat := [1733, 1877, 2111, 2417, 2447, 2927, 4547, 4673, 4967, 5639, 5711, 5849, 6197, 6803, 6899, 7211, 7349, 7853, 7919, 8243]
set_option maxRecDepth 4096 in
theorem rootCandidates189_checked : candidates edge 5 0 1163 tail190 best.bound = rootCandidates189 := by
  unfold candidates
  have row : edge 1163 = PrimePairCache.smallRow1163.contains := by funext q; rfl
  rw [row]
  decide

def innerTail189_20 : List Nat := []
theorem innerScan189_20 : scan edge (search edge 3) 4 [1163] 1163 innerTail189_20 best = best := rfl

def innerTail189_19 : List Nat := 8243 :: innerTail189_20
def innerCandidates189_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates189_19_checked : candidates edge 4 1163 8243 innerTail189_20 best.bound = innerCandidates189_19 := by
  unfold candidates
  have row : edge 8243 = SearchData.row8243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan189_19 : scan edge (search edge 3) 4 [1163] 1163 innerTail189_19 best = best := by
  rw [innerTail189_19, scan, if_neg (by decide), innerCandidates189_19_checked]
  change scan edge (search edge 3) 4 [1163] 1163 innerTail189_20
    (if 3 ≤ innerCandidates189_19.length then search edge 3 [1163, 8243] innerCandidates189_19 9406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan189_20

def innerTail189_18 : List Nat := 7919 :: innerTail189_19
def innerCandidates189_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates189_18_checked : candidates edge 4 1163 7919 innerTail189_19 best.bound = innerCandidates189_18 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan189_18 : scan edge (search edge 3) 4 [1163] 1163 innerTail189_18 best = best := by
  rw [innerTail189_18, scan, if_neg (by decide), innerCandidates189_18_checked]
  change scan edge (search edge 3) 4 [1163] 1163 innerTail189_19
    (if 3 ≤ innerCandidates189_18.length then search edge 3 [1163, 7919] innerCandidates189_18 9082 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan189_19

def innerTail189_17 : List Nat := 7853 :: innerTail189_18
def innerCandidates189_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates189_17_checked : candidates edge 4 1163 7853 innerTail189_18 best.bound = innerCandidates189_17 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan189_17 : scan edge (search edge 3) 4 [1163] 1163 innerTail189_17 best = best := by
  rw [innerTail189_17, scan, if_neg (by decide), innerCandidates189_17_checked]
  change scan edge (search edge 3) 4 [1163] 1163 innerTail189_18
    (if 3 ≤ innerCandidates189_17.length then search edge 3 [1163, 7853] innerCandidates189_17 9016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan189_18

def innerTail189_16 : List Nat := 7349 :: innerTail189_17
def innerCandidates189_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates189_16_checked : candidates edge 4 1163 7349 innerTail189_17 best.bound = innerCandidates189_16 := by
  unfold candidates
  have row : edge 7349 = SearchData.row7349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan189_16 : scan edge (search edge 3) 4 [1163] 1163 innerTail189_16 best = best := by
  rw [innerTail189_16, scan, if_neg (by decide), innerCandidates189_16_checked]
  change scan edge (search edge 3) 4 [1163] 1163 innerTail189_17
    (if 3 ≤ innerCandidates189_16.length then search edge 3 [1163, 7349] innerCandidates189_16 8512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan189_17

def innerTail189_15 : List Nat := 7211 :: innerTail189_16
def innerCandidates189_15 : List Nat := [7853]
set_option maxRecDepth 4096 in
theorem innerCandidates189_15_checked : candidates edge 4 1163 7211 innerTail189_16 best.bound = innerCandidates189_15 := by
  unfold candidates
  have row : edge 7211 = SearchData.row7211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan189_15 : scan edge (search edge 3) 4 [1163] 1163 innerTail189_15 best = best := by
  rw [innerTail189_15, scan, if_neg (by decide), innerCandidates189_15_checked]
  change scan edge (search edge 3) 4 [1163] 1163 innerTail189_16
    (if 3 ≤ innerCandidates189_15.length then search edge 3 [1163, 7211] innerCandidates189_15 8374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan189_16

def innerTail189_14 : List Nat := 6899 :: innerTail189_15
def innerCandidates189_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates189_14_checked : candidates edge 4 1163 6899 innerTail189_15 best.bound = innerCandidates189_14 := by
  unfold candidates
  have row : edge 6899 = SearchData.row6899.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan189_14 : scan edge (search edge 3) 4 [1163] 1163 innerTail189_14 best = best := by
  rw [innerTail189_14, scan, if_neg (by decide), innerCandidates189_14_checked]
  change scan edge (search edge 3) 4 [1163] 1163 innerTail189_15
    (if 3 ≤ innerCandidates189_14.length then search edge 3 [1163, 6899] innerCandidates189_14 8062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan189_15

def innerTail189_13 : List Nat := 6803 :: innerTail189_14
def innerCandidates189_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates189_13_checked : candidates edge 4 1163 6803 innerTail189_14 best.bound = innerCandidates189_13 := by
  unfold candidates
  have row : edge 6803 = SearchData.row6803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan189_13 : scan edge (search edge 3) 4 [1163] 1163 innerTail189_13 best = best := by
  rw [innerTail189_13, scan, if_neg (by decide), innerCandidates189_13_checked]
  change scan edge (search edge 3) 4 [1163] 1163 innerTail189_14
    (if 3 ≤ innerCandidates189_13.length then search edge 3 [1163, 6803] innerCandidates189_13 7966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan189_14

def innerTail189_12 : List Nat := 6197 :: innerTail189_13
def innerCandidates189_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates189_12_checked : candidates edge 4 1163 6197 innerTail189_13 best.bound = innerCandidates189_12 := by
  unfold candidates
  have row : edge 6197 = SearchData.row6197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan189_12 : scan edge (search edge 3) 4 [1163] 1163 innerTail189_12 best = best := by
  rw [innerTail189_12, scan, if_neg (by decide), innerCandidates189_12_checked]
  change scan edge (search edge 3) 4 [1163] 1163 innerTail189_13
    (if 3 ≤ innerCandidates189_12.length then search edge 3 [1163, 6197] innerCandidates189_12 7360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan189_13

def innerTail189_11 : List Nat := 5849 :: innerTail189_12
def innerCandidates189_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates189_11_checked : candidates edge 4 1163 5849 innerTail189_12 best.bound = innerCandidates189_11 := by
  unfold candidates
  have row : edge 5849 = SearchData.row5849.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan189_11 : scan edge (search edge 3) 4 [1163] 1163 innerTail189_11 best = best := by
  rw [innerTail189_11, scan, if_neg (by decide), innerCandidates189_11_checked]
  change scan edge (search edge 3) 4 [1163] 1163 innerTail189_12
    (if 3 ≤ innerCandidates189_11.length then search edge 3 [1163, 5849] innerCandidates189_11 7012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan189_12

def innerTail189_10 : List Nat := 5711 :: innerTail189_11
def innerCandidates189_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates189_10_checked : candidates edge 4 1163 5711 innerTail189_11 best.bound = innerCandidates189_10 := by
  unfold candidates
  have row : edge 5711 = SearchData.row5711.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan189_10 : scan edge (search edge 3) 4 [1163] 1163 innerTail189_10 best = best := by
  rw [innerTail189_10, scan, if_neg (by decide), innerCandidates189_10_checked]
  change scan edge (search edge 3) 4 [1163] 1163 innerTail189_11
    (if 3 ≤ innerCandidates189_10.length then search edge 3 [1163, 5711] innerCandidates189_10 6874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan189_11

def innerTail189_9 : List Nat := 5639 :: innerTail189_10
def innerCandidates189_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates189_9_checked : candidates edge 4 1163 5639 innerTail189_10 best.bound = innerCandidates189_9 := by
  unfold candidates
  have row : edge 5639 = SearchData.row5639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan189_9 : scan edge (search edge 3) 4 [1163] 1163 innerTail189_9 best = best := by
  rw [innerTail189_9, scan, if_neg (by decide), innerCandidates189_9_checked]
  change scan edge (search edge 3) 4 [1163] 1163 innerTail189_10
    (if 3 ≤ innerCandidates189_9.length then search edge 3 [1163, 5639] innerCandidates189_9 6802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan189_10

def innerTail189_8 : List Nat := 4967 :: innerTail189_9
def innerCandidates189_8 : List Nat := [5849]
set_option maxRecDepth 4096 in
theorem innerCandidates189_8_checked : candidates edge 4 1163 4967 innerTail189_9 best.bound = innerCandidates189_8 := by
  unfold candidates
  have row : edge 4967 = SearchData.row4967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan189_8 : scan edge (search edge 3) 4 [1163] 1163 innerTail189_8 best = best := by
  rw [innerTail189_8, scan, if_neg (by decide), innerCandidates189_8_checked]
  change scan edge (search edge 3) 4 [1163] 1163 innerTail189_9
    (if 3 ≤ innerCandidates189_8.length then search edge 3 [1163, 4967] innerCandidates189_8 6130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan189_9

def innerTail189_7 : List Nat := 4673 :: innerTail189_8
def innerCandidates189_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates189_7_checked : candidates edge 4 1163 4673 innerTail189_8 best.bound = innerCandidates189_7 := by
  unfold candidates
  have row : edge 4673 = SearchData.row4673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan189_7 : scan edge (search edge 3) 4 [1163] 1163 innerTail189_7 best = best := by
  rw [innerTail189_7, scan, if_neg (by decide), innerCandidates189_7_checked]
  change scan edge (search edge 3) 4 [1163] 1163 innerTail189_8
    (if 3 ≤ innerCandidates189_7.length then search edge 3 [1163, 4673] innerCandidates189_7 5836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan189_8

def innerTail189_6 : List Nat := 4547 :: innerTail189_7
def innerCandidates189_6 : List Nat := [7919]
set_option maxRecDepth 4096 in
theorem innerCandidates189_6_checked : candidates edge 4 1163 4547 innerTail189_7 best.bound = innerCandidates189_6 := by
  unfold candidates
  have row : edge 4547 = SearchData.row4547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan189_6 : scan edge (search edge 3) 4 [1163] 1163 innerTail189_6 best = best := by
  rw [innerTail189_6, scan, if_neg (by decide), innerCandidates189_6_checked]
  change scan edge (search edge 3) 4 [1163] 1163 innerTail189_7
    (if 3 ≤ innerCandidates189_6.length then search edge 3 [1163, 4547] innerCandidates189_6 5710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan189_7

def innerTail189_5 : List Nat := 2927 :: innerTail189_6
def innerCandidates189_5 : List Nat := [8243]
set_option maxRecDepth 4096 in
theorem innerCandidates189_5_checked : candidates edge 4 1163 2927 innerTail189_6 best.bound = innerCandidates189_5 := by
  unfold candidates
  have row : edge 2927 = SearchData.row2927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan189_5 : scan edge (search edge 3) 4 [1163] 1163 innerTail189_5 best = best := by
  rw [innerTail189_5, scan, if_neg (by decide), innerCandidates189_5_checked]
  change scan edge (search edge 3) 4 [1163] 1163 innerTail189_6
    (if 3 ≤ innerCandidates189_5.length then search edge 3 [1163, 2927] innerCandidates189_5 4090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan189_6

def innerTail189_4 : List Nat := 2447 :: innerTail189_5
def innerCandidates189_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates189_4_checked : candidates edge 4 1163 2447 innerTail189_5 best.bound = innerCandidates189_4 := by
  unfold candidates
  have row : edge 2447 = PrimePairCache.smallRow2447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan189_4 : scan edge (search edge 3) 4 [1163] 1163 innerTail189_4 best = best := by
  rw [innerTail189_4, scan, if_neg (by decide), innerCandidates189_4_checked]
  change scan edge (search edge 3) 4 [1163] 1163 innerTail189_5
    (if 3 ≤ innerCandidates189_4.length then search edge 3 [1163, 2447] innerCandidates189_4 3610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan189_5

def innerTail189_3 : List Nat := 2417 :: innerTail189_4
def innerCandidates189_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates189_3_checked : candidates edge 4 1163 2417 innerTail189_4 best.bound = innerCandidates189_3 := by
  unfold candidates
  have row : edge 2417 = PrimePairCache.smallRow2417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan189_3 : scan edge (search edge 3) 4 [1163] 1163 innerTail189_3 best = best := by
  rw [innerTail189_3, scan, if_neg (by decide), innerCandidates189_3_checked]
  change scan edge (search edge 3) 4 [1163] 1163 innerTail189_4
    (if 3 ≤ innerCandidates189_3.length then search edge 3 [1163, 2417] innerCandidates189_3 3580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan189_4

def innerTail189_2 : List Nat := 2111 :: innerTail189_3
def innerCandidates189_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates189_2_checked : candidates edge 4 1163 2111 innerTail189_3 best.bound = innerCandidates189_2 := by
  unfold candidates
  have row : edge 2111 = PrimePairCache.smallRow2111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan189_2 : scan edge (search edge 3) 4 [1163] 1163 innerTail189_2 best = best := by
  rw [innerTail189_2, scan, if_neg (by decide), innerCandidates189_2_checked]
  change scan edge (search edge 3) 4 [1163] 1163 innerTail189_3
    (if 3 ≤ innerCandidates189_2.length then search edge 3 [1163, 2111] innerCandidates189_2 3274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan189_3

def innerTail189_1 : List Nat := 1877 :: innerTail189_2
def innerCandidates189_1 : List Nat := [2447, 5711]
set_option maxRecDepth 4096 in
theorem innerCandidates189_1_checked : candidates edge 4 1163 1877 innerTail189_2 best.bound = innerCandidates189_1 := by
  unfold candidates
  have row : edge 1877 = PrimePairCache.smallRow1877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan189_1 : scan edge (search edge 3) 4 [1163] 1163 innerTail189_1 best = best := by
  rw [innerTail189_1, scan, if_neg (by decide), innerCandidates189_1_checked]
  change scan edge (search edge 3) 4 [1163] 1163 innerTail189_2
    (if 3 ≤ innerCandidates189_1.length then search edge 3 [1163, 1877] innerCandidates189_1 3040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan189_2

def innerTail189_0 : List Nat := 1733 :: innerTail189_1
def innerCandidates189_0 : List Nat := [2927]
set_option maxRecDepth 4096 in
theorem innerCandidates189_0_checked : candidates edge 4 1163 1733 innerTail189_1 best.bound = innerCandidates189_0 := by
  unfold candidates
  have row : edge 1733 = PrimePairCache.smallRow1733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan189_0 : scan edge (search edge 3) 4 [1163] 1163 innerTail189_0 best = best := by
  rw [innerTail189_0, scan, if_neg (by decide), innerCandidates189_0_checked]
  change scan edge (search edge 3) 4 [1163] 1163 innerTail189_1
    (if 3 ≤ innerCandidates189_0.length then search edge 3 [1163, 1733] innerCandidates189_0 2896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan189_1

theorem rootBranch189 : search edge 4 [1163] rootCandidates189 1163 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1163] 1163 innerTail189_0 best = best
  exact innerScan189_0

end ProjectEulerSolutions.PrimePairUpper.Search
