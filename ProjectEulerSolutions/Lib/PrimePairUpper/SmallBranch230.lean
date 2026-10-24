import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates47 : List Nat := [373, 433, 499, 547, 613, 631, 937, 1231, 1801, 2083, 2221, 2239, 2347, 2767, 3121, 3301, 3391, 3541, 3559, 3919, 4021, 5077, 5569, 5659, 5869, 6247, 6733, 7213, 7591, 7759]
set_option maxRecDepth 4096 in
theorem rootCandidates47_checked : candidates edge 5 0 229 tail48 best.bound = rootCandidates47 := by
  unfold candidates
  have row : edge 229 = PrimePairCache.smallRow229.contains := by funext q; rfl
  rw [row]
  decide

def innerTail47_30 : List Nat := []
theorem innerScan47_30 : scan edge (search edge 3) 4 [229] 229 innerTail47_30 best = best := rfl

def innerTail47_29 : List Nat := 7759 :: innerTail47_30
def innerCandidates47_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates47_29_checked : candidates edge 4 229 7759 innerTail47_30 best.bound = innerCandidates47_29 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_29 : scan edge (search edge 3) 4 [229] 229 innerTail47_29 best = best := by
  rw [innerTail47_29, scan, if_neg (by decide), innerCandidates47_29_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_30
    (if 3 ≤ innerCandidates47_29.length then search edge 3 [229, 7759] innerCandidates47_29 7988 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_30

def innerTail47_28 : List Nat := 7591 :: innerTail47_29
def innerCandidates47_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates47_28_checked : candidates edge 4 229 7591 innerTail47_29 best.bound = innerCandidates47_28 := by
  unfold candidates
  have row : edge 7591 = SearchData.row7591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_28 : scan edge (search edge 3) 4 [229] 229 innerTail47_28 best = best := by
  rw [innerTail47_28, scan, if_neg (by decide), innerCandidates47_28_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_29
    (if 3 ≤ innerCandidates47_28.length then search edge 3 [229, 7591] innerCandidates47_28 7820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_29

def innerTail47_27 : List Nat := 7213 :: innerTail47_28
def innerCandidates47_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates47_27_checked : candidates edge 4 229 7213 innerTail47_28 best.bound = innerCandidates47_27 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_27 : scan edge (search edge 3) 4 [229] 229 innerTail47_27 best = best := by
  rw [innerTail47_27, scan, if_neg (by decide), innerCandidates47_27_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_28
    (if 3 ≤ innerCandidates47_27.length then search edge 3 [229, 7213] innerCandidates47_27 7442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_28

def innerTail47_26 : List Nat := 6733 :: innerTail47_27
def innerCandidates47_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates47_26_checked : candidates edge 4 229 6733 innerTail47_27 best.bound = innerCandidates47_26 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_26 : scan edge (search edge 3) 4 [229] 229 innerTail47_26 best = best := by
  rw [innerTail47_26, scan, if_neg (by decide), innerCandidates47_26_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_27
    (if 3 ≤ innerCandidates47_26.length then search edge 3 [229, 6733] innerCandidates47_26 6962 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_27

def innerTail47_25 : List Nat := 6247 :: innerTail47_26
def innerCandidates47_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates47_25_checked : candidates edge 4 229 6247 innerTail47_26 best.bound = innerCandidates47_25 := by
  unfold candidates
  have row : edge 6247 = SearchData.row6247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_25 : scan edge (search edge 3) 4 [229] 229 innerTail47_25 best = best := by
  rw [innerTail47_25, scan, if_neg (by decide), innerCandidates47_25_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_26
    (if 3 ≤ innerCandidates47_25.length then search edge 3 [229, 6247] innerCandidates47_25 6476 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_26

def innerTail47_24 : List Nat := 5869 :: innerTail47_25
def innerCandidates47_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates47_24_checked : candidates edge 4 229 5869 innerTail47_25 best.bound = innerCandidates47_24 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_24 : scan edge (search edge 3) 4 [229] 229 innerTail47_24 best = best := by
  rw [innerTail47_24, scan, if_neg (by decide), innerCandidates47_24_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_25
    (if 3 ≤ innerCandidates47_24.length then search edge 3 [229, 5869] innerCandidates47_24 6098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_25

def innerTail47_23 : List Nat := 5659 :: innerTail47_24
def innerCandidates47_23 : List Nat := [7759]
set_option maxRecDepth 4096 in
theorem innerCandidates47_23_checked : candidates edge 4 229 5659 innerTail47_24 best.bound = innerCandidates47_23 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_23 : scan edge (search edge 3) 4 [229] 229 innerTail47_23 best = best := by
  rw [innerTail47_23, scan, if_neg (by decide), innerCandidates47_23_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_24
    (if 3 ≤ innerCandidates47_23.length then search edge 3 [229, 5659] innerCandidates47_23 5888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_24

def innerTail47_22 : List Nat := 5569 :: innerTail47_23
def innerCandidates47_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates47_22_checked : candidates edge 4 229 5569 innerTail47_23 best.bound = innerCandidates47_22 := by
  unfold candidates
  have row : edge 5569 = SearchData.row5569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_22 : scan edge (search edge 3) 4 [229] 229 innerTail47_22 best = best := by
  rw [innerTail47_22, scan, if_neg (by decide), innerCandidates47_22_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_23
    (if 3 ≤ innerCandidates47_22.length then search edge 3 [229, 5569] innerCandidates47_22 5798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_23

def innerTail47_21 : List Nat := 5077 :: innerTail47_22
def innerCandidates47_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates47_21_checked : candidates edge 4 229 5077 innerTail47_22 best.bound = innerCandidates47_21 := by
  unfold candidates
  have row : edge 5077 = SearchData.row5077.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_21 : scan edge (search edge 3) 4 [229] 229 innerTail47_21 best = best := by
  rw [innerTail47_21, scan, if_neg (by decide), innerCandidates47_21_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_22
    (if 3 ≤ innerCandidates47_21.length then search edge 3 [229, 5077] innerCandidates47_21 5306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_22

def innerTail47_20 : List Nat := 4021 :: innerTail47_21
def innerCandidates47_20 : List Nat := [6733]
set_option maxRecDepth 4096 in
theorem innerCandidates47_20_checked : candidates edge 4 229 4021 innerTail47_21 best.bound = innerCandidates47_20 := by
  unfold candidates
  have row : edge 4021 = SearchData.row4021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_20 : scan edge (search edge 3) 4 [229] 229 innerTail47_20 best = best := by
  rw [innerTail47_20, scan, if_neg (by decide), innerCandidates47_20_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_21
    (if 3 ≤ innerCandidates47_20.length then search edge 3 [229, 4021] innerCandidates47_20 4250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_21

def innerTail47_19 : List Nat := 3919 :: innerTail47_20
def innerCandidates47_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates47_19_checked : candidates edge 4 229 3919 innerTail47_20 best.bound = innerCandidates47_19 := by
  unfold candidates
  have row : edge 3919 = SearchData.row3919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_19 : scan edge (search edge 3) 4 [229] 229 innerTail47_19 best = best := by
  rw [innerTail47_19, scan, if_neg (by decide), innerCandidates47_19_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_20
    (if 3 ≤ innerCandidates47_19.length then search edge 3 [229, 3919] innerCandidates47_19 4148 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_20

def innerTail47_18 : List Nat := 3559 :: innerTail47_19
def innerCandidates47_18 : List Nat := [4021, 5869]
set_option maxRecDepth 4096 in
theorem innerCandidates47_18_checked : candidates edge 4 229 3559 innerTail47_19 best.bound = innerCandidates47_18 := by
  unfold candidates
  have row : edge 3559 = SearchData.row3559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_18 : scan edge (search edge 3) 4 [229] 229 innerTail47_18 best = best := by
  rw [innerTail47_18, scan, if_neg (by decide), innerCandidates47_18_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_19
    (if 3 ≤ innerCandidates47_18.length then search edge 3 [229, 3559] innerCandidates47_18 3788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_19

def innerTail47_17 : List Nat := 3541 :: innerTail47_18
def innerCandidates47_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates47_17_checked : candidates edge 4 229 3541 innerTail47_18 best.bound = innerCandidates47_17 := by
  unfold candidates
  have row : edge 3541 = SearchData.row3541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_17 : scan edge (search edge 3) 4 [229] 229 innerTail47_17 best = best := by
  rw [innerTail47_17, scan, if_neg (by decide), innerCandidates47_17_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_18
    (if 3 ≤ innerCandidates47_17.length then search edge 3 [229, 3541] innerCandidates47_17 3770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_18

def innerTail47_16 : List Nat := 3391 :: innerTail47_17
def innerCandidates47_16 : List Nat := [6733]
set_option maxRecDepth 4096 in
theorem innerCandidates47_16_checked : candidates edge 4 229 3391 innerTail47_17 best.bound = innerCandidates47_16 := by
  unfold candidates
  have row : edge 3391 = SearchData.row3391.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_16 : scan edge (search edge 3) 4 [229] 229 innerTail47_16 best = best := by
  rw [innerTail47_16, scan, if_neg (by decide), innerCandidates47_16_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_17
    (if 3 ≤ innerCandidates47_16.length then search edge 3 [229, 3391] innerCandidates47_16 3620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_17

def innerTail47_15 : List Nat := 3301 :: innerTail47_16
def innerCandidates47_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates47_15_checked : candidates edge 4 229 3301 innerTail47_16 best.bound = innerCandidates47_15 := by
  unfold candidates
  have row : edge 3301 = SearchData.row3301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_15 : scan edge (search edge 3) 4 [229] 229 innerTail47_15 best = best := by
  rw [innerTail47_15, scan, if_neg (by decide), innerCandidates47_15_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_16
    (if 3 ≤ innerCandidates47_15.length then search edge 3 [229, 3301] innerCandidates47_15 3530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_16

def innerTail47_14 : List Nat := 3121 :: innerTail47_15
def innerCandidates47_14 : List Nat := [7759]
set_option maxRecDepth 4096 in
theorem innerCandidates47_14_checked : candidates edge 4 229 3121 innerTail47_15 best.bound = innerCandidates47_14 := by
  unfold candidates
  have row : edge 3121 = SearchData.row3121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_14 : scan edge (search edge 3) 4 [229] 229 innerTail47_14 best = best := by
  rw [innerTail47_14, scan, if_neg (by decide), innerCandidates47_14_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_15
    (if 3 ≤ innerCandidates47_14.length then search edge 3 [229, 3121] innerCandidates47_14 3350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_15

def innerTail47_13 : List Nat := 2767 :: innerTail47_14
def innerCandidates47_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates47_13_checked : candidates edge 4 229 2767 innerTail47_14 best.bound = innerCandidates47_13 := by
  unfold candidates
  have row : edge 2767 = SearchData.row2767.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_13 : scan edge (search edge 3) 4 [229] 229 innerTail47_13 best = best := by
  rw [innerTail47_13, scan, if_neg (by decide), innerCandidates47_13_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_14
    (if 3 ≤ innerCandidates47_13.length then search edge 3 [229, 2767] innerCandidates47_13 2996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_14

def innerTail47_12 : List Nat := 2347 :: innerTail47_13
def innerCandidates47_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates47_12_checked : candidates edge 4 229 2347 innerTail47_13 best.bound = innerCandidates47_12 := by
  unfold candidates
  have row : edge 2347 = PrimePairCache.smallRow2347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_12 : scan edge (search edge 3) 4 [229] 229 innerTail47_12 best = best := by
  rw [innerTail47_12, scan, if_neg (by decide), innerCandidates47_12_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_13
    (if 3 ≤ innerCandidates47_12.length then search edge 3 [229, 2347] innerCandidates47_12 2576 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_13

def innerTail47_11 : List Nat := 2239 :: innerTail47_12
def innerCandidates47_11 : List Nat := [3919]
set_option maxRecDepth 4096 in
theorem innerCandidates47_11_checked : candidates edge 4 229 2239 innerTail47_12 best.bound = innerCandidates47_11 := by
  unfold candidates
  have row : edge 2239 = PrimePairCache.smallRow2239.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_11 : scan edge (search edge 3) 4 [229] 229 innerTail47_11 best = best := by
  rw [innerTail47_11, scan, if_neg (by decide), innerCandidates47_11_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_12
    (if 3 ≤ innerCandidates47_11.length then search edge 3 [229, 2239] innerCandidates47_11 2468 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_12

def innerTail47_10 : List Nat := 2221 :: innerTail47_11
def innerCandidates47_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates47_10_checked : candidates edge 4 229 2221 innerTail47_11 best.bound = innerCandidates47_10 := by
  unfold candidates
  have row : edge 2221 = PrimePairCache.smallRow2221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_10 : scan edge (search edge 3) 4 [229] 229 innerTail47_10 best = best := by
  rw [innerTail47_10, scan, if_neg (by decide), innerCandidates47_10_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_11
    (if 3 ≤ innerCandidates47_10.length then search edge 3 [229, 2221] innerCandidates47_10 2450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_11

def innerTail47_9 : List Nat := 2083 :: innerTail47_10
def innerCandidates47_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates47_9_checked : candidates edge 4 229 2083 innerTail47_10 best.bound = innerCandidates47_9 := by
  unfold candidates
  have row : edge 2083 = PrimePairCache.smallRow2083.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_9 : scan edge (search edge 3) 4 [229] 229 innerTail47_9 best = best := by
  rw [innerTail47_9, scan, if_neg (by decide), innerCandidates47_9_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_10
    (if 3 ≤ innerCandidates47_9.length then search edge 3 [229, 2083] innerCandidates47_9 2312 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_10

def innerTail47_8 : List Nat := 1801 :: innerTail47_9
def innerCandidates47_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates47_8_checked : candidates edge 4 229 1801 innerTail47_9 best.bound = innerCandidates47_8 := by
  unfold candidates
  have row : edge 1801 = PrimePairCache.smallRow1801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_8 : scan edge (search edge 3) 4 [229] 229 innerTail47_8 best = best := by
  rw [innerTail47_8, scan, if_neg (by decide), innerCandidates47_8_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_9
    (if 3 ≤ innerCandidates47_8.length then search edge 3 [229, 1801] innerCandidates47_8 2030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_9

def innerTail47_7 : List Nat := 1231 :: innerTail47_8
def innerCandidates47_7 : List Nat := [5659]
set_option maxRecDepth 4096 in
theorem innerCandidates47_7_checked : candidates edge 4 229 1231 innerTail47_8 best.bound = innerCandidates47_7 := by
  unfold candidates
  have row : edge 1231 = PrimePairCache.smallRow1231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_7 : scan edge (search edge 3) 4 [229] 229 innerTail47_7 best = best := by
  rw [innerTail47_7, scan, if_neg (by decide), innerCandidates47_7_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_8
    (if 3 ≤ innerCandidates47_7.length then search edge 3 [229, 1231] innerCandidates47_7 1460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_8

def innerTail47_6 : List Nat := 937 :: innerTail47_7
def innerCandidates47_6 : List Nat := [7213]
set_option maxRecDepth 4096 in
theorem innerCandidates47_6_checked : candidates edge 4 229 937 innerTail47_7 best.bound = innerCandidates47_6 := by
  unfold candidates
  have row : edge 937 = PrimePairCache.smallRow937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_6 : scan edge (search edge 3) 4 [229] 229 innerTail47_6 best = best := by
  rw [innerTail47_6, scan, if_neg (by decide), innerCandidates47_6_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_7
    (if 3 ≤ innerCandidates47_6.length then search edge 3 [229, 937] innerCandidates47_6 1166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_7

def innerTail47_5 : List Nat := 631 :: innerTail47_6
def innerCandidates47_5 : List Nat := [2221]
set_option maxRecDepth 4096 in
theorem innerCandidates47_5_checked : candidates edge 4 229 631 innerTail47_6 best.bound = innerCandidates47_5 := by
  unfold candidates
  have row : edge 631 = PrimePairCache.smallRow631.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_5 : scan edge (search edge 3) 4 [229] 229 innerTail47_5 best = best := by
  rw [innerTail47_5, scan, if_neg (by decide), innerCandidates47_5_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_6
    (if 3 ≤ innerCandidates47_5.length then search edge 3 [229, 631] innerCandidates47_5 860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_6

def innerTail47_4 : List Nat := 613 :: innerTail47_5
def innerCandidates47_4 : List Nat := [4021, 6733]
set_option maxRecDepth 4096 in
theorem innerCandidates47_4_checked : candidates edge 4 229 613 innerTail47_5 best.bound = innerCandidates47_4 := by
  unfold candidates
  have row : edge 613 = PrimePairCache.smallRow613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_4 : scan edge (search edge 3) 4 [229] 229 innerTail47_4 best = best := by
  rw [innerTail47_4, scan, if_neg (by decide), innerCandidates47_4_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_5
    (if 3 ≤ innerCandidates47_4.length then search edge 3 [229, 613] innerCandidates47_4 842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_5

def innerTail47_3 : List Nat := 547 :: innerTail47_4
def innerCandidates47_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates47_3_checked : candidates edge 4 229 547 innerTail47_4 best.bound = innerCandidates47_3 := by
  unfold candidates
  have row : edge 547 = PrimePairCache.smallRow547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_3 : scan edge (search edge 3) 4 [229] 229 innerTail47_3 best = best := by
  rw [innerTail47_3, scan, if_neg (by decide), innerCandidates47_3_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_4
    (if 3 ≤ innerCandidates47_3.length then search edge 3 [229, 547] innerCandidates47_3 776 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_4

def innerTail47_2 : List Nat := 499 :: innerTail47_3
def innerCandidates47_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates47_2_checked : candidates edge 4 229 499 innerTail47_3 best.bound = innerCandidates47_2 := by
  unfold candidates
  have row : edge 499 = PrimePairCache.smallRow499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_2 : scan edge (search edge 3) 4 [229] 229 innerTail47_2 best = best := by
  rw [innerTail47_2, scan, if_neg (by decide), innerCandidates47_2_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_3
    (if 3 ≤ innerCandidates47_2.length then search edge 3 [229, 499] innerCandidates47_2 728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_3

def innerTail47_1 : List Nat := 433 :: innerTail47_2
def innerCandidates47_1 : List Nat := [1801, 3391, 6247]
set_option maxRecDepth 4096 in
theorem innerCandidates47_1_checked : candidates edge 4 229 433 innerTail47_2 best.bound = innerCandidates47_1 := by
  unfold candidates
  have row : edge 433 = PrimePairCache.smallRow433.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild47_1 : search edge 3 [229, 433] innerCandidates47_1 662 best = best := by decide

theorem innerScan47_1 : scan edge (search edge 3) 4 [229] 229 innerTail47_1 best = best := by
  rw [innerTail47_1, scan, if_neg (by decide), innerCandidates47_1_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_2
    (if 3 ≤ innerCandidates47_1.length then search edge 3 [229, 433] innerCandidates47_1 662 best else best) = best
  rw [if_pos (by decide), innerChild47_1]
  exact innerScan47_2

def innerTail47_0 : List Nat := 373 :: innerTail47_1
def innerCandidates47_0 : List Nat := [937]
set_option maxRecDepth 4096 in
theorem innerCandidates47_0_checked : candidates edge 4 229 373 innerTail47_1 best.bound = innerCandidates47_0 := by
  unfold candidates
  have row : edge 373 = PrimePairCache.smallRow373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan47_0 : scan edge (search edge 3) 4 [229] 229 innerTail47_0 best = best := by
  rw [innerTail47_0, scan, if_neg (by decide), innerCandidates47_0_checked]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_1
    (if 3 ≤ innerCandidates47_0.length then search edge 3 [229, 373] innerCandidates47_0 602 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan47_1

theorem rootBranch47 : search edge 4 [229] rootCandidates47 229 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [229] 229 innerTail47_0 best = best
  exact innerScan47_0

def rootCandidates46 : List Nat := [281, 593, 719, 827, 1229, 1427, 1499, 1721, 2129, 2729, 2903, 2939, 3701, 3911, 4271, 4421, 4691, 4793, 5333, 5591, 7823, 8093]
set_option maxRecDepth 4096 in
theorem rootCandidates46_checked : candidates edge 5 0 227 tail47 best.bound = rootCandidates46 := by
  unfold candidates
  have row : edge 227 = PrimePairCache.smallRow227.contains := by funext q; rfl
  rw [row]
  decide

def innerTail46_22 : List Nat := []
theorem innerScan46_22 : scan edge (search edge 3) 4 [227] 227 innerTail46_22 best = best := rfl

def innerTail46_21 : List Nat := 8093 :: innerTail46_22
def innerCandidates46_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates46_21_checked : candidates edge 4 227 8093 innerTail46_22 best.bound = innerCandidates46_21 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan46_21 : scan edge (search edge 3) 4 [227] 227 innerTail46_21 best = best := by
  rw [innerTail46_21, scan, if_neg (by decide), innerCandidates46_21_checked]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_22
    (if 3 ≤ innerCandidates46_21.length then search edge 3 [227, 8093] innerCandidates46_21 8320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan46_22

def innerTail46_20 : List Nat := 7823 :: innerTail46_21
def innerCandidates46_20 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates46_20_checked : candidates edge 4 227 7823 innerTail46_21 best.bound = innerCandidates46_20 := by
  unfold candidates
  have row : edge 7823 = SearchData.row7823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan46_20 : scan edge (search edge 3) 4 [227] 227 innerTail46_20 best = best := by
  rw [innerTail46_20, scan, if_neg (by decide), innerCandidates46_20_checked]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_21
    (if 3 ≤ innerCandidates46_20.length then search edge 3 [227, 7823] innerCandidates46_20 8050 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan46_21

def innerTail46_19 : List Nat := 5591 :: innerTail46_20
def innerCandidates46_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates46_19_checked : candidates edge 4 227 5591 innerTail46_20 best.bound = innerCandidates46_19 := by
  unfold candidates
  have row : edge 5591 = SearchData.row5591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan46_19 : scan edge (search edge 3) 4 [227] 227 innerTail46_19 best = best := by
  rw [innerTail46_19, scan, if_neg (by decide), innerCandidates46_19_checked]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_20
    (if 3 ≤ innerCandidates46_19.length then search edge 3 [227, 5591] innerCandidates46_19 5818 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan46_20

def innerTail46_18 : List Nat := 5333 :: innerTail46_19
def innerCandidates46_18 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates46_18_checked : candidates edge 4 227 5333 innerTail46_19 best.bound = innerCandidates46_18 := by
  unfold candidates
  have row : edge 5333 = SearchData.row5333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan46_18 : scan edge (search edge 3) 4 [227] 227 innerTail46_18 best = best := by
  rw [innerTail46_18, scan, if_neg (by decide), innerCandidates46_18_checked]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_19
    (if 3 ≤ innerCandidates46_18.length then search edge 3 [227, 5333] innerCandidates46_18 5560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan46_19

def innerTail46_17 : List Nat := 4793 :: innerTail46_18
def innerCandidates46_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates46_17_checked : candidates edge 4 227 4793 innerTail46_18 best.bound = innerCandidates46_17 := by
  unfold candidates
  have row : edge 4793 = SearchData.row4793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan46_17 : scan edge (search edge 3) 4 [227] 227 innerTail46_17 best = best := by
  rw [innerTail46_17, scan, if_neg (by decide), innerCandidates46_17_checked]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_18
    (if 3 ≤ innerCandidates46_17.length then search edge 3 [227, 4793] innerCandidates46_17 5020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan46_18

def innerTail46_16 : List Nat := 4691 :: innerTail46_17
def innerCandidates46_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates46_16_checked : candidates edge 4 227 4691 innerTail46_17 best.bound = innerCandidates46_16 := by
  unfold candidates
  have row : edge 4691 = SearchData.row4691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan46_16 : scan edge (search edge 3) 4 [227] 227 innerTail46_16 best = best := by
  rw [innerTail46_16, scan, if_neg (by decide), innerCandidates46_16_checked]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_17
    (if 3 ≤ innerCandidates46_16.length then search edge 3 [227, 4691] innerCandidates46_16 4918 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan46_17

def innerTail46_15 : List Nat := 4421 :: innerTail46_16
def innerCandidates46_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates46_15_checked : candidates edge 4 227 4421 innerTail46_16 best.bound = innerCandidates46_15 := by
  unfold candidates
  have row : edge 4421 = SearchData.row4421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan46_15 : scan edge (search edge 3) 4 [227] 227 innerTail46_15 best = best := by
  rw [innerTail46_15, scan, if_neg (by decide), innerCandidates46_15_checked]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_16
    (if 3 ≤ innerCandidates46_15.length then search edge 3 [227, 4421] innerCandidates46_15 4648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan46_16

def innerTail46_14 : List Nat := 4271 :: innerTail46_15
def innerCandidates46_14 : List Nat := [7823]
set_option maxRecDepth 4096 in
theorem innerCandidates46_14_checked : candidates edge 4 227 4271 innerTail46_15 best.bound = innerCandidates46_14 := by
  unfold candidates
  have row : edge 4271 = SearchData.row4271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan46_14 : scan edge (search edge 3) 4 [227] 227 innerTail46_14 best = best := by
  rw [innerTail46_14, scan, if_neg (by decide), innerCandidates46_14_checked]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_15
    (if 3 ≤ innerCandidates46_14.length then search edge 3 [227, 4271] innerCandidates46_14 4498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan46_15

def innerTail46_13 : List Nat := 3911 :: innerTail46_14
def innerCandidates46_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates46_13_checked : candidates edge 4 227 3911 innerTail46_14 best.bound = innerCandidates46_13 := by
  unfold candidates
  have row : edge 3911 = SearchData.row3911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan46_13 : scan edge (search edge 3) 4 [227] 227 innerTail46_13 best = best := by
  rw [innerTail46_13, scan, if_neg (by decide), innerCandidates46_13_checked]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_14
    (if 3 ≤ innerCandidates46_13.length then search edge 3 [227, 3911] innerCandidates46_13 4138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan46_14

def innerTail46_12 : List Nat := 3701 :: innerTail46_13
def innerCandidates46_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates46_12_checked : candidates edge 4 227 3701 innerTail46_13 best.bound = innerCandidates46_12 := by
  unfold candidates
  have row : edge 3701 = SearchData.row3701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan46_12 : scan edge (search edge 3) 4 [227] 227 innerTail46_12 best = best := by
  rw [innerTail46_12, scan, if_neg (by decide), innerCandidates46_12_checked]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_13
    (if 3 ≤ innerCandidates46_12.length then search edge 3 [227, 3701] innerCandidates46_12 3928 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan46_13

def innerTail46_11 : List Nat := 2939 :: innerTail46_12
def innerCandidates46_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates46_11_checked : candidates edge 4 227 2939 innerTail46_12 best.bound = innerCandidates46_11 := by
  unfold candidates
  have row : edge 2939 = SearchData.row2939.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan46_11 : scan edge (search edge 3) 4 [227] 227 innerTail46_11 best = best := by
  rw [innerTail46_11, scan, if_neg (by decide), innerCandidates46_11_checked]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_12
    (if 3 ≤ innerCandidates46_11.length then search edge 3 [227, 2939] innerCandidates46_11 3166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan46_12

def innerTail46_10 : List Nat := 2903 :: innerTail46_11
def innerCandidates46_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates46_10_checked : candidates edge 4 227 2903 innerTail46_11 best.bound = innerCandidates46_10 := by
  unfold candidates
  have row : edge 2903 = SearchData.row2903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan46_10 : scan edge (search edge 3) 4 [227] 227 innerTail46_10 best = best := by
  rw [innerTail46_10, scan, if_neg (by decide), innerCandidates46_10_checked]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_11
    (if 3 ≤ innerCandidates46_10.length then search edge 3 [227, 2903] innerCandidates46_10 3130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan46_11

def innerTail46_9 : List Nat := 2729 :: innerTail46_10
def innerCandidates46_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates46_9_checked : candidates edge 4 227 2729 innerTail46_10 best.bound = innerCandidates46_9 := by
  unfold candidates
  have row : edge 2729 = SearchData.row2729.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan46_9 : scan edge (search edge 3) 4 [227] 227 innerTail46_9 best = best := by
  rw [innerTail46_9, scan, if_neg (by decide), innerCandidates46_9_checked]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_10
    (if 3 ≤ innerCandidates46_9.length then search edge 3 [227, 2729] innerCandidates46_9 2956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan46_10

def innerTail46_8 : List Nat := 2129 :: innerTail46_9
def innerCandidates46_8 : List Nat := [2729, 4271]
set_option maxRecDepth 4096 in
theorem innerCandidates46_8_checked : candidates edge 4 227 2129 innerTail46_9 best.bound = innerCandidates46_8 := by
  unfold candidates
  have row : edge 2129 = PrimePairCache.smallRow2129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan46_8 : scan edge (search edge 3) 4 [227] 227 innerTail46_8 best = best := by
  rw [innerTail46_8, scan, if_neg (by decide), innerCandidates46_8_checked]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_9
    (if 3 ≤ innerCandidates46_8.length then search edge 3 [227, 2129] innerCandidates46_8 2356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan46_9

def innerTail46_7 : List Nat := 1721 :: innerTail46_8
def innerCandidates46_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates46_7_checked : candidates edge 4 227 1721 innerTail46_8 best.bound = innerCandidates46_7 := by
  unfold candidates
  have row : edge 1721 = PrimePairCache.smallRow1721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan46_7 : scan edge (search edge 3) 4 [227] 227 innerTail46_7 best = best := by
  rw [innerTail46_7, scan, if_neg (by decide), innerCandidates46_7_checked]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_8
    (if 3 ≤ innerCandidates46_7.length then search edge 3 [227, 1721] innerCandidates46_7 1948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan46_8

def innerTail46_6 : List Nat := 1499 :: innerTail46_7
def innerCandidates46_6 : List Nat := [2129]
set_option maxRecDepth 4096 in
theorem innerCandidates46_6_checked : candidates edge 4 227 1499 innerTail46_7 best.bound = innerCandidates46_6 := by
  unfold candidates
  have row : edge 1499 = PrimePairCache.smallRow1499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan46_6 : scan edge (search edge 3) 4 [227] 227 innerTail46_6 best = best := by
  rw [innerTail46_6, scan, if_neg (by decide), innerCandidates46_6_checked]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_7
    (if 3 ≤ innerCandidates46_6.length then search edge 3 [227, 1499] innerCandidates46_6 1726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan46_7

def innerTail46_5 : List Nat := 1427 :: innerTail46_6
def innerCandidates46_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates46_5_checked : candidates edge 4 227 1427 innerTail46_6 best.bound = innerCandidates46_5 := by
  unfold candidates
  have row : edge 1427 = PrimePairCache.smallRow1427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan46_5 : scan edge (search edge 3) 4 [227] 227 innerTail46_5 best = best := by
  rw [innerTail46_5, scan, if_neg (by decide), innerCandidates46_5_checked]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_6
    (if 3 ≤ innerCandidates46_5.length then search edge 3 [227, 1427] innerCandidates46_5 1654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan46_6

def innerTail46_4 : List Nat := 1229 :: innerTail46_5
def innerCandidates46_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates46_4_checked : candidates edge 4 227 1229 innerTail46_5 best.bound = innerCandidates46_4 := by
  unfold candidates
  have row : edge 1229 = PrimePairCache.smallRow1229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan46_4 : scan edge (search edge 3) 4 [227] 227 innerTail46_4 best = best := by
  rw [innerTail46_4, scan, if_neg (by decide), innerCandidates46_4_checked]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_5
    (if 3 ≤ innerCandidates46_4.length then search edge 3 [227, 1229] innerCandidates46_4 1456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan46_5

def innerTail46_3 : List Nat := 827 :: innerTail46_4
def innerCandidates46_3 : List Nat := [1721]
set_option maxRecDepth 4096 in
theorem innerCandidates46_3_checked : candidates edge 4 227 827 innerTail46_4 best.bound = innerCandidates46_3 := by
  unfold candidates
  have row : edge 827 = PrimePairCache.smallRow827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan46_3 : scan edge (search edge 3) 4 [227] 227 innerTail46_3 best = best := by
  rw [innerTail46_3, scan, if_neg (by decide), innerCandidates46_3_checked]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_4
    (if 3 ≤ innerCandidates46_3.length then search edge 3 [227, 827] innerCandidates46_3 1054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan46_4

def innerTail46_2 : List Nat := 719 :: innerTail46_3
def innerCandidates46_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates46_2_checked : candidates edge 4 227 719 innerTail46_3 best.bound = innerCandidates46_2 := by
  unfold candidates
  have row : edge 719 = PrimePairCache.smallRow719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan46_2 : scan edge (search edge 3) 4 [227] 227 innerTail46_2 best = best := by
  rw [innerTail46_2, scan, if_neg (by decide), innerCandidates46_2_checked]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_3
    (if 3 ≤ innerCandidates46_2.length then search edge 3 [227, 719] innerCandidates46_2 946 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan46_3

def innerTail46_1 : List Nat := 593 :: innerTail46_2
def innerCandidates46_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates46_1_checked : candidates edge 4 227 593 innerTail46_2 best.bound = innerCandidates46_1 := by
  unfold candidates
  have row : edge 593 = PrimePairCache.smallRow593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan46_1 : scan edge (search edge 3) 4 [227] 227 innerTail46_1 best = best := by
  rw [innerTail46_1, scan, if_neg (by decide), innerCandidates46_1_checked]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_2
    (if 3 ≤ innerCandidates46_1.length then search edge 3 [227, 593] innerCandidates46_1 820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan46_2

def innerTail46_0 : List Nat := 281 :: innerTail46_1
def innerCandidates46_0 : List Nat := [719, 1427]
set_option maxRecDepth 4096 in
theorem innerCandidates46_0_checked : candidates edge 4 227 281 innerTail46_1 best.bound = innerCandidates46_0 := by
  unfold candidates
  have row : edge 281 = PrimePairCache.smallRow281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan46_0 : scan edge (search edge 3) 4 [227] 227 innerTail46_0 best = best := by
  rw [innerTail46_0, scan, if_neg (by decide), innerCandidates46_0_checked]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_1
    (if 3 ≤ innerCandidates46_0.length then search edge 3 [227, 281] innerCandidates46_0 508 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan46_1

theorem rootBranch46 : search edge 4 [227] rootCandidates46 227 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [227] 227 innerTail46_0 best = best
  exact innerScan46_0

end ProjectEulerSolutions.PrimePairUpper.Search
