import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates33 : List Nat := [163, 397, 433, 499, 607, 631, 769, 937, 1303, 2281, 2311, 2683, 2689, 3019, 4027, 4273, 4657, 4783, 4801, 5011, 5197, 5281, 5413, 6007, 6217, 6607, 6733, 6967, 7933]
set_option maxRecDepth 4096 in
theorem rootCandidates33_checked : candidates edge 5 0 151 tail34 best.bound = rootCandidates33 := by
  unfold candidates
  have row : edge 151 = PrimePairCache.smallRow151.contains := by funext q; rfl
  rw [row]
  decide

def innerTail33_29 : List Nat := []
theorem innerScan33_29 : scan edge (search edge 3) 4 [151] 151 innerTail33_29 best = best := rfl

def innerTail33_28 : List Nat := 7933 :: innerTail33_29
def innerCandidates33_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates33_28_checked : candidates edge 4 151 7933 innerTail33_29 best.bound = innerCandidates33_28 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_28 : scan edge (search edge 3) 4 [151] 151 innerTail33_28 best = best := by
  rw [innerTail33_28, scan, if_neg (by decide), innerCandidates33_28_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_29
    (if 3 ≤ innerCandidates33_28.length then search edge 3 [151, 7933] innerCandidates33_28 8084 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_29

def innerTail33_27 : List Nat := 6967 :: innerTail33_28
def innerCandidates33_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates33_27_checked : candidates edge 4 151 6967 innerTail33_28 best.bound = innerCandidates33_27 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_27 : scan edge (search edge 3) 4 [151] 151 innerTail33_27 best = best := by
  rw [innerTail33_27, scan, if_neg (by decide), innerCandidates33_27_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_28
    (if 3 ≤ innerCandidates33_27.length then search edge 3 [151, 6967] innerCandidates33_27 7118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_28

def innerTail33_26 : List Nat := 6733 :: innerTail33_27
def innerCandidates33_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates33_26_checked : candidates edge 4 151 6733 innerTail33_27 best.bound = innerCandidates33_26 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_26 : scan edge (search edge 3) 4 [151] 151 innerTail33_26 best = best := by
  rw [innerTail33_26, scan, if_neg (by decide), innerCandidates33_26_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_27
    (if 3 ≤ innerCandidates33_26.length then search edge 3 [151, 6733] innerCandidates33_26 6884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_27

def innerTail33_25 : List Nat := 6607 :: innerTail33_26
def innerCandidates33_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates33_25_checked : candidates edge 4 151 6607 innerTail33_26 best.bound = innerCandidates33_25 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_25 : scan edge (search edge 3) 4 [151] 151 innerTail33_25 best = best := by
  rw [innerTail33_25, scan, if_neg (by decide), innerCandidates33_25_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_26
    (if 3 ≤ innerCandidates33_25.length then search edge 3 [151, 6607] innerCandidates33_25 6758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_26

def innerTail33_24 : List Nat := 6217 :: innerTail33_25
def innerCandidates33_24 : List Nat := [7933]
set_option maxRecDepth 4096 in
theorem innerCandidates33_24_checked : candidates edge 4 151 6217 innerTail33_25 best.bound = innerCandidates33_24 := by
  unfold candidates
  have row : edge 6217 = SearchData.row6217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_24 : scan edge (search edge 3) 4 [151] 151 innerTail33_24 best = best := by
  rw [innerTail33_24, scan, if_neg (by decide), innerCandidates33_24_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_25
    (if 3 ≤ innerCandidates33_24.length then search edge 3 [151, 6217] innerCandidates33_24 6368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_25

def innerTail33_23 : List Nat := 6007 :: innerTail33_24
def innerCandidates33_23 : List Nat := [7933]
set_option maxRecDepth 4096 in
theorem innerCandidates33_23_checked : candidates edge 4 151 6007 innerTail33_24 best.bound = innerCandidates33_23 := by
  unfold candidates
  have row : edge 6007 = SearchData.row6007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_23 : scan edge (search edge 3) 4 [151] 151 innerTail33_23 best = best := by
  rw [innerTail33_23, scan, if_neg (by decide), innerCandidates33_23_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_24
    (if 3 ≤ innerCandidates33_23.length then search edge 3 [151, 6007] innerCandidates33_23 6158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_24

def innerTail33_22 : List Nat := 5413 :: innerTail33_23
def innerCandidates33_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates33_22_checked : candidates edge 4 151 5413 innerTail33_23 best.bound = innerCandidates33_22 := by
  unfold candidates
  have row : edge 5413 = SearchData.row5413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_22 : scan edge (search edge 3) 4 [151] 151 innerTail33_22 best = best := by
  rw [innerTail33_22, scan, if_neg (by decide), innerCandidates33_22_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_23
    (if 3 ≤ innerCandidates33_22.length then search edge 3 [151, 5413] innerCandidates33_22 5564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_23

def innerTail33_21 : List Nat := 5281 :: innerTail33_22
def innerCandidates33_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates33_21_checked : candidates edge 4 151 5281 innerTail33_22 best.bound = innerCandidates33_21 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_21 : scan edge (search edge 3) 4 [151] 151 innerTail33_21 best = best := by
  rw [innerTail33_21, scan, if_neg (by decide), innerCandidates33_21_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_22
    (if 3 ≤ innerCandidates33_21.length then search edge 3 [151, 5281] innerCandidates33_21 5432 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_22

def innerTail33_20 : List Nat := 5197 :: innerTail33_21
def innerCandidates33_20 : List Nat := [6733]
set_option maxRecDepth 4096 in
theorem innerCandidates33_20_checked : candidates edge 4 151 5197 innerTail33_21 best.bound = innerCandidates33_20 := by
  unfold candidates
  have row : edge 5197 = SearchData.row5197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_20 : scan edge (search edge 3) 4 [151] 151 innerTail33_20 best = best := by
  rw [innerTail33_20, scan, if_neg (by decide), innerCandidates33_20_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_21
    (if 3 ≤ innerCandidates33_20.length then search edge 3 [151, 5197] innerCandidates33_20 5348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_21

def innerTail33_19 : List Nat := 5011 :: innerTail33_20
def innerCandidates33_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates33_19_checked : candidates edge 4 151 5011 innerTail33_20 best.bound = innerCandidates33_19 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_19 : scan edge (search edge 3) 4 [151] 151 innerTail33_19 best = best := by
  rw [innerTail33_19, scan, if_neg (by decide), innerCandidates33_19_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_20
    (if 3 ≤ innerCandidates33_19.length then search edge 3 [151, 5011] innerCandidates33_19 5162 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_20

def innerTail33_18 : List Nat := 4801 :: innerTail33_19
def innerCandidates33_18 : List Nat := [5197, 6733]
set_option maxRecDepth 4096 in
theorem innerCandidates33_18_checked : candidates edge 4 151 4801 innerTail33_19 best.bound = innerCandidates33_18 := by
  unfold candidates
  have row : edge 4801 = SearchData.row4801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_18 : scan edge (search edge 3) 4 [151] 151 innerTail33_18 best = best := by
  rw [innerTail33_18, scan, if_neg (by decide), innerCandidates33_18_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_19
    (if 3 ≤ innerCandidates33_18.length then search edge 3 [151, 4801] innerCandidates33_18 4952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_19

def innerTail33_17 : List Nat := 4783 :: innerTail33_18
def innerCandidates33_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates33_17_checked : candidates edge 4 151 4783 innerTail33_18 best.bound = innerCandidates33_17 := by
  unfold candidates
  have row : edge 4783 = SearchData.row4783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_17 : scan edge (search edge 3) 4 [151] 151 innerTail33_17 best = best := by
  rw [innerTail33_17, scan, if_neg (by decide), innerCandidates33_17_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_18
    (if 3 ≤ innerCandidates33_17.length then search edge 3 [151, 4783] innerCandidates33_17 4934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_18

def innerTail33_16 : List Nat := 4657 :: innerTail33_17
def innerCandidates33_16 : List Nat := [6217, 7933]
set_option maxRecDepth 4096 in
theorem innerCandidates33_16_checked : candidates edge 4 151 4657 innerTail33_17 best.bound = innerCandidates33_16 := by
  unfold candidates
  have row : edge 4657 = SearchData.row4657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_16 : scan edge (search edge 3) 4 [151] 151 innerTail33_16 best = best := by
  rw [innerTail33_16, scan, if_neg (by decide), innerCandidates33_16_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_17
    (if 3 ≤ innerCandidates33_16.length then search edge 3 [151, 4657] innerCandidates33_16 4808 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_17

def innerTail33_15 : List Nat := 4273 :: innerTail33_16
def innerCandidates33_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates33_15_checked : candidates edge 4 151 4273 innerTail33_16 best.bound = innerCandidates33_15 := by
  unfold candidates
  have row : edge 4273 = SearchData.row4273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_15 : scan edge (search edge 3) 4 [151] 151 innerTail33_15 best = best := by
  rw [innerTail33_15, scan, if_neg (by decide), innerCandidates33_15_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_16
    (if 3 ≤ innerCandidates33_15.length then search edge 3 [151, 4273] innerCandidates33_15 4424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_16

def innerTail33_14 : List Nat := 4027 :: innerTail33_15
def innerCandidates33_14 : List Nat := [4783, 5413]
set_option maxRecDepth 4096 in
theorem innerCandidates33_14_checked : candidates edge 4 151 4027 innerTail33_15 best.bound = innerCandidates33_14 := by
  unfold candidates
  have row : edge 4027 = SearchData.row4027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_14 : scan edge (search edge 3) 4 [151] 151 innerTail33_14 best = best := by
  rw [innerTail33_14, scan, if_neg (by decide), innerCandidates33_14_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_15
    (if 3 ≤ innerCandidates33_14.length then search edge 3 [151, 4027] innerCandidates33_14 4178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_15

def innerTail33_13 : List Nat := 3019 :: innerTail33_14
def innerCandidates33_13 : List Nat := [5413]
set_option maxRecDepth 4096 in
theorem innerCandidates33_13_checked : candidates edge 4 151 3019 innerTail33_14 best.bound = innerCandidates33_13 := by
  unfold candidates
  have row : edge 3019 = SearchData.row3019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_13 : scan edge (search edge 3) 4 [151] 151 innerTail33_13 best = best := by
  rw [innerTail33_13, scan, if_neg (by decide), innerCandidates33_13_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_14
    (if 3 ≤ innerCandidates33_13.length then search edge 3 [151, 3019] innerCandidates33_13 3170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_14

def innerTail33_12 : List Nat := 2689 :: innerTail33_13
def innerCandidates33_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates33_12_checked : candidates edge 4 151 2689 innerTail33_13 best.bound = innerCandidates33_12 := by
  unfold candidates
  have row : edge 2689 = SearchData.row2689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_12 : scan edge (search edge 3) 4 [151] 151 innerTail33_12 best = best := by
  rw [innerTail33_12, scan, if_neg (by decide), innerCandidates33_12_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_13
    (if 3 ≤ innerCandidates33_12.length then search edge 3 [151, 2689] innerCandidates33_12 2840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_13

def innerTail33_11 : List Nat := 2683 :: innerTail33_12
def innerCandidates33_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates33_11_checked : candidates edge 4 151 2683 innerTail33_12 best.bound = innerCandidates33_11 := by
  unfold candidates
  have row : edge 2683 = SearchData.row2683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_11 : scan edge (search edge 3) 4 [151] 151 innerTail33_11 best = best := by
  rw [innerTail33_11, scan, if_neg (by decide), innerCandidates33_11_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_12
    (if 3 ≤ innerCandidates33_11.length then search edge 3 [151, 2683] innerCandidates33_11 2834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_12

def innerTail33_10 : List Nat := 2311 :: innerTail33_11
def innerCandidates33_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates33_10_checked : candidates edge 4 151 2311 innerTail33_11 best.bound = innerCandidates33_10 := by
  unfold candidates
  have row : edge 2311 = PrimePairCache.smallRow2311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_10 : scan edge (search edge 3) 4 [151] 151 innerTail33_10 best = best := by
  rw [innerTail33_10, scan, if_neg (by decide), innerCandidates33_10_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_11
    (if 3 ≤ innerCandidates33_10.length then search edge 3 [151, 2311] innerCandidates33_10 2462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_11

def innerTail33_9 : List Nat := 2281 :: innerTail33_10
def innerCandidates33_9 : List Nat := [4273]
set_option maxRecDepth 4096 in
theorem innerCandidates33_9_checked : candidates edge 4 151 2281 innerTail33_10 best.bound = innerCandidates33_9 := by
  unfold candidates
  have row : edge 2281 = PrimePairCache.smallRow2281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_9 : scan edge (search edge 3) 4 [151] 151 innerTail33_9 best = best := by
  rw [innerTail33_9, scan, if_neg (by decide), innerCandidates33_9_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_10
    (if 3 ≤ innerCandidates33_9.length then search edge 3 [151, 2281] innerCandidates33_9 2432 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_10

def innerTail33_8 : List Nat := 1303 :: innerTail33_9
def innerCandidates33_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates33_8_checked : candidates edge 4 151 1303 innerTail33_9 best.bound = innerCandidates33_8 := by
  unfold candidates
  have row : edge 1303 = PrimePairCache.smallRow1303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_8 : scan edge (search edge 3) 4 [151] 151 innerTail33_8 best = best := by
  rw [innerTail33_8, scan, if_neg (by decide), innerCandidates33_8_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_9
    (if 3 ≤ innerCandidates33_8.length then search edge 3 [151, 1303] innerCandidates33_8 1454 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_9

def innerTail33_7 : List Nat := 937 :: innerTail33_8
def innerCandidates33_7 : List Nat := [4273, 5281]
set_option maxRecDepth 4096 in
theorem innerCandidates33_7_checked : candidates edge 4 151 937 innerTail33_8 best.bound = innerCandidates33_7 := by
  unfold candidates
  have row : edge 937 = PrimePairCache.smallRow937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_7 : scan edge (search edge 3) 4 [151] 151 innerTail33_7 best = best := by
  rw [innerTail33_7, scan, if_neg (by decide), innerCandidates33_7_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_8
    (if 3 ≤ innerCandidates33_7.length then search edge 3 [151, 937] innerCandidates33_7 1088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_8

def innerTail33_6 : List Nat := 769 :: innerTail33_7
def innerCandidates33_6 : List Nat := [5197, 6967]
set_option maxRecDepth 4096 in
theorem innerCandidates33_6_checked : candidates edge 4 151 769 innerTail33_7 best.bound = innerCandidates33_6 := by
  unfold candidates
  have row : edge 769 = PrimePairCache.smallRow769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_6 : scan edge (search edge 3) 4 [151] 151 innerTail33_6 best = best := by
  rw [innerTail33_6, scan, if_neg (by decide), innerCandidates33_6_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_7
    (if 3 ≤ innerCandidates33_6.length then search edge 3 [151, 769] innerCandidates33_6 920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_7

def innerTail33_5 : List Nat := 631 :: innerTail33_6
def innerCandidates33_5 : List Nat := [4801]
set_option maxRecDepth 4096 in
theorem innerCandidates33_5_checked : candidates edge 4 151 631 innerTail33_6 best.bound = innerCandidates33_5 := by
  unfold candidates
  have row : edge 631 = PrimePairCache.smallRow631.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_5 : scan edge (search edge 3) 4 [151] 151 innerTail33_5 best = best := by
  rw [innerTail33_5, scan, if_neg (by decide), innerCandidates33_5_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_6
    (if 3 ≤ innerCandidates33_5.length then search edge 3 [151, 631] innerCandidates33_5 782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_6

def innerTail33_4 : List Nat := 607 :: innerTail33_5
def innerCandidates33_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates33_4_checked : candidates edge 4 151 607 innerTail33_5 best.bound = innerCandidates33_4 := by
  unfold candidates
  have row : edge 607 = PrimePairCache.smallRow607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_4 : scan edge (search edge 3) 4 [151] 151 innerTail33_4 best = best := by
  rw [innerTail33_4, scan, if_neg (by decide), innerCandidates33_4_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_5
    (if 3 ≤ innerCandidates33_4.length then search edge 3 [151, 607] innerCandidates33_4 758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_5

def innerTail33_3 : List Nat := 499 :: innerTail33_4
def innerCandidates33_3 : List Nat := [1303, 5281]
set_option maxRecDepth 4096 in
theorem innerCandidates33_3_checked : candidates edge 4 151 499 innerTail33_4 best.bound = innerCandidates33_3 := by
  unfold candidates
  have row : edge 499 = PrimePairCache.smallRow499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_3 : scan edge (search edge 3) 4 [151] 151 innerTail33_3 best = best := by
  rw [innerTail33_3, scan, if_neg (by decide), innerCandidates33_3_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_4
    (if 3 ≤ innerCandidates33_3.length then search edge 3 [151, 499] innerCandidates33_3 650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_4

def innerTail33_2 : List Nat := 433 :: innerTail33_3
def innerCandidates33_2 : List Nat := [5281]
set_option maxRecDepth 4096 in
theorem innerCandidates33_2_checked : candidates edge 4 151 433 innerTail33_3 best.bound = innerCandidates33_2 := by
  unfold candidates
  have row : edge 433 = PrimePairCache.smallRow433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_2 : scan edge (search edge 3) 4 [151] 151 innerTail33_2 best = best := by
  rw [innerTail33_2, scan, if_neg (by decide), innerCandidates33_2_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_3
    (if 3 ≤ innerCandidates33_2.length then search edge 3 [151, 433] innerCandidates33_2 584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_3

def innerTail33_1 : List Nat := 397 :: innerTail33_2
def innerCandidates33_1 : List Nat := [4657, 5197, 7933]
set_option maxRecDepth 4096 in
theorem innerCandidates33_1_checked : candidates edge 4 151 397 innerTail33_2 best.bound = innerCandidates33_1 := by
  unfold candidates
  have row : edge 397 = PrimePairCache.smallRow397.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild33_1 : search edge 3 [151, 397] innerCandidates33_1 548 best = best := by decide

theorem innerScan33_1 : scan edge (search edge 3) 4 [151] 151 innerTail33_1 best = best := by
  rw [innerTail33_1, scan, if_neg (by decide), innerCandidates33_1_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_2
    (if 3 ≤ innerCandidates33_1.length then search edge 3 [151, 397] innerCandidates33_1 548 best else best) = best
  rw [if_pos (by decide), innerChild33_1]
  exact innerScan33_2

def innerTail33_0 : List Nat := 163 :: innerTail33_1
def innerCandidates33_0 : List Nat := [2311]
set_option maxRecDepth 4096 in
theorem innerCandidates33_0_checked : candidates edge 4 151 163 innerTail33_1 best.bound = innerCandidates33_0 := by
  unfold candidates
  have row : edge 163 = PrimePairCache.smallRow163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan33_0 : scan edge (search edge 3) 4 [151] 151 innerTail33_0 best = best := by
  rw [innerTail33_0, scan, if_neg (by decide), innerCandidates33_0_checked]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_1
    (if 3 ≤ innerCandidates33_0.length then search edge 3 [151, 163] innerCandidates33_0 314 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan33_1

theorem rootBranch33 : search edge 4 [151] rootCandidates33 151 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [151] 151 innerTail33_0 best = best
  exact innerScan33_0

def rootCandidates32 : List Nat := [173, 251, 491, 563, 953, 971, 1571, 1583, 1979, 2819, 3491, 3617, 4133, 4289, 5009, 5279, 5717, 5867, 5939, 6581, 6791, 6917, 7019, 7151, 7193, 7283, 8009]
set_option maxRecDepth 4096 in
theorem rootCandidates32_checked : candidates edge 5 0 149 tail33 best.bound = rootCandidates32 := by
  unfold candidates
  have row : edge 149 = PrimePairCache.smallRow149.contains := by funext q; rfl
  rw [row]
  decide

def innerTail32_27 : List Nat := []
theorem innerScan32_27 : scan edge (search edge 3) 4 [149] 149 innerTail32_27 best = best := rfl

def innerTail32_26 : List Nat := 8009 :: innerTail32_27
def innerCandidates32_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates32_26_checked : candidates edge 4 149 8009 innerTail32_27 best.bound = innerCandidates32_26 := by
  unfold candidates
  have row : edge 8009 = SearchData.row8009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_26 : scan edge (search edge 3) 4 [149] 149 innerTail32_26 best = best := by
  rw [innerTail32_26, scan, if_neg (by decide), innerCandidates32_26_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_27
    (if 3 ≤ innerCandidates32_26.length then search edge 3 [149, 8009] innerCandidates32_26 8158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_27

def innerTail32_25 : List Nat := 7283 :: innerTail32_26
def innerCandidates32_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates32_25_checked : candidates edge 4 149 7283 innerTail32_26 best.bound = innerCandidates32_25 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_25 : scan edge (search edge 3) 4 [149] 149 innerTail32_25 best = best := by
  rw [innerTail32_25, scan, if_neg (by decide), innerCandidates32_25_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_26
    (if 3 ≤ innerCandidates32_25.length then search edge 3 [149, 7283] innerCandidates32_25 7432 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_26

def innerTail32_24 : List Nat := 7193 :: innerTail32_25
def innerCandidates32_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates32_24_checked : candidates edge 4 149 7193 innerTail32_25 best.bound = innerCandidates32_24 := by
  unfold candidates
  have row : edge 7193 = SearchData.row7193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_24 : scan edge (search edge 3) 4 [149] 149 innerTail32_24 best = best := by
  rw [innerTail32_24, scan, if_neg (by decide), innerCandidates32_24_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_25
    (if 3 ≤ innerCandidates32_24.length then search edge 3 [149, 7193] innerCandidates32_24 7342 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_25

def innerTail32_23 : List Nat := 7151 :: innerTail32_24
def innerCandidates32_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates32_23_checked : candidates edge 4 149 7151 innerTail32_24 best.bound = innerCandidates32_23 := by
  unfold candidates
  have row : edge 7151 = SearchData.row7151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_23 : scan edge (search edge 3) 4 [149] 149 innerTail32_23 best = best := by
  rw [innerTail32_23, scan, if_neg (by decide), innerCandidates32_23_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_24
    (if 3 ≤ innerCandidates32_23.length then search edge 3 [149, 7151] innerCandidates32_23 7300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_24

def innerTail32_22 : List Nat := 7019 :: innerTail32_23
def innerCandidates32_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates32_22_checked : candidates edge 4 149 7019 innerTail32_23 best.bound = innerCandidates32_22 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_22 : scan edge (search edge 3) 4 [149] 149 innerTail32_22 best = best := by
  rw [innerTail32_22, scan, if_neg (by decide), innerCandidates32_22_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_23
    (if 3 ≤ innerCandidates32_22.length then search edge 3 [149, 7019] innerCandidates32_22 7168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_23

def innerTail32_21 : List Nat := 6917 :: innerTail32_22
def innerCandidates32_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates32_21_checked : candidates edge 4 149 6917 innerTail32_22 best.bound = innerCandidates32_21 := by
  unfold candidates
  have row : edge 6917 = SearchData.row6917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_21 : scan edge (search edge 3) 4 [149] 149 innerTail32_21 best = best := by
  rw [innerTail32_21, scan, if_neg (by decide), innerCandidates32_21_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_22
    (if 3 ≤ innerCandidates32_21.length then search edge 3 [149, 6917] innerCandidates32_21 7066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_22

def innerTail32_20 : List Nat := 6791 :: innerTail32_21
def innerCandidates32_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates32_20_checked : candidates edge 4 149 6791 innerTail32_21 best.bound = innerCandidates32_20 := by
  unfold candidates
  have row : edge 6791 = SearchData.row6791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_20 : scan edge (search edge 3) 4 [149] 149 innerTail32_20 best = best := by
  rw [innerTail32_20, scan, if_neg (by decide), innerCandidates32_20_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_21
    (if 3 ≤ innerCandidates32_20.length then search edge 3 [149, 6791] innerCandidates32_20 6940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_21

def innerTail32_19 : List Nat := 6581 :: innerTail32_20
def innerCandidates32_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates32_19_checked : candidates edge 4 149 6581 innerTail32_20 best.bound = innerCandidates32_19 := by
  unfold candidates
  have row : edge 6581 = SearchData.row6581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_19 : scan edge (search edge 3) 4 [149] 149 innerTail32_19 best = best := by
  rw [innerTail32_19, scan, if_neg (by decide), innerCandidates32_19_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_20
    (if 3 ≤ innerCandidates32_19.length then search edge 3 [149, 6581] innerCandidates32_19 6730 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_20

def innerTail32_18 : List Nat := 5939 :: innerTail32_19
def innerCandidates32_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates32_18_checked : candidates edge 4 149 5939 innerTail32_19 best.bound = innerCandidates32_18 := by
  unfold candidates
  have row : edge 5939 = SearchData.row5939.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_18 : scan edge (search edge 3) 4 [149] 149 innerTail32_18 best = best := by
  rw [innerTail32_18, scan, if_neg (by decide), innerCandidates32_18_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_19
    (if 3 ≤ innerCandidates32_18.length then search edge 3 [149, 5939] innerCandidates32_18 6088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_19

def innerTail32_17 : List Nat := 5867 :: innerTail32_18
def innerCandidates32_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates32_17_checked : candidates edge 4 149 5867 innerTail32_18 best.bound = innerCandidates32_17 := by
  unfold candidates
  have row : edge 5867 = SearchData.row5867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_17 : scan edge (search edge 3) 4 [149] 149 innerTail32_17 best = best := by
  rw [innerTail32_17, scan, if_neg (by decide), innerCandidates32_17_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_18
    (if 3 ≤ innerCandidates32_17.length then search edge 3 [149, 5867] innerCandidates32_17 6016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_18

def innerTail32_16 : List Nat := 5717 :: innerTail32_17
def innerCandidates32_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates32_16_checked : candidates edge 4 149 5717 innerTail32_17 best.bound = innerCandidates32_16 := by
  unfold candidates
  have row : edge 5717 = SearchData.row5717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_16 : scan edge (search edge 3) 4 [149] 149 innerTail32_16 best = best := by
  rw [innerTail32_16, scan, if_neg (by decide), innerCandidates32_16_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_17
    (if 3 ≤ innerCandidates32_16.length then search edge 3 [149, 5717] innerCandidates32_16 5866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_17

def innerTail32_15 : List Nat := 5279 :: innerTail32_16
def innerCandidates32_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates32_15_checked : candidates edge 4 149 5279 innerTail32_16 best.bound = innerCandidates32_15 := by
  unfold candidates
  have row : edge 5279 = SearchData.row5279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_15 : scan edge (search edge 3) 4 [149] 149 innerTail32_15 best = best := by
  rw [innerTail32_15, scan, if_neg (by decide), innerCandidates32_15_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_16
    (if 3 ≤ innerCandidates32_15.length then search edge 3 [149, 5279] innerCandidates32_15 5428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_16

def innerTail32_14 : List Nat := 5009 :: innerTail32_15
def innerCandidates32_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates32_14_checked : candidates edge 4 149 5009 innerTail32_15 best.bound = innerCandidates32_14 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_14 : scan edge (search edge 3) 4 [149] 149 innerTail32_14 best = best := by
  rw [innerTail32_14, scan, if_neg (by decide), innerCandidates32_14_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_15
    (if 3 ≤ innerCandidates32_14.length then search edge 3 [149, 5009] innerCandidates32_14 5158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_15

def innerTail32_13 : List Nat := 4289 :: innerTail32_14
def innerCandidates32_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates32_13_checked : candidates edge 4 149 4289 innerTail32_14 best.bound = innerCandidates32_13 := by
  unfold candidates
  have row : edge 4289 = SearchData.row4289.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_13 : scan edge (search edge 3) 4 [149] 149 innerTail32_13 best = best := by
  rw [innerTail32_13, scan, if_neg (by decide), innerCandidates32_13_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_14
    (if 3 ≤ innerCandidates32_13.length then search edge 3 [149, 4289] innerCandidates32_13 4438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_14

def innerTail32_12 : List Nat := 4133 :: innerTail32_13
def innerCandidates32_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates32_12_checked : candidates edge 4 149 4133 innerTail32_13 best.bound = innerCandidates32_12 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_12 : scan edge (search edge 3) 4 [149] 149 innerTail32_12 best = best := by
  rw [innerTail32_12, scan, if_neg (by decide), innerCandidates32_12_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_13
    (if 3 ≤ innerCandidates32_12.length then search edge 3 [149, 4133] innerCandidates32_12 4282 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_13

def innerTail32_11 : List Nat := 3617 :: innerTail32_12
def innerCandidates32_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates32_11_checked : candidates edge 4 149 3617 innerTail32_12 best.bound = innerCandidates32_11 := by
  unfold candidates
  have row : edge 3617 = SearchData.row3617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_11 : scan edge (search edge 3) 4 [149] 149 innerTail32_11 best = best := by
  rw [innerTail32_11, scan, if_neg (by decide), innerCandidates32_11_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_12
    (if 3 ≤ innerCandidates32_11.length then search edge 3 [149, 3617] innerCandidates32_11 3766 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_12

def innerTail32_10 : List Nat := 3491 :: innerTail32_11
def innerCandidates32_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates32_10_checked : candidates edge 4 149 3491 innerTail32_11 best.bound = innerCandidates32_10 := by
  unfold candidates
  have row : edge 3491 = SearchData.row3491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_10 : scan edge (search edge 3) 4 [149] 149 innerTail32_10 best = best := by
  rw [innerTail32_10, scan, if_neg (by decide), innerCandidates32_10_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_11
    (if 3 ≤ innerCandidates32_10.length then search edge 3 [149, 3491] innerCandidates32_10 3640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_11

def innerTail32_9 : List Nat := 2819 :: innerTail32_10
def innerCandidates32_9 : List Nat := [3617]
set_option maxRecDepth 4096 in
theorem innerCandidates32_9_checked : candidates edge 4 149 2819 innerTail32_10 best.bound = innerCandidates32_9 := by
  unfold candidates
  have row : edge 2819 = SearchData.row2819.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_9 : scan edge (search edge 3) 4 [149] 149 innerTail32_9 best = best := by
  rw [innerTail32_9, scan, if_neg (by decide), innerCandidates32_9_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_10
    (if 3 ≤ innerCandidates32_9.length then search edge 3 [149, 2819] innerCandidates32_9 2968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_10

def innerTail32_8 : List Nat := 1979 :: innerTail32_9
def innerCandidates32_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates32_8_checked : candidates edge 4 149 1979 innerTail32_9 best.bound = innerCandidates32_8 := by
  unfold candidates
  have row : edge 1979 = PrimePairCache.smallRow1979.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_8 : scan edge (search edge 3) 4 [149] 149 innerTail32_8 best = best := by
  rw [innerTail32_8, scan, if_neg (by decide), innerCandidates32_8_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_9
    (if 3 ≤ innerCandidates32_8.length then search edge 3 [149, 1979] innerCandidates32_8 2128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_9

def innerTail32_7 : List Nat := 1583 :: innerTail32_8
def innerCandidates32_7 : List Nat := [5009]
set_option maxRecDepth 4096 in
theorem innerCandidates32_7_checked : candidates edge 4 149 1583 innerTail32_8 best.bound = innerCandidates32_7 := by
  unfold candidates
  have row : edge 1583 = PrimePairCache.smallRow1583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_7 : scan edge (search edge 3) 4 [149] 149 innerTail32_7 best = best := by
  rw [innerTail32_7, scan, if_neg (by decide), innerCandidates32_7_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_8
    (if 3 ≤ innerCandidates32_7.length then search edge 3 [149, 1583] innerCandidates32_7 1732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_8

def innerTail32_6 : List Nat := 1571 :: innerTail32_7
def innerCandidates32_6 : List Nat := [2819, 4133]
set_option maxRecDepth 4096 in
theorem innerCandidates32_6_checked : candidates edge 4 149 1571 innerTail32_7 best.bound = innerCandidates32_6 := by
  unfold candidates
  have row : edge 1571 = PrimePairCache.smallRow1571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_6 : scan edge (search edge 3) 4 [149] 149 innerTail32_6 best = best := by
  rw [innerTail32_6, scan, if_neg (by decide), innerCandidates32_6_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_7
    (if 3 ≤ innerCandidates32_6.length then search edge 3 [149, 1571] innerCandidates32_6 1720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_7

def innerTail32_5 : List Nat := 971 :: innerTail32_6
def innerCandidates32_5 : List Nat := [5009, 7193]
set_option maxRecDepth 4096 in
theorem innerCandidates32_5_checked : candidates edge 4 149 971 innerTail32_6 best.bound = innerCandidates32_5 := by
  unfold candidates
  have row : edge 971 = PrimePairCache.smallRow971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_5 : scan edge (search edge 3) 4 [149] 149 innerTail32_5 best = best := by
  rw [innerTail32_5, scan, if_neg (by decide), innerCandidates32_5_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_6
    (if 3 ≤ innerCandidates32_5.length then search edge 3 [149, 971] innerCandidates32_5 1120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_6

def innerTail32_4 : List Nat := 953 :: innerTail32_5
def innerCandidates32_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates32_4_checked : candidates edge 4 149 953 innerTail32_5 best.bound = innerCandidates32_4 := by
  unfold candidates
  have row : edge 953 = PrimePairCache.smallRow953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_4 : scan edge (search edge 3) 4 [149] 149 innerTail32_4 best = best := by
  rw [innerTail32_4, scan, if_neg (by decide), innerCandidates32_4_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_5
    (if 3 ≤ innerCandidates32_4.length then search edge 3 [149, 953] innerCandidates32_4 1102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_5

def innerTail32_3 : List Nat := 563 :: innerTail32_4
def innerCandidates32_3 : List Nat := [971, 1979, 5717, 6581, 6791, 7019]
set_option maxRecDepth 4096 in
theorem innerCandidates32_3_checked : candidates edge 4 149 563 innerTail32_4 best.bound = innerCandidates32_3 := by
  unfold candidates
  have row : edge 563 = PrimePairCache.smallRow563.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild32_3 : search edge 3 [149, 563] innerCandidates32_3 712 best = best := by decide

theorem innerScan32_3 : scan edge (search edge 3) 4 [149] 149 innerTail32_3 best = best := by
  rw [innerTail32_3, scan, if_neg (by decide), innerCandidates32_3_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_4
    (if 3 ≤ innerCandidates32_3.length then search edge 3 [149, 563] innerCandidates32_3 712 best else best) = best
  rw [if_pos (by decide), innerChild32_3]
  exact innerScan32_4

def innerTail32_2 : List Nat := 491 :: innerTail32_3
def innerCandidates32_2 : List Nat := [3617]
set_option maxRecDepth 4096 in
theorem innerCandidates32_2_checked : candidates edge 4 149 491 innerTail32_3 best.bound = innerCandidates32_2 := by
  unfold candidates
  have row : edge 491 = PrimePairCache.smallRow491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_2 : scan edge (search edge 3) 4 [149] 149 innerTail32_2 best = best := by
  rw [innerTail32_2, scan, if_neg (by decide), innerCandidates32_2_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_3
    (if 3 ≤ innerCandidates32_2.length then search edge 3 [149, 491] innerCandidates32_2 640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_3

def innerTail32_1 : List Nat := 251 :: innerTail32_2
def innerCandidates32_1 : List Nat := [491, 971, 1979, 6581]
set_option maxRecDepth 4096 in
theorem innerCandidates32_1_checked : candidates edge 4 149 251 innerTail32_2 best.bound = innerCandidates32_1 := by
  unfold candidates
  have row : edge 251 = PrimePairCache.smallRow251.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild32_1 : search edge 3 [149, 251] innerCandidates32_1 400 best = best := by decide

theorem innerScan32_1 : scan edge (search edge 3) 4 [149] 149 innerTail32_1 best = best := by
  rw [innerTail32_1, scan, if_neg (by decide), innerCandidates32_1_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_2
    (if 3 ≤ innerCandidates32_1.length then search edge 3 [149, 251] innerCandidates32_1 400 best else best) = best
  rw [if_pos (by decide), innerChild32_1]
  exact innerScan32_2

def innerTail32_0 : List Nat := 173 :: innerTail32_1
def innerCandidates32_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates32_0_checked : candidates edge 4 149 173 innerTail32_1 best.bound = innerCandidates32_0 := by
  unfold candidates
  have row : edge 173 = PrimePairCache.smallRow173.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan32_0 : scan edge (search edge 3) 4 [149] 149 innerTail32_0 best = best := by
  rw [innerTail32_0, scan, if_neg (by decide), innerCandidates32_0_checked]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_1
    (if 3 ≤ innerCandidates32_0.length then search edge 3 [149, 173] innerCandidates32_0 322 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan32_1

theorem rootBranch32 : search edge 4 [149] rootCandidates32 149 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [149] 149 innerTail32_0 best = best
  exact innerScan32_0

end ProjectEulerSolutions.PrimePairUpper.Search
