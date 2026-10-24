import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates64 : List Nat := [349, 577, 739, 883, 907, 937, 1303, 1531, 1669, 1831, 1933, 1999, 2719, 2731, 2833, 3067, 3361, 3637, 3769, 4261, 4513, 4639, 4651, 5107, 5407, 5779, 6361, 6427, 6451, 6553, 6949, 7417, 7537, 8221, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates64_checked : candidates edge 5 0 331 tail65 best.bound = rootCandidates64 := by
  unfold candidates
  have row : edge 331 = PrimePairCache.smallRow331.contains := by funext q; rfl
  rw [row]
  decide

def innerTail64_35 : List Nat := []
theorem innerScan64_35 : scan edge (search edge 3) 4 [331] 331 innerTail64_35 best = best := rfl

def innerTail64_34 : List Nat := 8377 :: innerTail64_35
def innerCandidates64_34 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates64_34_checked : candidates edge 4 331 8377 innerTail64_35 best.bound = innerCandidates64_34 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_34 : scan edge (search edge 3) 4 [331] 331 innerTail64_34 best = best := by
  rw [innerTail64_34, scan, if_neg (by decide), innerCandidates64_34_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_35
    (if 3 ≤ innerCandidates64_34.length then search edge 3 [331, 8377] innerCandidates64_34 8708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_35

def innerTail64_33 : List Nat := 8221 :: innerTail64_34
def innerCandidates64_33 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates64_33_checked : candidates edge 4 331 8221 innerTail64_34 best.bound = innerCandidates64_33 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_33 : scan edge (search edge 3) 4 [331] 331 innerTail64_33 best = best := by
  rw [innerTail64_33, scan, if_neg (by decide), innerCandidates64_33_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_34
    (if 3 ≤ innerCandidates64_33.length then search edge 3 [331, 8221] innerCandidates64_33 8552 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_34

def innerTail64_32 : List Nat := 7537 :: innerTail64_33
def innerCandidates64_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates64_32_checked : candidates edge 4 331 7537 innerTail64_33 best.bound = innerCandidates64_32 := by
  unfold candidates
  have row : edge 7537 = SearchData.row7537.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_32 : scan edge (search edge 3) 4 [331] 331 innerTail64_32 best = best := by
  rw [innerTail64_32, scan, if_neg (by decide), innerCandidates64_32_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_33
    (if 3 ≤ innerCandidates64_32.length then search edge 3 [331, 7537] innerCandidates64_32 7868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_33

def innerTail64_31 : List Nat := 7417 :: innerTail64_32
def innerCandidates64_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates64_31_checked : candidates edge 4 331 7417 innerTail64_32 best.bound = innerCandidates64_31 := by
  unfold candidates
  have row : edge 7417 = SearchData.row7417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_31 : scan edge (search edge 3) 4 [331] 331 innerTail64_31 best = best := by
  rw [innerTail64_31, scan, if_neg (by decide), innerCandidates64_31_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_32
    (if 3 ≤ innerCandidates64_31.length then search edge 3 [331, 7417] innerCandidates64_31 7748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_32

def innerTail64_30 : List Nat := 6949 :: innerTail64_31
def innerCandidates64_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates64_30_checked : candidates edge 4 331 6949 innerTail64_31 best.bound = innerCandidates64_30 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_30 : scan edge (search edge 3) 4 [331] 331 innerTail64_30 best = best := by
  rw [innerTail64_30, scan, if_neg (by decide), innerCandidates64_30_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_31
    (if 3 ≤ innerCandidates64_30.length then search edge 3 [331, 6949] innerCandidates64_30 7280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_31

def innerTail64_29 : List Nat := 6553 :: innerTail64_30
def innerCandidates64_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates64_29_checked : candidates edge 4 331 6553 innerTail64_30 best.bound = innerCandidates64_29 := by
  unfold candidates
  have row : edge 6553 = SearchData.row6553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_29 : scan edge (search edge 3) 4 [331] 331 innerTail64_29 best = best := by
  rw [innerTail64_29, scan, if_neg (by decide), innerCandidates64_29_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_30
    (if 3 ≤ innerCandidates64_29.length then search edge 3 [331, 6553] innerCandidates64_29 6884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_30

def innerTail64_28 : List Nat := 6451 :: innerTail64_29
def innerCandidates64_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates64_28_checked : candidates edge 4 331 6451 innerTail64_29 best.bound = innerCandidates64_28 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_28 : scan edge (search edge 3) 4 [331] 331 innerTail64_28 best = best := by
  rw [innerTail64_28, scan, if_neg (by decide), innerCandidates64_28_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_29
    (if 3 ≤ innerCandidates64_28.length then search edge 3 [331, 6451] innerCandidates64_28 6782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_29

def innerTail64_27 : List Nat := 6427 :: innerTail64_28
def innerCandidates64_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates64_27_checked : candidates edge 4 331 6427 innerTail64_28 best.bound = innerCandidates64_27 := by
  unfold candidates
  have row : edge 6427 = SearchData.row6427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_27 : scan edge (search edge 3) 4 [331] 331 innerTail64_27 best = best := by
  rw [innerTail64_27, scan, if_neg (by decide), innerCandidates64_27_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_28
    (if 3 ≤ innerCandidates64_27.length then search edge 3 [331, 6427] innerCandidates64_27 6758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_28

def innerTail64_26 : List Nat := 6361 :: innerTail64_27
def innerCandidates64_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates64_26_checked : candidates edge 4 331 6361 innerTail64_27 best.bound = innerCandidates64_26 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_26 : scan edge (search edge 3) 4 [331] 331 innerTail64_26 best = best := by
  rw [innerTail64_26, scan, if_neg (by decide), innerCandidates64_26_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_27
    (if 3 ≤ innerCandidates64_26.length then search edge 3 [331, 6361] innerCandidates64_26 6692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_27

def innerTail64_25 : List Nat := 5779 :: innerTail64_26
def innerCandidates64_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates64_25_checked : candidates edge 4 331 5779 innerTail64_26 best.bound = innerCandidates64_25 := by
  unfold candidates
  have row : edge 5779 = SearchData.row5779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_25 : scan edge (search edge 3) 4 [331] 331 innerTail64_25 best = best := by
  rw [innerTail64_25, scan, if_neg (by decide), innerCandidates64_25_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_26
    (if 3 ≤ innerCandidates64_25.length then search edge 3 [331, 5779] innerCandidates64_25 6110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_26

def innerTail64_24 : List Nat := 5407 :: innerTail64_25
def innerCandidates64_24 : List Nat := [5779]
set_option maxRecDepth 4096 in
theorem innerCandidates64_24_checked : candidates edge 4 331 5407 innerTail64_25 best.bound = innerCandidates64_24 := by
  unfold candidates
  have row : edge 5407 = SearchData.row5407.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_24 : scan edge (search edge 3) 4 [331] 331 innerTail64_24 best = best := by
  rw [innerTail64_24, scan, if_neg (by decide), innerCandidates64_24_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_25
    (if 3 ≤ innerCandidates64_24.length then search edge 3 [331, 5407] innerCandidates64_24 5738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_25

def innerTail64_23 : List Nat := 5107 :: innerTail64_24
def innerCandidates64_23 : List Nat := [5779]
set_option maxRecDepth 4096 in
theorem innerCandidates64_23_checked : candidates edge 4 331 5107 innerTail64_24 best.bound = innerCandidates64_23 := by
  unfold candidates
  have row : edge 5107 = SearchData.row5107.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_23 : scan edge (search edge 3) 4 [331] 331 innerTail64_23 best = best := by
  rw [innerTail64_23, scan, if_neg (by decide), innerCandidates64_23_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_24
    (if 3 ≤ innerCandidates64_23.length then search edge 3 [331, 5107] innerCandidates64_23 5438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_24

def innerTail64_22 : List Nat := 4651 :: innerTail64_23
def innerCandidates64_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates64_22_checked : candidates edge 4 331 4651 innerTail64_23 best.bound = innerCandidates64_22 := by
  unfold candidates
  have row : edge 4651 = SearchData.row4651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_22 : scan edge (search edge 3) 4 [331] 331 innerTail64_22 best = best := by
  rw [innerTail64_22, scan, if_neg (by decide), innerCandidates64_22_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_23
    (if 3 ≤ innerCandidates64_22.length then search edge 3 [331, 4651] innerCandidates64_22 4982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_23

def innerTail64_21 : List Nat := 4639 :: innerTail64_22
def innerCandidates64_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates64_21_checked : candidates edge 4 331 4639 innerTail64_22 best.bound = innerCandidates64_21 := by
  unfold candidates
  have row : edge 4639 = SearchData.row4639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_21 : scan edge (search edge 3) 4 [331] 331 innerTail64_21 best = best := by
  rw [innerTail64_21, scan, if_neg (by decide), innerCandidates64_21_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_22
    (if 3 ≤ innerCandidates64_21.length then search edge 3 [331, 4639] innerCandidates64_21 4970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_22

def innerTail64_20 : List Nat := 4513 :: innerTail64_21
def innerCandidates64_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates64_20_checked : candidates edge 4 331 4513 innerTail64_21 best.bound = innerCandidates64_20 := by
  unfold candidates
  have row : edge 4513 = SearchData.row4513.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_20 : scan edge (search edge 3) 4 [331] 331 innerTail64_20 best = best := by
  rw [innerTail64_20, scan, if_neg (by decide), innerCandidates64_20_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_21
    (if 3 ≤ innerCandidates64_20.length then search edge 3 [331, 4513] innerCandidates64_20 4844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_21

def innerTail64_19 : List Nat := 4261 :: innerTail64_20
def innerCandidates64_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates64_19_checked : candidates edge 4 331 4261 innerTail64_20 best.bound = innerCandidates64_19 := by
  unfold candidates
  have row : edge 4261 = SearchData.row4261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_19 : scan edge (search edge 3) 4 [331] 331 innerTail64_19 best = best := by
  rw [innerTail64_19, scan, if_neg (by decide), innerCandidates64_19_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_20
    (if 3 ≤ innerCandidates64_19.length then search edge 3 [331, 4261] innerCandidates64_19 4592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_20

def innerTail64_18 : List Nat := 3769 :: innerTail64_19
def innerCandidates64_18 : List Nat := [6949]
set_option maxRecDepth 4096 in
theorem innerCandidates64_18_checked : candidates edge 4 331 3769 innerTail64_19 best.bound = innerCandidates64_18 := by
  unfold candidates
  have row : edge 3769 = SearchData.row3769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_18 : scan edge (search edge 3) 4 [331] 331 innerTail64_18 best = best := by
  rw [innerTail64_18, scan, if_neg (by decide), innerCandidates64_18_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_19
    (if 3 ≤ innerCandidates64_18.length then search edge 3 [331, 3769] innerCandidates64_18 4100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_19

def innerTail64_17 : List Nat := 3637 :: innerTail64_18
def innerCandidates64_17 : List Nat := [5407]
set_option maxRecDepth 4096 in
theorem innerCandidates64_17_checked : candidates edge 4 331 3637 innerTail64_18 best.bound = innerCandidates64_17 := by
  unfold candidates
  have row : edge 3637 = SearchData.row3637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_17 : scan edge (search edge 3) 4 [331] 331 innerTail64_17 best = best := by
  rw [innerTail64_17, scan, if_neg (by decide), innerCandidates64_17_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_18
    (if 3 ≤ innerCandidates64_17.length then search edge 3 [331, 3637] innerCandidates64_17 3968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_18

def innerTail64_16 : List Nat := 3361 :: innerTail64_17
def innerCandidates64_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates64_16_checked : candidates edge 4 331 3361 innerTail64_17 best.bound = innerCandidates64_16 := by
  unfold candidates
  have row : edge 3361 = SearchData.row3361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_16 : scan edge (search edge 3) 4 [331] 331 innerTail64_16 best = best := by
  rw [innerTail64_16, scan, if_neg (by decide), innerCandidates64_16_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_17
    (if 3 ≤ innerCandidates64_16.length then search edge 3 [331, 3361] innerCandidates64_16 3692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_17

def innerTail64_15 : List Nat := 3067 :: innerTail64_16
def innerCandidates64_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates64_15_checked : candidates edge 4 331 3067 innerTail64_16 best.bound = innerCandidates64_15 := by
  unfold candidates
  have row : edge 3067 = SearchData.row3067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_15 : scan edge (search edge 3) 4 [331] 331 innerTail64_15 best = best := by
  rw [innerTail64_15, scan, if_neg (by decide), innerCandidates64_15_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_16
    (if 3 ≤ innerCandidates64_15.length then search edge 3 [331, 3067] innerCandidates64_15 3398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_16

def innerTail64_14 : List Nat := 2833 :: innerTail64_15
def innerCandidates64_14 : List Nat := [6361]
set_option maxRecDepth 4096 in
theorem innerCandidates64_14_checked : candidates edge 4 331 2833 innerTail64_15 best.bound = innerCandidates64_14 := by
  unfold candidates
  have row : edge 2833 = SearchData.row2833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_14 : scan edge (search edge 3) 4 [331] 331 innerTail64_14 best = best := by
  rw [innerTail64_14, scan, if_neg (by decide), innerCandidates64_14_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_15
    (if 3 ≤ innerCandidates64_14.length then search edge 3 [331, 2833] innerCandidates64_14 3164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_15

def innerTail64_13 : List Nat := 2731 :: innerTail64_14
def innerCandidates64_13 : List Nat := [6451]
set_option maxRecDepth 4096 in
theorem innerCandidates64_13_checked : candidates edge 4 331 2731 innerTail64_14 best.bound = innerCandidates64_13 := by
  unfold candidates
  have row : edge 2731 = SearchData.row2731.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_13 : scan edge (search edge 3) 4 [331] 331 innerTail64_13 best = best := by
  rw [innerTail64_13, scan, if_neg (by decide), innerCandidates64_13_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_14
    (if 3 ≤ innerCandidates64_13.length then search edge 3 [331, 2731] innerCandidates64_13 3062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_14

def innerTail64_12 : List Nat := 2719 :: innerTail64_13
def innerCandidates64_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates64_12_checked : candidates edge 4 331 2719 innerTail64_13 best.bound = innerCandidates64_12 := by
  unfold candidates
  have row : edge 2719 = SearchData.row2719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_12 : scan edge (search edge 3) 4 [331] 331 innerTail64_12 best = best := by
  rw [innerTail64_12, scan, if_neg (by decide), innerCandidates64_12_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_13
    (if 3 ≤ innerCandidates64_12.length then search edge 3 [331, 2719] innerCandidates64_12 3050 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_13

def innerTail64_11 : List Nat := 1999 :: innerTail64_12
def innerCandidates64_11 : List Nat := [3361]
set_option maxRecDepth 4096 in
theorem innerCandidates64_11_checked : candidates edge 4 331 1999 innerTail64_12 best.bound = innerCandidates64_11 := by
  unfold candidates
  have row : edge 1999 = PrimePairCache.smallRow1999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_11 : scan edge (search edge 3) 4 [331] 331 innerTail64_11 best = best := by
  rw [innerTail64_11, scan, if_neg (by decide), innerCandidates64_11_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_12
    (if 3 ≤ innerCandidates64_11.length then search edge 3 [331, 1999] innerCandidates64_11 2330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_12

def innerTail64_10 : List Nat := 1933 :: innerTail64_11
def innerCandidates64_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates64_10_checked : candidates edge 4 331 1933 innerTail64_11 best.bound = innerCandidates64_10 := by
  unfold candidates
  have row : edge 1933 = PrimePairCache.smallRow1933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_10 : scan edge (search edge 3) 4 [331] 331 innerTail64_10 best = best := by
  rw [innerTail64_10, scan, if_neg (by decide), innerCandidates64_10_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_11
    (if 3 ≤ innerCandidates64_10.length then search edge 3 [331, 1933] innerCandidates64_10 2264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_11

def innerTail64_9 : List Nat := 1831 :: innerTail64_10
def innerCandidates64_9 : List Nat := [3769]
set_option maxRecDepth 4096 in
theorem innerCandidates64_9_checked : candidates edge 4 331 1831 innerTail64_10 best.bound = innerCandidates64_9 := by
  unfold candidates
  have row : edge 1831 = PrimePairCache.smallRow1831.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_9 : scan edge (search edge 3) 4 [331] 331 innerTail64_9 best = best := by
  rw [innerTail64_9, scan, if_neg (by decide), innerCandidates64_9_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_10
    (if 3 ≤ innerCandidates64_9.length then search edge 3 [331, 1831] innerCandidates64_9 2162 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_10

def innerTail64_8 : List Nat := 1669 :: innerTail64_9
def innerCandidates64_8 : List Nat := [4651]
set_option maxRecDepth 4096 in
theorem innerCandidates64_8_checked : candidates edge 4 331 1669 innerTail64_9 best.bound = innerCandidates64_8 := by
  unfold candidates
  have row : edge 1669 = PrimePairCache.smallRow1669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_8 : scan edge (search edge 3) 4 [331] 331 innerTail64_8 best = best := by
  rw [innerTail64_8, scan, if_neg (by decide), innerCandidates64_8_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_9
    (if 3 ≤ innerCandidates64_8.length then search edge 3 [331, 1669] innerCandidates64_8 2000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_9

def innerTail64_7 : List Nat := 1531 :: innerTail64_8
def innerCandidates64_7 : List Nat := [6361, 7537]
set_option maxRecDepth 4096 in
theorem innerCandidates64_7_checked : candidates edge 4 331 1531 innerTail64_8 best.bound = innerCandidates64_7 := by
  unfold candidates
  have row : edge 1531 = PrimePairCache.smallRow1531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_7 : scan edge (search edge 3) 4 [331] 331 innerTail64_7 best = best := by
  rw [innerTail64_7, scan, if_neg (by decide), innerCandidates64_7_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_8
    (if 3 ≤ innerCandidates64_7.length then search edge 3 [331, 1531] innerCandidates64_7 1862 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_8

def innerTail64_6 : List Nat := 1303 :: innerTail64_7
def innerCandidates64_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates64_6_checked : candidates edge 4 331 1303 innerTail64_7 best.bound = innerCandidates64_6 := by
  unfold candidates
  have row : edge 1303 = PrimePairCache.smallRow1303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_6 : scan edge (search edge 3) 4 [331] 331 innerTail64_6 best = best := by
  rw [innerTail64_6, scan, if_neg (by decide), innerCandidates64_6_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_7
    (if 3 ≤ innerCandidates64_6.length then search edge 3 [331, 1303] innerCandidates64_6 1634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_7

def innerTail64_5 : List Nat := 937 :: innerTail64_6
def innerCandidates64_5 : List Nat := [5407]
set_option maxRecDepth 4096 in
theorem innerCandidates64_5_checked : candidates edge 4 331 937 innerTail64_6 best.bound = innerCandidates64_5 := by
  unfold candidates
  have row : edge 937 = PrimePairCache.smallRow937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_5 : scan edge (search edge 3) 4 [331] 331 innerTail64_5 best = best := by
  rw [innerTail64_5, scan, if_neg (by decide), innerCandidates64_5_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_6
    (if 3 ≤ innerCandidates64_5.length then search edge 3 [331, 937] innerCandidates64_5 1268 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_6

def innerTail64_4 : List Nat := 907 :: innerTail64_5
def innerCandidates64_4 : List Nat := [2731]
set_option maxRecDepth 4096 in
theorem innerCandidates64_4_checked : candidates edge 4 331 907 innerTail64_5 best.bound = innerCandidates64_4 := by
  unfold candidates
  have row : edge 907 = PrimePairCache.smallRow907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_4 : scan edge (search edge 3) 4 [331] 331 innerTail64_4 best = best := by
  rw [innerTail64_4, scan, if_neg (by decide), innerCandidates64_4_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_5
    (if 3 ≤ innerCandidates64_4.length then search edge 3 [331, 907] innerCandidates64_4 1238 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_5

def innerTail64_3 : List Nat := 883 :: innerTail64_4
def innerCandidates64_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates64_3_checked : candidates edge 4 331 883 innerTail64_4 best.bound = innerCandidates64_3 := by
  unfold candidates
  have row : edge 883 = PrimePairCache.smallRow883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_3 : scan edge (search edge 3) 4 [331] 331 innerTail64_3 best = best := by
  rw [innerTail64_3, scan, if_neg (by decide), innerCandidates64_3_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_4
    (if 3 ≤ innerCandidates64_3.length then search edge 3 [331, 883] innerCandidates64_3 1214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_4

def innerTail64_2 : List Nat := 739 :: innerTail64_3
def innerCandidates64_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates64_2_checked : candidates edge 4 331 739 innerTail64_3 best.bound = innerCandidates64_2 := by
  unfold candidates
  have row : edge 739 = PrimePairCache.smallRow739.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_2 : scan edge (search edge 3) 4 [331] 331 innerTail64_2 best = best := by
  rw [innerTail64_2, scan, if_neg (by decide), innerCandidates64_2_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_3
    (if 3 ≤ innerCandidates64_2.length then search edge 3 [331, 739] innerCandidates64_2 1070 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_3

def innerTail64_1 : List Nat := 577 :: innerTail64_2
def innerCandidates64_1 : List Nat := [937, 6949]
set_option maxRecDepth 4096 in
theorem innerCandidates64_1_checked : candidates edge 4 331 577 innerTail64_2 best.bound = innerCandidates64_1 := by
  unfold candidates
  have row : edge 577 = PrimePairCache.smallRow577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_1 : scan edge (search edge 3) 4 [331] 331 innerTail64_1 best = best := by
  rw [innerTail64_1, scan, if_neg (by decide), innerCandidates64_1_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_2
    (if 3 ≤ innerCandidates64_1.length then search edge 3 [331, 577] innerCandidates64_1 908 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_2

def innerTail64_0 : List Nat := 349 :: innerTail64_1
def innerCandidates64_0 : List Nat := [2719, 8377]
set_option maxRecDepth 4096 in
theorem innerCandidates64_0_checked : candidates edge 4 331 349 innerTail64_1 best.bound = innerCandidates64_0 := by
  unfold candidates
  have row : edge 349 = PrimePairCache.smallRow349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan64_0 : scan edge (search edge 3) 4 [331] 331 innerTail64_0 best = best := by
  rw [innerTail64_0, scan, if_neg (by decide), innerCandidates64_0_checked]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_1
    (if 3 ≤ innerCandidates64_0.length then search edge 3 [331, 349] innerCandidates64_0 680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan64_1

theorem rootBranch64 : search edge 4 [331] rootCandidates64 331 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [331] 331 innerTail64_0 best = best
  exact innerScan64_0

def rootCandidates63 : List Nat := [353, 419, 503, 773, 983, 1187, 1601, 1697, 1709, 2027, 2237, 2243, 2399, 2441, 2801, 3119, 3539, 3659, 4091, 4793, 4799, 4973, 5009, 5099, 5801, 6053, 6221, 7607, 7949]
set_option maxRecDepth 4096 in
theorem rootCandidates63_checked : candidates edge 5 0 317 tail64 best.bound = rootCandidates63 := by
  unfold candidates
  have row : edge 317 = PrimePairCache.smallRow317.contains := by funext q; rfl
  rw [row]
  decide

def innerTail63_29 : List Nat := []
theorem innerScan63_29 : scan edge (search edge 3) 4 [317] 317 innerTail63_29 best = best := rfl

def innerTail63_28 : List Nat := 7949 :: innerTail63_29
def innerCandidates63_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates63_28_checked : candidates edge 4 317 7949 innerTail63_29 best.bound = innerCandidates63_28 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_28 : scan edge (search edge 3) 4 [317] 317 innerTail63_28 best = best := by
  rw [innerTail63_28, scan, if_neg (by decide), innerCandidates63_28_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_29
    (if 3 ≤ innerCandidates63_28.length then search edge 3 [317, 7949] innerCandidates63_28 8266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_29

def innerTail63_27 : List Nat := 7607 :: innerTail63_28
def innerCandidates63_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates63_27_checked : candidates edge 4 317 7607 innerTail63_28 best.bound = innerCandidates63_27 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_27 : scan edge (search edge 3) 4 [317] 317 innerTail63_27 best = best := by
  rw [innerTail63_27, scan, if_neg (by decide), innerCandidates63_27_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_28
    (if 3 ≤ innerCandidates63_27.length then search edge 3 [317, 7607] innerCandidates63_27 7924 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_28

def innerTail63_26 : List Nat := 6221 :: innerTail63_27
def innerCandidates63_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates63_26_checked : candidates edge 4 317 6221 innerTail63_27 best.bound = innerCandidates63_26 := by
  unfold candidates
  have row : edge 6221 = SearchData.row6221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_26 : scan edge (search edge 3) 4 [317] 317 innerTail63_26 best = best := by
  rw [innerTail63_26, scan, if_neg (by decide), innerCandidates63_26_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_27
    (if 3 ≤ innerCandidates63_26.length then search edge 3 [317, 6221] innerCandidates63_26 6538 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_27

def innerTail63_25 : List Nat := 6053 :: innerTail63_26
def innerCandidates63_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates63_25_checked : candidates edge 4 317 6053 innerTail63_26 best.bound = innerCandidates63_25 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_25 : scan edge (search edge 3) 4 [317] 317 innerTail63_25 best = best := by
  rw [innerTail63_25, scan, if_neg (by decide), innerCandidates63_25_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_26
    (if 3 ≤ innerCandidates63_25.length then search edge 3 [317, 6053] innerCandidates63_25 6370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_26

def innerTail63_24 : List Nat := 5801 :: innerTail63_25
def innerCandidates63_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates63_24_checked : candidates edge 4 317 5801 innerTail63_25 best.bound = innerCandidates63_24 := by
  unfold candidates
  have row : edge 5801 = SearchData.row5801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_24 : scan edge (search edge 3) 4 [317] 317 innerTail63_24 best = best := by
  rw [innerTail63_24, scan, if_neg (by decide), innerCandidates63_24_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_25
    (if 3 ≤ innerCandidates63_24.length then search edge 3 [317, 5801] innerCandidates63_24 6118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_25

def innerTail63_23 : List Nat := 5099 :: innerTail63_24
def innerCandidates63_23 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates63_23_checked : candidates edge 4 317 5099 innerTail63_24 best.bound = innerCandidates63_23 := by
  unfold candidates
  have row : edge 5099 = SearchData.row5099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_23 : scan edge (search edge 3) 4 [317] 317 innerTail63_23 best = best := by
  rw [innerTail63_23, scan, if_neg (by decide), innerCandidates63_23_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_24
    (if 3 ≤ innerCandidates63_23.length then search edge 3 [317, 5099] innerCandidates63_23 5416 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_24

def innerTail63_22 : List Nat := 5009 :: innerTail63_23
def innerCandidates63_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates63_22_checked : candidates edge 4 317 5009 innerTail63_23 best.bound = innerCandidates63_22 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_22 : scan edge (search edge 3) 4 [317] 317 innerTail63_22 best = best := by
  rw [innerTail63_22, scan, if_neg (by decide), innerCandidates63_22_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_23
    (if 3 ≤ innerCandidates63_22.length then search edge 3 [317, 5009] innerCandidates63_22 5326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_23

def innerTail63_21 : List Nat := 4973 :: innerTail63_22
def innerCandidates63_21 : List Nat := [6221]
set_option maxRecDepth 4096 in
theorem innerCandidates63_21_checked : candidates edge 4 317 4973 innerTail63_22 best.bound = innerCandidates63_21 := by
  unfold candidates
  have row : edge 4973 = SearchData.row4973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_21 : scan edge (search edge 3) 4 [317] 317 innerTail63_21 best = best := by
  rw [innerTail63_21, scan, if_neg (by decide), innerCandidates63_21_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_22
    (if 3 ≤ innerCandidates63_21.length then search edge 3 [317, 4973] innerCandidates63_21 5290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_22

def innerTail63_20 : List Nat := 4799 :: innerTail63_21
def innerCandidates63_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates63_20_checked : candidates edge 4 317 4799 innerTail63_21 best.bound = innerCandidates63_20 := by
  unfold candidates
  have row : edge 4799 = SearchData.row4799.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_20 : scan edge (search edge 3) 4 [317] 317 innerTail63_20 best = best := by
  rw [innerTail63_20, scan, if_neg (by decide), innerCandidates63_20_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_21
    (if 3 ≤ innerCandidates63_20.length then search edge 3 [317, 4799] innerCandidates63_20 5116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_21

def innerTail63_19 : List Nat := 4793 :: innerTail63_20
def innerCandidates63_19 : List Nat := [5099]
set_option maxRecDepth 4096 in
theorem innerCandidates63_19_checked : candidates edge 4 317 4793 innerTail63_20 best.bound = innerCandidates63_19 := by
  unfold candidates
  have row : edge 4793 = SearchData.row4793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_19 : scan edge (search edge 3) 4 [317] 317 innerTail63_19 best = best := by
  rw [innerTail63_19, scan, if_neg (by decide), innerCandidates63_19_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_20
    (if 3 ≤ innerCandidates63_19.length then search edge 3 [317, 4793] innerCandidates63_19 5110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_20

def innerTail63_18 : List Nat := 4091 :: innerTail63_19
def innerCandidates63_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates63_18_checked : candidates edge 4 317 4091 innerTail63_19 best.bound = innerCandidates63_18 := by
  unfold candidates
  have row : edge 4091 = SearchData.row4091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_18 : scan edge (search edge 3) 4 [317] 317 innerTail63_18 best = best := by
  rw [innerTail63_18, scan, if_neg (by decide), innerCandidates63_18_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_19
    (if 3 ≤ innerCandidates63_18.length then search edge 3 [317, 4091] innerCandidates63_18 4408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_19

def innerTail63_17 : List Nat := 3659 :: innerTail63_18
def innerCandidates63_17 : List Nat := [4091]
set_option maxRecDepth 4096 in
theorem innerCandidates63_17_checked : candidates edge 4 317 3659 innerTail63_18 best.bound = innerCandidates63_17 := by
  unfold candidates
  have row : edge 3659 = SearchData.row3659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_17 : scan edge (search edge 3) 4 [317] 317 innerTail63_17 best = best := by
  rw [innerTail63_17, scan, if_neg (by decide), innerCandidates63_17_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_18
    (if 3 ≤ innerCandidates63_17.length then search edge 3 [317, 3659] innerCandidates63_17 3976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_18

def innerTail63_16 : List Nat := 3539 :: innerTail63_17
def innerCandidates63_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates63_16_checked : candidates edge 4 317 3539 innerTail63_17 best.bound = innerCandidates63_16 := by
  unfold candidates
  have row : edge 3539 = SearchData.row3539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_16 : scan edge (search edge 3) 4 [317] 317 innerTail63_16 best = best := by
  rw [innerTail63_16, scan, if_neg (by decide), innerCandidates63_16_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_17
    (if 3 ≤ innerCandidates63_16.length then search edge 3 [317, 3539] innerCandidates63_16 3856 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_17

def innerTail63_15 : List Nat := 3119 :: innerTail63_16
def innerCandidates63_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates63_15_checked : candidates edge 4 317 3119 innerTail63_16 best.bound = innerCandidates63_15 := by
  unfold candidates
  have row : edge 3119 = SearchData.row3119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_15 : scan edge (search edge 3) 4 [317] 317 innerTail63_15 best = best := by
  rw [innerTail63_15, scan, if_neg (by decide), innerCandidates63_15_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_16
    (if 3 ≤ innerCandidates63_15.length then search edge 3 [317, 3119] innerCandidates63_15 3436 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_16

def innerTail63_14 : List Nat := 2801 :: innerTail63_15
def innerCandidates63_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates63_14_checked : candidates edge 4 317 2801 innerTail63_15 best.bound = innerCandidates63_14 := by
  unfold candidates
  have row : edge 2801 = SearchData.row2801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_14 : scan edge (search edge 3) 4 [317] 317 innerTail63_14 best = best := by
  rw [innerTail63_14, scan, if_neg (by decide), innerCandidates63_14_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_15
    (if 3 ≤ innerCandidates63_14.length then search edge 3 [317, 2801] innerCandidates63_14 3118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_15

def innerTail63_13 : List Nat := 2441 :: innerTail63_14
def innerCandidates63_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates63_13_checked : candidates edge 4 317 2441 innerTail63_14 best.bound = innerCandidates63_13 := by
  unfold candidates
  have row : edge 2441 = PrimePairCache.smallRow2441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_13 : scan edge (search edge 3) 4 [317] 317 innerTail63_13 best = best := by
  rw [innerTail63_13, scan, if_neg (by decide), innerCandidates63_13_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_14
    (if 3 ≤ innerCandidates63_13.length then search edge 3 [317, 2441] innerCandidates63_13 2758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_14

def innerTail63_12 : List Nat := 2399 :: innerTail63_13
def innerCandidates63_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates63_12_checked : candidates edge 4 317 2399 innerTail63_13 best.bound = innerCandidates63_12 := by
  unfold candidates
  have row : edge 2399 = PrimePairCache.smallRow2399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_12 : scan edge (search edge 3) 4 [317] 317 innerTail63_12 best = best := by
  rw [innerTail63_12, scan, if_neg (by decide), innerCandidates63_12_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_13
    (if 3 ≤ innerCandidates63_12.length then search edge 3 [317, 2399] innerCandidates63_12 2716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_13

def innerTail63_11 : List Nat := 2243 :: innerTail63_12
def innerCandidates63_11 : List Nat := [3539]
set_option maxRecDepth 4096 in
theorem innerCandidates63_11_checked : candidates edge 4 317 2243 innerTail63_12 best.bound = innerCandidates63_11 := by
  unfold candidates
  have row : edge 2243 = PrimePairCache.smallRow2243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_11 : scan edge (search edge 3) 4 [317] 317 innerTail63_11 best = best := by
  rw [innerTail63_11, scan, if_neg (by decide), innerCandidates63_11_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_12
    (if 3 ≤ innerCandidates63_11.length then search edge 3 [317, 2243] innerCandidates63_11 2560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_12

def innerTail63_10 : List Nat := 2237 :: innerTail63_11
def innerCandidates63_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates63_10_checked : candidates edge 4 317 2237 innerTail63_11 best.bound = innerCandidates63_10 := by
  unfold candidates
  have row : edge 2237 = PrimePairCache.smallRow2237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_10 : scan edge (search edge 3) 4 [317] 317 innerTail63_10 best = best := by
  rw [innerTail63_10, scan, if_neg (by decide), innerCandidates63_10_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_11
    (if 3 ≤ innerCandidates63_10.length then search edge 3 [317, 2237] innerCandidates63_10 2554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_11

def innerTail63_9 : List Nat := 2027 :: innerTail63_10
def innerCandidates63_9 : List Nat := [3119]
set_option maxRecDepth 4096 in
theorem innerCandidates63_9_checked : candidates edge 4 317 2027 innerTail63_10 best.bound = innerCandidates63_9 := by
  unfold candidates
  have row : edge 2027 = PrimePairCache.smallRow2027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_9 : scan edge (search edge 3) 4 [317] 317 innerTail63_9 best = best := by
  rw [innerTail63_9, scan, if_neg (by decide), innerCandidates63_9_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_10
    (if 3 ≤ innerCandidates63_9.length then search edge 3 [317, 2027] innerCandidates63_9 2344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_10

def innerTail63_8 : List Nat := 1709 :: innerTail63_9
def innerCandidates63_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates63_8_checked : candidates edge 4 317 1709 innerTail63_9 best.bound = innerCandidates63_8 := by
  unfold candidates
  have row : edge 1709 = PrimePairCache.smallRow1709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_8 : scan edge (search edge 3) 4 [317] 317 innerTail63_8 best = best := by
  rw [innerTail63_8, scan, if_neg (by decide), innerCandidates63_8_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_9
    (if 3 ≤ innerCandidates63_8.length then search edge 3 [317, 1709] innerCandidates63_8 2026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_9

def innerTail63_7 : List Nat := 1697 :: innerTail63_8
def innerCandidates63_7 : List Nat := [2027, 5099]
set_option maxRecDepth 4096 in
theorem innerCandidates63_7_checked : candidates edge 4 317 1697 innerTail63_8 best.bound = innerCandidates63_7 := by
  unfold candidates
  have row : edge 1697 = PrimePairCache.smallRow1697.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_7 : scan edge (search edge 3) 4 [317] 317 innerTail63_7 best = best := by
  rw [innerTail63_7, scan, if_neg (by decide), innerCandidates63_7_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_8
    (if 3 ≤ innerCandidates63_7.length then search edge 3 [317, 1697] innerCandidates63_7 2014 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_8

def innerTail63_6 : List Nat := 1601 :: innerTail63_7
def innerCandidates63_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates63_6_checked : candidates edge 4 317 1601 innerTail63_7 best.bound = innerCandidates63_6 := by
  unfold candidates
  have row : edge 1601 = PrimePairCache.smallRow1601.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_6 : scan edge (search edge 3) 4 [317] 317 innerTail63_6 best = best := by
  rw [innerTail63_6, scan, if_neg (by decide), innerCandidates63_6_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_7
    (if 3 ≤ innerCandidates63_6.length then search edge 3 [317, 1601] innerCandidates63_6 1918 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_7

def innerTail63_5 : List Nat := 1187 :: innerTail63_6
def innerCandidates63_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates63_5_checked : candidates edge 4 317 1187 innerTail63_6 best.bound = innerCandidates63_5 := by
  unfold candidates
  have row : edge 1187 = PrimePairCache.smallRow1187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_5 : scan edge (search edge 3) 4 [317] 317 innerTail63_5 best = best := by
  rw [innerTail63_5, scan, if_neg (by decide), innerCandidates63_5_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_6
    (if 3 ≤ innerCandidates63_5.length then search edge 3 [317, 1187] innerCandidates63_5 1504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_6

def innerTail63_4 : List Nat := 983 :: innerTail63_5
def innerCandidates63_4 : List Nat := [2399]
set_option maxRecDepth 4096 in
theorem innerCandidates63_4_checked : candidates edge 4 317 983 innerTail63_5 best.bound = innerCandidates63_4 := by
  unfold candidates
  have row : edge 983 = PrimePairCache.smallRow983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_4 : scan edge (search edge 3) 4 [317] 317 innerTail63_4 best = best := by
  rw [innerTail63_4, scan, if_neg (by decide), innerCandidates63_4_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_5
    (if 3 ≤ innerCandidates63_4.length then search edge 3 [317, 983] innerCandidates63_4 1300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_5

def innerTail63_3 : List Nat := 773 :: innerTail63_4
def innerCandidates63_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates63_3_checked : candidates edge 4 317 773 innerTail63_4 best.bound = innerCandidates63_3 := by
  unfold candidates
  have row : edge 773 = PrimePairCache.smallRow773.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_3 : scan edge (search edge 3) 4 [317] 317 innerTail63_3 best = best := by
  rw [innerTail63_3, scan, if_neg (by decide), innerCandidates63_3_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_4
    (if 3 ≤ innerCandidates63_3.length then search edge 3 [317, 773] innerCandidates63_3 1090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_4

def innerTail63_2 : List Nat := 503 :: innerTail63_3
def innerCandidates63_2 : List Nat := [1601]
set_option maxRecDepth 4096 in
theorem innerCandidates63_2_checked : candidates edge 4 317 503 innerTail63_3 best.bound = innerCandidates63_2 := by
  unfold candidates
  have row : edge 503 = PrimePairCache.smallRow503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_2 : scan edge (search edge 3) 4 [317] 317 innerTail63_2 best = best := by
  rw [innerTail63_2, scan, if_neg (by decide), innerCandidates63_2_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_3
    (if 3 ≤ innerCandidates63_2.length then search edge 3 [317, 503] innerCandidates63_2 820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_3

def innerTail63_1 : List Nat := 419 :: innerTail63_2
def innerCandidates63_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates63_1_checked : candidates edge 4 317 419 innerTail63_2 best.bound = innerCandidates63_1 := by
  unfold candidates
  have row : edge 419 = PrimePairCache.smallRow419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_1 : scan edge (search edge 3) 4 [317] 317 innerTail63_1 best = best := by
  rw [innerTail63_1, scan, if_neg (by decide), innerCandidates63_1_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_2
    (if 3 ≤ innerCandidates63_1.length then search edge 3 [317, 419] innerCandidates63_1 736 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_2

def innerTail63_0 : List Nat := 353 :: innerTail63_1
def innerCandidates63_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates63_0_checked : candidates edge 4 317 353 innerTail63_1 best.bound = innerCandidates63_0 := by
  unfold candidates
  have row : edge 353 = PrimePairCache.smallRow353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan63_0 : scan edge (search edge 3) 4 [317] 317 innerTail63_0 best = best := by
  rw [innerTail63_0, scan, if_neg (by decide), innerCandidates63_0_checked]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_1
    (if 3 ≤ innerCandidates63_0.length then search edge 3 [317, 353] innerCandidates63_0 670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan63_1

theorem rootBranch63 : search edge 4 [317] rootCandidates63 317 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [317] 317 innerTail63_0 best = best
  exact innerScan63_0

end ProjectEulerSolutions.PrimePairUpper.Search
