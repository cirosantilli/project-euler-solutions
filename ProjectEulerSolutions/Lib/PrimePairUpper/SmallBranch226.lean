import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates51 : List Nat := [347, 431, 491, 761, 941, 971, 1151, 1433, 1709, 1979, 2063, 2207, 2417, 2789, 2801, 2999, 3011, 3347, 3557, 3803, 4019, 4877, 5147, 5189, 5237, 5273, 5651, 5897, 6197, 6473, 6581, 6863, 7079, 7127, 7499, 7901, 7919, 8093, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates51_checked : candidates edge 5 0 251 tail52 best.bound = rootCandidates51 := by
  unfold candidates
  have row : edge 251 = PrimePairCache.smallRow251.contains := by funext q; rfl
  rw [row]
  decide

def innerTail51_39 : List Nat := []
theorem innerScan51_39 : scan edge (search edge 3) 4 [251] 251 innerTail51_39 best = best := rfl

def innerTail51_38 : List Nat := 8387 :: innerTail51_39
def innerCandidates51_38 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_38_checked : candidates edge 4 251 8387 innerTail51_39 best.bound = innerCandidates51_38 := by
  unfold candidates
  have row : edge 8387 = SearchData.row8387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_38 : scan edge (search edge 3) 4 [251] 251 innerTail51_38 best = best := by
  rw [innerTail51_38, scan, if_neg (by decide), innerCandidates51_38_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_39
    (if 3 ≤ innerCandidates51_38.length then search edge 3 [251, 8387] innerCandidates51_38 8638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_39

def innerTail51_37 : List Nat := 8093 :: innerTail51_38
def innerCandidates51_37 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_37_checked : candidates edge 4 251 8093 innerTail51_38 best.bound = innerCandidates51_37 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_37 : scan edge (search edge 3) 4 [251] 251 innerTail51_37 best = best := by
  rw [innerTail51_37, scan, if_neg (by decide), innerCandidates51_37_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_38
    (if 3 ≤ innerCandidates51_37.length then search edge 3 [251, 8093] innerCandidates51_37 8344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_38

def innerTail51_36 : List Nat := 7919 :: innerTail51_37
def innerCandidates51_36 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_36_checked : candidates edge 4 251 7919 innerTail51_37 best.bound = innerCandidates51_36 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_36 : scan edge (search edge 3) 4 [251] 251 innerTail51_36 best = best := by
  rw [innerTail51_36, scan, if_neg (by decide), innerCandidates51_36_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_37
    (if 3 ≤ innerCandidates51_36.length then search edge 3 [251, 7919] innerCandidates51_36 8170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_37

def innerTail51_35 : List Nat := 7901 :: innerTail51_36
def innerCandidates51_35 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_35_checked : candidates edge 4 251 7901 innerTail51_36 best.bound = innerCandidates51_35 := by
  unfold candidates
  have row : edge 7901 = SearchData.row7901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_35 : scan edge (search edge 3) 4 [251] 251 innerTail51_35 best = best := by
  rw [innerTail51_35, scan, if_neg (by decide), innerCandidates51_35_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_36
    (if 3 ≤ innerCandidates51_35.length then search edge 3 [251, 7901] innerCandidates51_35 8152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_36

def innerTail51_34 : List Nat := 7499 :: innerTail51_35
def innerCandidates51_34 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_34_checked : candidates edge 4 251 7499 innerTail51_35 best.bound = innerCandidates51_34 := by
  unfold candidates
  have row : edge 7499 = SearchData.row7499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_34 : scan edge (search edge 3) 4 [251] 251 innerTail51_34 best = best := by
  rw [innerTail51_34, scan, if_neg (by decide), innerCandidates51_34_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_35
    (if 3 ≤ innerCandidates51_34.length then search edge 3 [251, 7499] innerCandidates51_34 7750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_35

def innerTail51_33 : List Nat := 7127 :: innerTail51_34
def innerCandidates51_33 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_33_checked : candidates edge 4 251 7127 innerTail51_34 best.bound = innerCandidates51_33 := by
  unfold candidates
  have row : edge 7127 = SearchData.row7127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_33 : scan edge (search edge 3) 4 [251] 251 innerTail51_33 best = best := by
  rw [innerTail51_33, scan, if_neg (by decide), innerCandidates51_33_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_34
    (if 3 ≤ innerCandidates51_33.length then search edge 3 [251, 7127] innerCandidates51_33 7378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_34

def innerTail51_32 : List Nat := 7079 :: innerTail51_33
def innerCandidates51_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_32_checked : candidates edge 4 251 7079 innerTail51_33 best.bound = innerCandidates51_32 := by
  unfold candidates
  have row : edge 7079 = SearchData.row7079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_32 : scan edge (search edge 3) 4 [251] 251 innerTail51_32 best = best := by
  rw [innerTail51_32, scan, if_neg (by decide), innerCandidates51_32_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_33
    (if 3 ≤ innerCandidates51_32.length then search edge 3 [251, 7079] innerCandidates51_32 7330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_33

def innerTail51_31 : List Nat := 6863 :: innerTail51_32
def innerCandidates51_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_31_checked : candidates edge 4 251 6863 innerTail51_32 best.bound = innerCandidates51_31 := by
  unfold candidates
  have row : edge 6863 = SearchData.row6863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_31 : scan edge (search edge 3) 4 [251] 251 innerTail51_31 best = best := by
  rw [innerTail51_31, scan, if_neg (by decide), innerCandidates51_31_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_32
    (if 3 ≤ innerCandidates51_31.length then search edge 3 [251, 6863] innerCandidates51_31 7114 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_32

def innerTail51_30 : List Nat := 6581 :: innerTail51_31
def innerCandidates51_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_30_checked : candidates edge 4 251 6581 innerTail51_31 best.bound = innerCandidates51_30 := by
  unfold candidates
  have row : edge 6581 = SearchData.row6581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_30 : scan edge (search edge 3) 4 [251] 251 innerTail51_30 best = best := by
  rw [innerTail51_30, scan, if_neg (by decide), innerCandidates51_30_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_31
    (if 3 ≤ innerCandidates51_30.length then search edge 3 [251, 6581] innerCandidates51_30 6832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_31

def innerTail51_29 : List Nat := 6473 :: innerTail51_30
def innerCandidates51_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_29_checked : candidates edge 4 251 6473 innerTail51_30 best.bound = innerCandidates51_29 := by
  unfold candidates
  have row : edge 6473 = SearchData.row6473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_29 : scan edge (search edge 3) 4 [251] 251 innerTail51_29 best = best := by
  rw [innerTail51_29, scan, if_neg (by decide), innerCandidates51_29_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_30
    (if 3 ≤ innerCandidates51_29.length then search edge 3 [251, 6473] innerCandidates51_29 6724 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_30

def innerTail51_28 : List Nat := 6197 :: innerTail51_29
def innerCandidates51_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_28_checked : candidates edge 4 251 6197 innerTail51_29 best.bound = innerCandidates51_28 := by
  unfold candidates
  have row : edge 6197 = SearchData.row6197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_28 : scan edge (search edge 3) 4 [251] 251 innerTail51_28 best = best := by
  rw [innerTail51_28, scan, if_neg (by decide), innerCandidates51_28_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_29
    (if 3 ≤ innerCandidates51_28.length then search edge 3 [251, 6197] innerCandidates51_28 6448 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_29

def innerTail51_27 : List Nat := 5897 :: innerTail51_28
def innerCandidates51_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_27_checked : candidates edge 4 251 5897 innerTail51_28 best.bound = innerCandidates51_27 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_27 : scan edge (search edge 3) 4 [251] 251 innerTail51_27 best = best := by
  rw [innerTail51_27, scan, if_neg (by decide), innerCandidates51_27_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_28
    (if 3 ≤ innerCandidates51_27.length then search edge 3 [251, 5897] innerCandidates51_27 6148 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_28

def innerTail51_26 : List Nat := 5651 :: innerTail51_27
def innerCandidates51_26 : List Nat := [7499, 8093]
set_option maxRecDepth 4096 in
theorem innerCandidates51_26_checked : candidates edge 4 251 5651 innerTail51_27 best.bound = innerCandidates51_26 := by
  unfold candidates
  have row : edge 5651 = SearchData.row5651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_26 : scan edge (search edge 3) 4 [251] 251 innerTail51_26 best = best := by
  rw [innerTail51_26, scan, if_neg (by decide), innerCandidates51_26_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_27
    (if 3 ≤ innerCandidates51_26.length then search edge 3 [251, 5651] innerCandidates51_26 5902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_27

def innerTail51_25 : List Nat := 5273 :: innerTail51_26
def innerCandidates51_25 : List Nat := [5651, 7127, 8093]
set_option maxRecDepth 4096 in
theorem innerCandidates51_25_checked : candidates edge 4 251 5273 innerTail51_26 best.bound = innerCandidates51_25 := by
  unfold candidates
  have row : edge 5273 = SearchData.row5273.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild51_25 : search edge 3 [251, 5273] innerCandidates51_25 5524 best = best := by decide

theorem innerScan51_25 : scan edge (search edge 3) 4 [251] 251 innerTail51_25 best = best := by
  rw [innerTail51_25, scan, if_neg (by decide), innerCandidates51_25_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_26
    (if 3 ≤ innerCandidates51_25.length then search edge 3 [251, 5273] innerCandidates51_25 5524 best else best) = best
  rw [if_pos (by decide), innerChild51_25]
  exact innerScan51_26

def innerTail51_24 : List Nat := 5237 :: innerTail51_25
def innerCandidates51_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_24_checked : candidates edge 4 251 5237 innerTail51_25 best.bound = innerCandidates51_24 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_24 : scan edge (search edge 3) 4 [251] 251 innerTail51_24 best = best := by
  rw [innerTail51_24, scan, if_neg (by decide), innerCandidates51_24_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_25
    (if 3 ≤ innerCandidates51_24.length then search edge 3 [251, 5237] innerCandidates51_24 5488 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_25

def innerTail51_23 : List Nat := 5189 :: innerTail51_24
def innerCandidates51_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_23_checked : candidates edge 4 251 5189 innerTail51_24 best.bound = innerCandidates51_23 := by
  unfold candidates
  have row : edge 5189 = SearchData.row5189.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_23 : scan edge (search edge 3) 4 [251] 251 innerTail51_23 best = best := by
  rw [innerTail51_23, scan, if_neg (by decide), innerCandidates51_23_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_24
    (if 3 ≤ innerCandidates51_23.length then search edge 3 [251, 5189] innerCandidates51_23 5440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_24

def innerTail51_22 : List Nat := 5147 :: innerTail51_23
def innerCandidates51_22 : List Nat := [5273]
set_option maxRecDepth 4096 in
theorem innerCandidates51_22_checked : candidates edge 4 251 5147 innerTail51_23 best.bound = innerCandidates51_22 := by
  unfold candidates
  have row : edge 5147 = SearchData.row5147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_22 : scan edge (search edge 3) 4 [251] 251 innerTail51_22 best = best := by
  rw [innerTail51_22, scan, if_neg (by decide), innerCandidates51_22_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_23
    (if 3 ≤ innerCandidates51_22.length then search edge 3 [251, 5147] innerCandidates51_22 5398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_23

def innerTail51_21 : List Nat := 4877 :: innerTail51_22
def innerCandidates51_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_21_checked : candidates edge 4 251 4877 innerTail51_22 best.bound = innerCandidates51_21 := by
  unfold candidates
  have row : edge 4877 = SearchData.row4877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_21 : scan edge (search edge 3) 4 [251] 251 innerTail51_21 best = best := by
  rw [innerTail51_21, scan, if_neg (by decide), innerCandidates51_21_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_22
    (if 3 ≤ innerCandidates51_21.length then search edge 3 [251, 4877] innerCandidates51_21 5128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_22

def innerTail51_20 : List Nat := 4019 :: innerTail51_21
def innerCandidates51_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_20_checked : candidates edge 4 251 4019 innerTail51_21 best.bound = innerCandidates51_20 := by
  unfold candidates
  have row : edge 4019 = SearchData.row4019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_20 : scan edge (search edge 3) 4 [251] 251 innerTail51_20 best = best := by
  rw [innerTail51_20, scan, if_neg (by decide), innerCandidates51_20_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_21
    (if 3 ≤ innerCandidates51_20.length then search edge 3 [251, 4019] innerCandidates51_20 4270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_21

def innerTail51_19 : List Nat := 3803 :: innerTail51_20
def innerCandidates51_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_19_checked : candidates edge 4 251 3803 innerTail51_20 best.bound = innerCandidates51_19 := by
  unfold candidates
  have row : edge 3803 = SearchData.row3803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_19 : scan edge (search edge 3) 4 [251] 251 innerTail51_19 best = best := by
  rw [innerTail51_19, scan, if_neg (by decide), innerCandidates51_19_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_20
    (if 3 ≤ innerCandidates51_19.length then search edge 3 [251, 3803] innerCandidates51_19 4054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_20

def innerTail51_18 : List Nat := 3557 :: innerTail51_19
def innerCandidates51_18 : List Nat := [3803]
set_option maxRecDepth 4096 in
theorem innerCandidates51_18_checked : candidates edge 4 251 3557 innerTail51_19 best.bound = innerCandidates51_18 := by
  unfold candidates
  have row : edge 3557 = SearchData.row3557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_18 : scan edge (search edge 3) 4 [251] 251 innerTail51_18 best = best := by
  rw [innerTail51_18, scan, if_neg (by decide), innerCandidates51_18_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_19
    (if 3 ≤ innerCandidates51_18.length then search edge 3 [251, 3557] innerCandidates51_18 3808 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_19

def innerTail51_17 : List Nat := 3347 :: innerTail51_18
def innerCandidates51_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_17_checked : candidates edge 4 251 3347 innerTail51_18 best.bound = innerCandidates51_17 := by
  unfold candidates
  have row : edge 3347 = SearchData.row3347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_17 : scan edge (search edge 3) 4 [251] 251 innerTail51_17 best = best := by
  rw [innerTail51_17, scan, if_neg (by decide), innerCandidates51_17_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_18
    (if 3 ≤ innerCandidates51_17.length then search edge 3 [251, 3347] innerCandidates51_17 3598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_18

def innerTail51_16 : List Nat := 3011 :: innerTail51_17
def innerCandidates51_16 : List Nat := [7127]
set_option maxRecDepth 4096 in
theorem innerCandidates51_16_checked : candidates edge 4 251 3011 innerTail51_17 best.bound = innerCandidates51_16 := by
  unfold candidates
  have row : edge 3011 = SearchData.row3011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_16 : scan edge (search edge 3) 4 [251] 251 innerTail51_16 best = best := by
  rw [innerTail51_16, scan, if_neg (by decide), innerCandidates51_16_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_17
    (if 3 ≤ innerCandidates51_16.length then search edge 3 [251, 3011] innerCandidates51_16 3262 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_17

def innerTail51_15 : List Nat := 2999 :: innerTail51_16
def innerCandidates51_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_15_checked : candidates edge 4 251 2999 innerTail51_16 best.bound = innerCandidates51_15 := by
  unfold candidates
  have row : edge 2999 = SearchData.row2999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_15 : scan edge (search edge 3) 4 [251] 251 innerTail51_15 best = best := by
  rw [innerTail51_15, scan, if_neg (by decide), innerCandidates51_15_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_16
    (if 3 ≤ innerCandidates51_15.length then search edge 3 [251, 2999] innerCandidates51_15 3250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_16

def innerTail51_14 : List Nat := 2801 :: innerTail51_15
def innerCandidates51_14 : List Nat := [3803, 7919, 8387]
set_option maxRecDepth 4096 in
theorem innerCandidates51_14_checked : candidates edge 4 251 2801 innerTail51_15 best.bound = innerCandidates51_14 := by
  unfold candidates
  have row : edge 2801 = SearchData.row2801.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild51_14 : search edge 3 [251, 2801] innerCandidates51_14 3052 best = best := by decide

theorem innerScan51_14 : scan edge (search edge 3) 4 [251] 251 innerTail51_14 best = best := by
  rw [innerTail51_14, scan, if_neg (by decide), innerCandidates51_14_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_15
    (if 3 ≤ innerCandidates51_14.length then search edge 3 [251, 2801] innerCandidates51_14 3052 best else best) = best
  rw [if_pos (by decide), innerChild51_14]
  exact innerScan51_15

def innerTail51_13 : List Nat := 2789 :: innerTail51_14
def innerCandidates51_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_13_checked : candidates edge 4 251 2789 innerTail51_14 best.bound = innerCandidates51_13 := by
  unfold candidates
  have row : edge 2789 = SearchData.row2789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_13 : scan edge (search edge 3) 4 [251] 251 innerTail51_13 best = best := by
  rw [innerTail51_13, scan, if_neg (by decide), innerCandidates51_13_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_14
    (if 3 ≤ innerCandidates51_13.length then search edge 3 [251, 2789] innerCandidates51_13 3040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_14

def innerTail51_12 : List Nat := 2417 :: innerTail51_13
def innerCandidates51_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_12_checked : candidates edge 4 251 2417 innerTail51_13 best.bound = innerCandidates51_12 := by
  unfold candidates
  have row : edge 2417 = PrimePairCache.smallRow2417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_12 : scan edge (search edge 3) 4 [251] 251 innerTail51_12 best = best := by
  rw [innerTail51_12, scan, if_neg (by decide), innerCandidates51_12_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_13
    (if 3 ≤ innerCandidates51_12.length then search edge 3 [251, 2417] innerCandidates51_12 2668 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_13

def innerTail51_11 : List Nat := 2207 :: innerTail51_12
def innerCandidates51_11 : List Nat := [3347]
set_option maxRecDepth 4096 in
theorem innerCandidates51_11_checked : candidates edge 4 251 2207 innerTail51_12 best.bound = innerCandidates51_11 := by
  unfold candidates
  have row : edge 2207 = PrimePairCache.smallRow2207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_11 : scan edge (search edge 3) 4 [251] 251 innerTail51_11 best = best := by
  rw [innerTail51_11, scan, if_neg (by decide), innerCandidates51_11_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_12
    (if 3 ≤ innerCandidates51_11.length then search edge 3 [251, 2207] innerCandidates51_11 2458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_12

def innerTail51_10 : List Nat := 2063 :: innerTail51_11
def innerCandidates51_10 : List Nat := [3803]
set_option maxRecDepth 4096 in
theorem innerCandidates51_10_checked : candidates edge 4 251 2063 innerTail51_11 best.bound = innerCandidates51_10 := by
  unfold candidates
  have row : edge 2063 = PrimePairCache.smallRow2063.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_10 : scan edge (search edge 3) 4 [251] 251 innerTail51_10 best = best := by
  rw [innerTail51_10, scan, if_neg (by decide), innerCandidates51_10_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_11
    (if 3 ≤ innerCandidates51_10.length then search edge 3 [251, 2063] innerCandidates51_10 2314 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_11

def innerTail51_9 : List Nat := 1979 :: innerTail51_10
def innerCandidates51_9 : List Nat := [2207, 2999]
set_option maxRecDepth 4096 in
theorem innerCandidates51_9_checked : candidates edge 4 251 1979 innerTail51_10 best.bound = innerCandidates51_9 := by
  unfold candidates
  have row : edge 1979 = PrimePairCache.smallRow1979.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_9 : scan edge (search edge 3) 4 [251] 251 innerTail51_9 best = best := by
  rw [innerTail51_9, scan, if_neg (by decide), innerCandidates51_9_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_10
    (if 3 ≤ innerCandidates51_9.length then search edge 3 [251, 1979] innerCandidates51_9 2230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_10

def innerTail51_8 : List Nat := 1709 :: innerTail51_9
def innerCandidates51_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_8_checked : candidates edge 4 251 1709 innerTail51_9 best.bound = innerCandidates51_8 := by
  unfold candidates
  have row : edge 1709 = PrimePairCache.smallRow1709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_8 : scan edge (search edge 3) 4 [251] 251 innerTail51_8 best = best := by
  rw [innerTail51_8, scan, if_neg (by decide), innerCandidates51_8_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_9
    (if 3 ≤ innerCandidates51_8.length then search edge 3 [251, 1709] innerCandidates51_8 1960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_9

def innerTail51_7 : List Nat := 1433 :: innerTail51_8
def innerCandidates51_7 : List Nat := [2999]
set_option maxRecDepth 4096 in
theorem innerCandidates51_7_checked : candidates edge 4 251 1433 innerTail51_8 best.bound = innerCandidates51_7 := by
  unfold candidates
  have row : edge 1433 = PrimePairCache.smallRow1433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_7 : scan edge (search edge 3) 4 [251] 251 innerTail51_7 best = best := by
  rw [innerTail51_7, scan, if_neg (by decide), innerCandidates51_7_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_8
    (if 3 ≤ innerCandidates51_7.length then search edge 3 [251, 1433] innerCandidates51_7 1684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_8

def innerTail51_6 : List Nat := 1151 :: innerTail51_7
def innerCandidates51_6 : List Nat := [1433, 2789, 5897, 6473, 6863]
set_option maxRecDepth 4096 in
theorem innerCandidates51_6_checked : candidates edge 4 251 1151 innerTail51_7 best.bound = innerCandidates51_6 := by
  unfold candidates
  have row : edge 1151 = PrimePairCache.smallRow1151.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild51_6 : search edge 3 [251, 1151] innerCandidates51_6 1402 best = best := by decide

theorem innerScan51_6 : scan edge (search edge 3) 4 [251] 251 innerTail51_6 best = best := by
  rw [innerTail51_6, scan, if_neg (by decide), innerCandidates51_6_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_7
    (if 3 ≤ innerCandidates51_6.length then search edge 3 [251, 1151] innerCandidates51_6 1402 best else best) = best
  rw [if_pos (by decide), innerChild51_6]
  exact innerScan51_7

def innerTail51_5 : List Nat := 971 :: innerTail51_6
def innerCandidates51_5 : List Nat := [7919]
set_option maxRecDepth 4096 in
theorem innerCandidates51_5_checked : candidates edge 4 251 971 innerTail51_6 best.bound = innerCandidates51_5 := by
  unfold candidates
  have row : edge 971 = PrimePairCache.smallRow971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_5 : scan edge (search edge 3) 4 [251] 251 innerTail51_5 best = best := by
  rw [innerTail51_5, scan, if_neg (by decide), innerCandidates51_5_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_6
    (if 3 ≤ innerCandidates51_5.length then search edge 3 [251, 971] innerCandidates51_5 1222 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_6

def innerTail51_4 : List Nat := 941 :: innerTail51_5
def innerCandidates51_4 : List Nat := [1709]
set_option maxRecDepth 4096 in
theorem innerCandidates51_4_checked : candidates edge 4 251 941 innerTail51_5 best.bound = innerCandidates51_4 := by
  unfold candidates
  have row : edge 941 = PrimePairCache.smallRow941.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_4 : scan edge (search edge 3) 4 [251] 251 innerTail51_4 best = best := by
  rw [innerTail51_4, scan, if_neg (by decide), innerCandidates51_4_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_5
    (if 3 ≤ innerCandidates51_4.length then search edge 3 [251, 941] innerCandidates51_4 1192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_5

def innerTail51_3 : List Nat := 761 :: innerTail51_4
def innerCandidates51_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates51_3_checked : candidates edge 4 251 761 innerTail51_4 best.bound = innerCandidates51_3 := by
  unfold candidates
  have row : edge 761 = PrimePairCache.smallRow761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_3 : scan edge (search edge 3) 4 [251] 251 innerTail51_3 best = best := by
  rw [innerTail51_3, scan, if_neg (by decide), innerCandidates51_3_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_4
    (if 3 ≤ innerCandidates51_3.length then search edge 3 [251, 761] innerCandidates51_3 1012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_4

def innerTail51_2 : List Nat := 491 :: innerTail51_3
def innerCandidates51_2 : List Nat := [7079]
set_option maxRecDepth 4096 in
theorem innerCandidates51_2_checked : candidates edge 4 251 491 innerTail51_3 best.bound = innerCandidates51_2 := by
  unfold candidates
  have row : edge 491 = PrimePairCache.smallRow491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_2 : scan edge (search edge 3) 4 [251] 251 innerTail51_2 best = best := by
  rw [innerTail51_2, scan, if_neg (by decide), innerCandidates51_2_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_3
    (if 3 ≤ innerCandidates51_2.length then search edge 3 [251, 491] innerCandidates51_2 742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_3

def innerTail51_1 : List Nat := 431 :: innerTail51_2
def innerCandidates51_1 : List Nat := [2417, 6197, 7127]
set_option maxRecDepth 4096 in
theorem innerCandidates51_1_checked : candidates edge 4 251 431 innerTail51_2 best.bound = innerCandidates51_1 := by
  unfold candidates
  have row : edge 431 = PrimePairCache.smallRow431.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild51_1 : search edge 3 [251, 431] innerCandidates51_1 682 best = best := by decide

theorem innerScan51_1 : scan edge (search edge 3) 4 [251] 251 innerTail51_1 best = best := by
  rw [innerTail51_1, scan, if_neg (by decide), innerCandidates51_1_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_2
    (if 3 ≤ innerCandidates51_1.length then search edge 3 [251, 431] innerCandidates51_1 682 best else best) = best
  rw [if_pos (by decide), innerChild51_1]
  exact innerScan51_2

def innerTail51_0 : List Nat := 347 :: innerTail51_1
def innerCandidates51_0 : List Nat := [7499]
set_option maxRecDepth 4096 in
theorem innerCandidates51_0_checked : candidates edge 4 251 347 innerTail51_1 best.bound = innerCandidates51_0 := by
  unfold candidates
  have row : edge 347 = PrimePairCache.smallRow347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan51_0 : scan edge (search edge 3) 4 [251] 251 innerTail51_0 best = best := by
  rw [innerTail51_0, scan, if_neg (by decide), innerCandidates51_0_checked]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_1
    (if 3 ≤ innerCandidates51_0.length then search edge 3 [251, 347] innerCandidates51_0 598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan51_1

theorem rootBranch51 : search edge 4 [251] rootCandidates51 251 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [251] 251 innerTail51_0 best = best
  exact innerScan51_0

end ProjectEulerSolutions.PrimePairUpper.Search
