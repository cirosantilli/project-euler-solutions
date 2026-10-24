import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates30 : List Nat := [191, 197, 239, 353, 359, 491, 587, 659, 743, 947, 1031, 2621, 2963, 3137, 3167, 3191, 3677, 3989, 4481, 4721, 5237, 5981, 6173, 6197, 6359, 6491, 6971, 7043, 7121, 7127, 7457, 7499, 7517, 7577, 7643, 7757, 8147, 8219, 8231]
set_option maxRecDepth 4096 in
theorem rootCandidates30_checked : candidates edge 5 0 137 tail31 best.bound = rootCandidates30 := by
  unfold candidates
  have row : edge 137 = PrimePairCache.smallRow137.contains := by funext q; rfl
  rw [row]
  decide

def innerTail30_39 : List Nat := []
theorem innerScan30_39 : scan edge (search edge 3) 4 [137] 137 innerTail30_39 best = best := rfl

def innerTail30_38 : List Nat := 8231 :: innerTail30_39
def innerCandidates30_38 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates30_38_checked : candidates edge 4 137 8231 innerTail30_39 best.bound = innerCandidates30_38 := by
  unfold candidates
  have row : edge 8231 = SearchData.row8231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_38 : scan edge (search edge 3) 4 [137] 137 innerTail30_38 best = best := by
  rw [innerTail30_38, scan, if_neg (by decide), innerCandidates30_38_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_39
    (if 3 ≤ innerCandidates30_38.length then search edge 3 [137, 8231] innerCandidates30_38 8368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_39

def innerTail30_37 : List Nat := 8219 :: innerTail30_38
def innerCandidates30_37 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates30_37_checked : candidates edge 4 137 8219 innerTail30_38 best.bound = innerCandidates30_37 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_37 : scan edge (search edge 3) 4 [137] 137 innerTail30_37 best = best := by
  rw [innerTail30_37, scan, if_neg (by decide), innerCandidates30_37_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_38
    (if 3 ≤ innerCandidates30_37.length then search edge 3 [137, 8219] innerCandidates30_37 8356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_38

def innerTail30_36 : List Nat := 8147 :: innerTail30_37
def innerCandidates30_36 : List Nat := [8219]
set_option maxRecDepth 4096 in
theorem innerCandidates30_36_checked : candidates edge 4 137 8147 innerTail30_37 best.bound = innerCandidates30_36 := by
  unfold candidates
  have row : edge 8147 = SearchData.row8147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_36 : scan edge (search edge 3) 4 [137] 137 innerTail30_36 best = best := by
  rw [innerTail30_36, scan, if_neg (by decide), innerCandidates30_36_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_37
    (if 3 ≤ innerCandidates30_36.length then search edge 3 [137, 8147] innerCandidates30_36 8284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_37

def innerTail30_35 : List Nat := 7757 :: innerTail30_36
def innerCandidates30_35 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates30_35_checked : candidates edge 4 137 7757 innerTail30_36 best.bound = innerCandidates30_35 := by
  unfold candidates
  have row : edge 7757 = SearchData.row7757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_35 : scan edge (search edge 3) 4 [137] 137 innerTail30_35 best = best := by
  rw [innerTail30_35, scan, if_neg (by decide), innerCandidates30_35_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_36
    (if 3 ≤ innerCandidates30_35.length then search edge 3 [137, 7757] innerCandidates30_35 7894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_36

def innerTail30_34 : List Nat := 7643 :: innerTail30_35
def innerCandidates30_34 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates30_34_checked : candidates edge 4 137 7643 innerTail30_35 best.bound = innerCandidates30_34 := by
  unfold candidates
  have row : edge 7643 = SearchData.row7643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_34 : scan edge (search edge 3) 4 [137] 137 innerTail30_34 best = best := by
  rw [innerTail30_34, scan, if_neg (by decide), innerCandidates30_34_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_35
    (if 3 ≤ innerCandidates30_34.length then search edge 3 [137, 7643] innerCandidates30_34 7780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_35

def innerTail30_33 : List Nat := 7577 :: innerTail30_34
def innerCandidates30_33 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates30_33_checked : candidates edge 4 137 7577 innerTail30_34 best.bound = innerCandidates30_33 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_33 : scan edge (search edge 3) 4 [137] 137 innerTail30_33 best = best := by
  rw [innerTail30_33, scan, if_neg (by decide), innerCandidates30_33_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_34
    (if 3 ≤ innerCandidates30_33.length then search edge 3 [137, 7577] innerCandidates30_33 7714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_34

def innerTail30_32 : List Nat := 7517 :: innerTail30_33
def innerCandidates30_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates30_32_checked : candidates edge 4 137 7517 innerTail30_33 best.bound = innerCandidates30_32 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_32 : scan edge (search edge 3) 4 [137] 137 innerTail30_32 best = best := by
  rw [innerTail30_32, scan, if_neg (by decide), innerCandidates30_32_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_33
    (if 3 ≤ innerCandidates30_32.length then search edge 3 [137, 7517] innerCandidates30_32 7654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_33

def innerTail30_31 : List Nat := 7499 :: innerTail30_32
def innerCandidates30_31 : List Nat := [7757]
set_option maxRecDepth 4096 in
theorem innerCandidates30_31_checked : candidates edge 4 137 7499 innerTail30_32 best.bound = innerCandidates30_31 := by
  unfold candidates
  have row : edge 7499 = SearchData.row7499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_31 : scan edge (search edge 3) 4 [137] 137 innerTail30_31 best = best := by
  rw [innerTail30_31, scan, if_neg (by decide), innerCandidates30_31_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_32
    (if 3 ≤ innerCandidates30_31.length then search edge 3 [137, 7499] innerCandidates30_31 7636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_32

def innerTail30_30 : List Nat := 7457 :: innerTail30_31
def innerCandidates30_30 : List Nat := [7757]
set_option maxRecDepth 4096 in
theorem innerCandidates30_30_checked : candidates edge 4 137 7457 innerTail30_31 best.bound = innerCandidates30_30 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_30 : scan edge (search edge 3) 4 [137] 137 innerTail30_30 best = best := by
  rw [innerTail30_30, scan, if_neg (by decide), innerCandidates30_30_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_31
    (if 3 ≤ innerCandidates30_30.length then search edge 3 [137, 7457] innerCandidates30_30 7594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_31

def innerTail30_29 : List Nat := 7127 :: innerTail30_30
def innerCandidates30_29 : List Nat := [7757]
set_option maxRecDepth 4096 in
theorem innerCandidates30_29_checked : candidates edge 4 137 7127 innerTail30_30 best.bound = innerCandidates30_29 := by
  unfold candidates
  have row : edge 7127 = SearchData.row7127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_29 : scan edge (search edge 3) 4 [137] 137 innerTail30_29 best = best := by
  rw [innerTail30_29, scan, if_neg (by decide), innerCandidates30_29_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_30
    (if 3 ≤ innerCandidates30_29.length then search edge 3 [137, 7127] innerCandidates30_29 7264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_30

def innerTail30_28 : List Nat := 7121 :: innerTail30_29
def innerCandidates30_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates30_28_checked : candidates edge 4 137 7121 innerTail30_29 best.bound = innerCandidates30_28 := by
  unfold candidates
  have row : edge 7121 = SearchData.row7121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_28 : scan edge (search edge 3) 4 [137] 137 innerTail30_28 best = best := by
  rw [innerTail30_28, scan, if_neg (by decide), innerCandidates30_28_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_29
    (if 3 ≤ innerCandidates30_28.length then search edge 3 [137, 7121] innerCandidates30_28 7258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_29

def innerTail30_27 : List Nat := 7043 :: innerTail30_28
def innerCandidates30_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates30_27_checked : candidates edge 4 137 7043 innerTail30_28 best.bound = innerCandidates30_27 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_27 : scan edge (search edge 3) 4 [137] 137 innerTail30_27 best = best := by
  rw [innerTail30_27, scan, if_neg (by decide), innerCandidates30_27_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_28
    (if 3 ≤ innerCandidates30_27.length then search edge 3 [137, 7043] innerCandidates30_27 7180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_28

def innerTail30_26 : List Nat := 6971 :: innerTail30_27
def innerCandidates30_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates30_26_checked : candidates edge 4 137 6971 innerTail30_27 best.bound = innerCandidates30_26 := by
  unfold candidates
  have row : edge 6971 = SearchData.row6971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_26 : scan edge (search edge 3) 4 [137] 137 innerTail30_26 best = best := by
  rw [innerTail30_26, scan, if_neg (by decide), innerCandidates30_26_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_27
    (if 3 ≤ innerCandidates30_26.length then search edge 3 [137, 6971] innerCandidates30_26 7108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_27

def innerTail30_25 : List Nat := 6491 :: innerTail30_26
def innerCandidates30_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates30_25_checked : candidates edge 4 137 6491 innerTail30_26 best.bound = innerCandidates30_25 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_25 : scan edge (search edge 3) 4 [137] 137 innerTail30_25 best = best := by
  rw [innerTail30_25, scan, if_neg (by decide), innerCandidates30_25_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_26
    (if 3 ≤ innerCandidates30_25.length then search edge 3 [137, 6491] innerCandidates30_25 6628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_26

def innerTail30_24 : List Nat := 6359 :: innerTail30_25
def innerCandidates30_24 : List Nat := [7577]
set_option maxRecDepth 4096 in
theorem innerCandidates30_24_checked : candidates edge 4 137 6359 innerTail30_25 best.bound = innerCandidates30_24 := by
  unfold candidates
  have row : edge 6359 = SearchData.row6359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_24 : scan edge (search edge 3) 4 [137] 137 innerTail30_24 best = best := by
  rw [innerTail30_24, scan, if_neg (by decide), innerCandidates30_24_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_25
    (if 3 ≤ innerCandidates30_24.length then search edge 3 [137, 6359] innerCandidates30_24 6496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_25

def innerTail30_23 : List Nat := 6197 :: innerTail30_24
def innerCandidates30_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates30_23_checked : candidates edge 4 137 6197 innerTail30_24 best.bound = innerCandidates30_23 := by
  unfold candidates
  have row : edge 6197 = SearchData.row6197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_23 : scan edge (search edge 3) 4 [137] 137 innerTail30_23 best = best := by
  rw [innerTail30_23, scan, if_neg (by decide), innerCandidates30_23_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_24
    (if 3 ≤ innerCandidates30_23.length then search edge 3 [137, 6197] innerCandidates30_23 6334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_24

def innerTail30_22 : List Nat := 6173 :: innerTail30_23
def innerCandidates30_22 : List Nat := [8231]
set_option maxRecDepth 4096 in
theorem innerCandidates30_22_checked : candidates edge 4 137 6173 innerTail30_23 best.bound = innerCandidates30_22 := by
  unfold candidates
  have row : edge 6173 = SearchData.row6173.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_22 : scan edge (search edge 3) 4 [137] 137 innerTail30_22 best = best := by
  rw [innerTail30_22, scan, if_neg (by decide), innerCandidates30_22_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_23
    (if 3 ≤ innerCandidates30_22.length then search edge 3 [137, 6173] innerCandidates30_22 6310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_23

def innerTail30_21 : List Nat := 5981 :: innerTail30_22
def innerCandidates30_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates30_21_checked : candidates edge 4 137 5981 innerTail30_22 best.bound = innerCandidates30_21 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_21 : scan edge (search edge 3) 4 [137] 137 innerTail30_21 best = best := by
  rw [innerTail30_21, scan, if_neg (by decide), innerCandidates30_21_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_22
    (if 3 ≤ innerCandidates30_21.length then search edge 3 [137, 5981] innerCandidates30_21 6118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_22

def innerTail30_20 : List Nat := 5237 :: innerTail30_21
def innerCandidates30_20 : List Nat := [7043]
set_option maxRecDepth 4096 in
theorem innerCandidates30_20_checked : candidates edge 4 137 5237 innerTail30_21 best.bound = innerCandidates30_20 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_20 : scan edge (search edge 3) 4 [137] 137 innerTail30_20 best = best := by
  rw [innerTail30_20, scan, if_neg (by decide), innerCandidates30_20_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_21
    (if 3 ≤ innerCandidates30_20.length then search edge 3 [137, 5237] innerCandidates30_20 5374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_21

def innerTail30_19 : List Nat := 4721 :: innerTail30_20
def innerCandidates30_19 : List Nat := [6971]
set_option maxRecDepth 4096 in
theorem innerCandidates30_19_checked : candidates edge 4 137 4721 innerTail30_20 best.bound = innerCandidates30_19 := by
  unfold candidates
  have row : edge 4721 = SearchData.row4721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_19 : scan edge (search edge 3) 4 [137] 137 innerTail30_19 best = best := by
  rw [innerTail30_19, scan, if_neg (by decide), innerCandidates30_19_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_20
    (if 3 ≤ innerCandidates30_19.length then search edge 3 [137, 4721] innerCandidates30_19 4858 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_20

def innerTail30_18 : List Nat := 4481 :: innerTail30_19
def innerCandidates30_18 : List Nat := [6491]
set_option maxRecDepth 4096 in
theorem innerCandidates30_18_checked : candidates edge 4 137 4481 innerTail30_19 best.bound = innerCandidates30_18 := by
  unfold candidates
  have row : edge 4481 = SearchData.row4481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_18 : scan edge (search edge 3) 4 [137] 137 innerTail30_18 best = best := by
  rw [innerTail30_18, scan, if_neg (by decide), innerCandidates30_18_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_19
    (if 3 ≤ innerCandidates30_18.length then search edge 3 [137, 4481] innerCandidates30_18 4618 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_19

def innerTail30_17 : List Nat := 3989 :: innerTail30_18
def innerCandidates30_17 : List Nat := [7043]
set_option maxRecDepth 4096 in
theorem innerCandidates30_17_checked : candidates edge 4 137 3989 innerTail30_18 best.bound = innerCandidates30_17 := by
  unfold candidates
  have row : edge 3989 = SearchData.row3989.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_17 : scan edge (search edge 3) 4 [137] 137 innerTail30_17 best = best := by
  rw [innerTail30_17, scan, if_neg (by decide), innerCandidates30_17_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_18
    (if 3 ≤ innerCandidates30_17.length then search edge 3 [137, 3989] innerCandidates30_17 4126 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_18

def innerTail30_16 : List Nat := 3677 :: innerTail30_17
def innerCandidates30_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates30_16_checked : candidates edge 4 137 3677 innerTail30_17 best.bound = innerCandidates30_16 := by
  unfold candidates
  have row : edge 3677 = SearchData.row3677.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_16 : scan edge (search edge 3) 4 [137] 137 innerTail30_16 best = best := by
  rw [innerTail30_16, scan, if_neg (by decide), innerCandidates30_16_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_17
    (if 3 ≤ innerCandidates30_16.length then search edge 3 [137, 3677] innerCandidates30_16 3814 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_17

def innerTail30_15 : List Nat := 3191 :: innerTail30_16
def innerCandidates30_15 : List Nat := [7043]
set_option maxRecDepth 4096 in
theorem innerCandidates30_15_checked : candidates edge 4 137 3191 innerTail30_16 best.bound = innerCandidates30_15 := by
  unfold candidates
  have row : edge 3191 = SearchData.row3191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_15 : scan edge (search edge 3) 4 [137] 137 innerTail30_15 best = best := by
  rw [innerTail30_15, scan, if_neg (by decide), innerCandidates30_15_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_16
    (if 3 ≤ innerCandidates30_15.length then search edge 3 [137, 3191] innerCandidates30_15 3328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_16

def innerTail30_14 : List Nat := 3167 :: innerTail30_15
def innerCandidates30_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates30_14_checked : candidates edge 4 137 3167 innerTail30_15 best.bound = innerCandidates30_14 := by
  unfold candidates
  have row : edge 3167 = SearchData.row3167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_14 : scan edge (search edge 3) 4 [137] 137 innerTail30_14 best = best := by
  rw [innerTail30_14, scan, if_neg (by decide), innerCandidates30_14_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_15
    (if 3 ≤ innerCandidates30_14.length then search edge 3 [137, 3167] innerCandidates30_14 3304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_15

def innerTail30_13 : List Nat := 3137 :: innerTail30_14
def innerCandidates30_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates30_13_checked : candidates edge 4 137 3137 innerTail30_14 best.bound = innerCandidates30_13 := by
  unfold candidates
  have row : edge 3137 = SearchData.row3137.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_13 : scan edge (search edge 3) 4 [137] 137 innerTail30_13 best = best := by
  rw [innerTail30_13, scan, if_neg (by decide), innerCandidates30_13_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_14
    (if 3 ≤ innerCandidates30_13.length then search edge 3 [137, 3137] innerCandidates30_13 3274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_14

def innerTail30_12 : List Nat := 2963 :: innerTail30_13
def innerCandidates30_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates30_12_checked : candidates edge 4 137 2963 innerTail30_13 best.bound = innerCandidates30_12 := by
  unfold candidates
  have row : edge 2963 = SearchData.row2963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_12 : scan edge (search edge 3) 4 [137] 137 innerTail30_12 best = best := by
  rw [innerTail30_12, scan, if_neg (by decide), innerCandidates30_12_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_13
    (if 3 ≤ innerCandidates30_12.length then search edge 3 [137, 2963] innerCandidates30_12 3100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_13

def innerTail30_11 : List Nat := 2621 :: innerTail30_12
def innerCandidates30_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates30_11_checked : candidates edge 4 137 2621 innerTail30_12 best.bound = innerCandidates30_11 := by
  unfold candidates
  have row : edge 2621 = PrimePairCache.smallRow2621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_11 : scan edge (search edge 3) 4 [137] 137 innerTail30_11 best = best := by
  rw [innerTail30_11, scan, if_neg (by decide), innerCandidates30_11_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_12
    (if 3 ≤ innerCandidates30_11.length then search edge 3 [137, 2621] innerCandidates30_11 2758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_12

def innerTail30_10 : List Nat := 1031 :: innerTail30_11
def innerCandidates30_10 : List Nat := [5237]
set_option maxRecDepth 4096 in
theorem innerCandidates30_10_checked : candidates edge 4 137 1031 innerTail30_11 best.bound = innerCandidates30_10 := by
  unfold candidates
  have row : edge 1031 = PrimePairCache.smallRow1031.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_10 : scan edge (search edge 3) 4 [137] 137 innerTail30_10 best = best := by
  rw [innerTail30_10, scan, if_neg (by decide), innerCandidates30_10_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_11
    (if 3 ≤ innerCandidates30_10.length then search edge 3 [137, 1031] innerCandidates30_10 1168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_11

def innerTail30_9 : List Nat := 947 :: innerTail30_10
def innerCandidates30_9 : List Nat := [5981]
set_option maxRecDepth 4096 in
theorem innerCandidates30_9_checked : candidates edge 4 137 947 innerTail30_10 best.bound = innerCandidates30_9 := by
  unfold candidates
  have row : edge 947 = PrimePairCache.smallRow947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_9 : scan edge (search edge 3) 4 [137] 137 innerTail30_9 best = best := by
  rw [innerTail30_9, scan, if_neg (by decide), innerCandidates30_9_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_10
    (if 3 ≤ innerCandidates30_9.length then search edge 3 [137, 947] innerCandidates30_9 1084 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_10

def innerTail30_8 : List Nat := 743 :: innerTail30_9
def innerCandidates30_8 : List Nat := [947]
set_option maxRecDepth 4096 in
theorem innerCandidates30_8_checked : candidates edge 4 137 743 innerTail30_9 best.bound = innerCandidates30_8 := by
  unfold candidates
  have row : edge 743 = PrimePairCache.smallRow743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_8 : scan edge (search edge 3) 4 [137] 137 innerTail30_8 best = best := by
  rw [innerTail30_8, scan, if_neg (by decide), innerCandidates30_8_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_9
    (if 3 ≤ innerCandidates30_8.length then search edge 3 [137, 743] innerCandidates30_8 880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_9

def innerTail30_7 : List Nat := 659 :: innerTail30_8
def innerCandidates30_7 : List Nat := [947, 7517]
set_option maxRecDepth 4096 in
theorem innerCandidates30_7_checked : candidates edge 4 137 659 innerTail30_8 best.bound = innerCandidates30_7 := by
  unfold candidates
  have row : edge 659 = PrimePairCache.smallRow659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_7 : scan edge (search edge 3) 4 [137] 137 innerTail30_7 best = best := by
  rw [innerTail30_7, scan, if_neg (by decide), innerCandidates30_7_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_8
    (if 3 ≤ innerCandidates30_7.length then search edge 3 [137, 659] innerCandidates30_7 796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_8

def innerTail30_6 : List Nat := 587 :: innerTail30_7
def innerCandidates30_6 : List Nat := [7127]
set_option maxRecDepth 4096 in
theorem innerCandidates30_6_checked : candidates edge 4 137 587 innerTail30_7 best.bound = innerCandidates30_6 := by
  unfold candidates
  have row : edge 587 = PrimePairCache.smallRow587.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_6 : scan edge (search edge 3) 4 [137] 137 innerTail30_6 best = best := by
  rw [innerTail30_6, scan, if_neg (by decide), innerCandidates30_6_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_7
    (if 3 ≤ innerCandidates30_6.length then search edge 3 [137, 587] innerCandidates30_6 724 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_7

def innerTail30_5 : List Nat := 491 :: innerTail30_6
def innerCandidates30_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates30_5_checked : candidates edge 4 137 491 innerTail30_6 best.bound = innerCandidates30_5 := by
  unfold candidates
  have row : edge 491 = PrimePairCache.smallRow491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_5 : scan edge (search edge 3) 4 [137] 137 innerTail30_5 best = best := by
  rw [innerTail30_5, scan, if_neg (by decide), innerCandidates30_5_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_6
    (if 3 ≤ innerCandidates30_5.length then search edge 3 [137, 491] innerCandidates30_5 628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_6

def innerTail30_4 : List Nat := 359 :: innerTail30_5
def innerCandidates30_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates30_4_checked : candidates edge 4 137 359 innerTail30_5 best.bound = innerCandidates30_4 := by
  unfold candidates
  have row : edge 359 = PrimePairCache.smallRow359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_4 : scan edge (search edge 3) 4 [137] 137 innerTail30_4 best = best := by
  rw [innerTail30_4, scan, if_neg (by decide), innerCandidates30_4_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_5
    (if 3 ≤ innerCandidates30_4.length then search edge 3 [137, 359] innerCandidates30_4 496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_5

def innerTail30_3 : List Nat := 353 :: innerTail30_4
def innerCandidates30_3 : List Nat := [359, 7517, 8219]
set_option maxRecDepth 4096 in
theorem innerCandidates30_3_checked : candidates edge 4 137 353 innerTail30_4 best.bound = innerCandidates30_3 := by
  unfold candidates
  have row : edge 353 = PrimePairCache.smallRow353.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild30_3 : search edge 3 [137, 353] innerCandidates30_3 490 best = best := by decide

theorem innerScan30_3 : scan edge (search edge 3) 4 [137] 137 innerTail30_3 best = best := by
  rw [innerTail30_3, scan, if_neg (by decide), innerCandidates30_3_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_4
    (if 3 ≤ innerCandidates30_3.length then search edge 3 [137, 353] innerCandidates30_3 490 best else best) = best
  rw [if_pos (by decide), innerChild30_3]
  exact innerScan30_4

def innerTail30_2 : List Nat := 239 :: innerTail30_3
def innerCandidates30_2 : List Nat := [947, 6197]
set_option maxRecDepth 4096 in
theorem innerCandidates30_2_checked : candidates edge 4 137 239 innerTail30_3 best.bound = innerCandidates30_2 := by
  unfold candidates
  have row : edge 239 = PrimePairCache.smallRow239.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_2 : scan edge (search edge 3) 4 [137] 137 innerTail30_2 best = best := by
  rw [innerTail30_2, scan, if_neg (by decide), innerCandidates30_2_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_3
    (if 3 ≤ innerCandidates30_2.length then search edge 3 [137, 239] innerCandidates30_2 376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_3

def innerTail30_1 : List Nat := 197 :: innerTail30_2
def innerCandidates30_1 : List Nat := [947, 6173]
set_option maxRecDepth 4096 in
theorem innerCandidates30_1_checked : candidates edge 4 137 197 innerTail30_2 best.bound = innerCandidates30_1 := by
  unfold candidates
  have row : edge 197 = PrimePairCache.smallRow197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_1 : scan edge (search edge 3) 4 [137] 137 innerTail30_1 best = best := by
  rw [innerTail30_1, scan, if_neg (by decide), innerCandidates30_1_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_2
    (if 3 ≤ innerCandidates30_1.length then search edge 3 [137, 197] innerCandidates30_1 334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_2

def innerTail30_0 : List Nat := 191 :: innerTail30_1
def innerCandidates30_0 : List Nat := [5981]
set_option maxRecDepth 4096 in
theorem innerCandidates30_0_checked : candidates edge 4 137 191 innerTail30_1 best.bound = innerCandidates30_0 := by
  unfold candidates
  have row : edge 191 = PrimePairCache.smallRow191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan30_0 : scan edge (search edge 3) 4 [137] 137 innerTail30_0 best = best := by
  rw [innerTail30_0, scan, if_neg (by decide), innerCandidates30_0_checked]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_1
    (if 3 ≤ innerCandidates30_0.length then search edge 3 [137, 191] innerCandidates30_0 328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan30_1

theorem rootBranch30 : search edge 4 [137] rootCandidates30 137 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [137] 137 innerTail30_0 best = best
  exact innerScan30_0

end ProjectEulerSolutions.PrimePairUpper.Search
