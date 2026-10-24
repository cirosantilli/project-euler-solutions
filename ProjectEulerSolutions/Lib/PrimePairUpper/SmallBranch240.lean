import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates29 : List Nat := [449, 479, 617, 641, 743, 797, 839, 941, 1217, 1229, 1523, 1901, 2411, 2963, 3083, 3467, 3881, 3929, 4127, 4133, 4217, 4259, 4283, 4673, 5231, 5399, 5441, 5519, 5651, 6299, 6779, 6971, 7013, 7487, 7703, 7727]
set_option maxRecDepth 4096 in
theorem rootCandidates29_checked : candidates edge 5 0 131 tail30 best.bound = rootCandidates29 := by
  unfold candidates
  have row : edge 131 = PrimePairCache.smallRow131.contains := by funext q; rfl
  rw [row]
  decide

def innerTail29_36 : List Nat := []
theorem innerScan29_36 : scan edge (search edge 3) 4 [131] 131 innerTail29_36 best = best := rfl

def innerTail29_35 : List Nat := 7727 :: innerTail29_36
def innerCandidates29_35 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates29_35_checked : candidates edge 4 131 7727 innerTail29_36 best.bound = innerCandidates29_35 := by
  unfold candidates
  have row : edge 7727 = SearchData.row7727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_35 : scan edge (search edge 3) 4 [131] 131 innerTail29_35 best = best := by
  rw [innerTail29_35, scan, if_neg (by decide), innerCandidates29_35_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_36
    (if 3 ≤ innerCandidates29_35.length then search edge 3 [131, 7727] innerCandidates29_35 7858 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_36

def innerTail29_34 : List Nat := 7703 :: innerTail29_35
def innerCandidates29_34 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates29_34_checked : candidates edge 4 131 7703 innerTail29_35 best.bound = innerCandidates29_34 := by
  unfold candidates
  have row : edge 7703 = SearchData.row7703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_34 : scan edge (search edge 3) 4 [131] 131 innerTail29_34 best = best := by
  rw [innerTail29_34, scan, if_neg (by decide), innerCandidates29_34_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_35
    (if 3 ≤ innerCandidates29_34.length then search edge 3 [131, 7703] innerCandidates29_34 7834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_35

def innerTail29_33 : List Nat := 7487 :: innerTail29_34
def innerCandidates29_33 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates29_33_checked : candidates edge 4 131 7487 innerTail29_34 best.bound = innerCandidates29_33 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_33 : scan edge (search edge 3) 4 [131] 131 innerTail29_33 best = best := by
  rw [innerTail29_33, scan, if_neg (by decide), innerCandidates29_33_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_34
    (if 3 ≤ innerCandidates29_33.length then search edge 3 [131, 7487] innerCandidates29_33 7618 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_34

def innerTail29_32 : List Nat := 7013 :: innerTail29_33
def innerCandidates29_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates29_32_checked : candidates edge 4 131 7013 innerTail29_33 best.bound = innerCandidates29_32 := by
  unfold candidates
  have row : edge 7013 = SearchData.row7013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_32 : scan edge (search edge 3) 4 [131] 131 innerTail29_32 best = best := by
  rw [innerTail29_32, scan, if_neg (by decide), innerCandidates29_32_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_33
    (if 3 ≤ innerCandidates29_32.length then search edge 3 [131, 7013] innerCandidates29_32 7144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_33

def innerTail29_31 : List Nat := 6971 :: innerTail29_32
def innerCandidates29_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates29_31_checked : candidates edge 4 131 6971 innerTail29_32 best.bound = innerCandidates29_31 := by
  unfold candidates
  have row : edge 6971 = SearchData.row6971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_31 : scan edge (search edge 3) 4 [131] 131 innerTail29_31 best = best := by
  rw [innerTail29_31, scan, if_neg (by decide), innerCandidates29_31_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_32
    (if 3 ≤ innerCandidates29_31.length then search edge 3 [131, 6971] innerCandidates29_31 7102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_32

def innerTail29_30 : List Nat := 6779 :: innerTail29_31
def innerCandidates29_30 : List Nat := [7487]
set_option maxRecDepth 4096 in
theorem innerCandidates29_30_checked : candidates edge 4 131 6779 innerTail29_31 best.bound = innerCandidates29_30 := by
  unfold candidates
  have row : edge 6779 = SearchData.row6779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_30 : scan edge (search edge 3) 4 [131] 131 innerTail29_30 best = best := by
  rw [innerTail29_30, scan, if_neg (by decide), innerCandidates29_30_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_31
    (if 3 ≤ innerCandidates29_30.length then search edge 3 [131, 6779] innerCandidates29_30 6910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_31

def innerTail29_29 : List Nat := 6299 :: innerTail29_30
def innerCandidates29_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates29_29_checked : candidates edge 4 131 6299 innerTail29_30 best.bound = innerCandidates29_29 := by
  unfold candidates
  have row : edge 6299 = SearchData.row6299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_29 : scan edge (search edge 3) 4 [131] 131 innerTail29_29 best = best := by
  rw [innerTail29_29, scan, if_neg (by decide), innerCandidates29_29_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_30
    (if 3 ≤ innerCandidates29_29.length then search edge 3 [131, 6299] innerCandidates29_29 6430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_30

def innerTail29_28 : List Nat := 5651 :: innerTail29_29
def innerCandidates29_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates29_28_checked : candidates edge 4 131 5651 innerTail29_29 best.bound = innerCandidates29_28 := by
  unfold candidates
  have row : edge 5651 = SearchData.row5651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_28 : scan edge (search edge 3) 4 [131] 131 innerTail29_28 best = best := by
  rw [innerTail29_28, scan, if_neg (by decide), innerCandidates29_28_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_29
    (if 3 ≤ innerCandidates29_28.length then search edge 3 [131, 5651] innerCandidates29_28 5782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_29

def innerTail29_27 : List Nat := 5519 :: innerTail29_28
def innerCandidates29_27 : List Nat := [6299]
set_option maxRecDepth 4096 in
theorem innerCandidates29_27_checked : candidates edge 4 131 5519 innerTail29_28 best.bound = innerCandidates29_27 := by
  unfold candidates
  have row : edge 5519 = SearchData.row5519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_27 : scan edge (search edge 3) 4 [131] 131 innerTail29_27 best = best := by
  rw [innerTail29_27, scan, if_neg (by decide), innerCandidates29_27_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_28
    (if 3 ≤ innerCandidates29_27.length then search edge 3 [131, 5519] innerCandidates29_27 5650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_28

def innerTail29_26 : List Nat := 5441 :: innerTail29_27
def innerCandidates29_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates29_26_checked : candidates edge 4 131 5441 innerTail29_27 best.bound = innerCandidates29_26 := by
  unfold candidates
  have row : edge 5441 = SearchData.row5441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_26 : scan edge (search edge 3) 4 [131] 131 innerTail29_26 best = best := by
  rw [innerTail29_26, scan, if_neg (by decide), innerCandidates29_26_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_27
    (if 3 ≤ innerCandidates29_26.length then search edge 3 [131, 5441] innerCandidates29_26 5572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_27

def innerTail29_25 : List Nat := 5399 :: innerTail29_26
def innerCandidates29_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates29_25_checked : candidates edge 4 131 5399 innerTail29_26 best.bound = innerCandidates29_25 := by
  unfold candidates
  have row : edge 5399 = SearchData.row5399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_25 : scan edge (search edge 3) 4 [131] 131 innerTail29_25 best = best := by
  rw [innerTail29_25, scan, if_neg (by decide), innerCandidates29_25_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_26
    (if 3 ≤ innerCandidates29_25.length then search edge 3 [131, 5399] innerCandidates29_25 5530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_26

def innerTail29_24 : List Nat := 5231 :: innerTail29_25
def innerCandidates29_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates29_24_checked : candidates edge 4 131 5231 innerTail29_25 best.bound = innerCandidates29_24 := by
  unfold candidates
  have row : edge 5231 = SearchData.row5231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_24 : scan edge (search edge 3) 4 [131] 131 innerTail29_24 best = best := by
  rw [innerTail29_24, scan, if_neg (by decide), innerCandidates29_24_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_25
    (if 3 ≤ innerCandidates29_24.length then search edge 3 [131, 5231] innerCandidates29_24 5362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_25

def innerTail29_23 : List Nat := 4673 :: innerTail29_24
def innerCandidates29_23 : List Nat := [5519, 6779]
set_option maxRecDepth 4096 in
theorem innerCandidates29_23_checked : candidates edge 4 131 4673 innerTail29_24 best.bound = innerCandidates29_23 := by
  unfold candidates
  have row : edge 4673 = SearchData.row4673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_23 : scan edge (search edge 3) 4 [131] 131 innerTail29_23 best = best := by
  rw [innerTail29_23, scan, if_neg (by decide), innerCandidates29_23_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_24
    (if 3 ≤ innerCandidates29_23.length then search edge 3 [131, 4673] innerCandidates29_23 4804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_24

def innerTail29_22 : List Nat := 4283 :: innerTail29_23
def innerCandidates29_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates29_22_checked : candidates edge 4 131 4283 innerTail29_23 best.bound = innerCandidates29_22 := by
  unfold candidates
  have row : edge 4283 = SearchData.row4283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_22 : scan edge (search edge 3) 4 [131] 131 innerTail29_22 best = best := by
  rw [innerTail29_22, scan, if_neg (by decide), innerCandidates29_22_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_23
    (if 3 ≤ innerCandidates29_22.length then search edge 3 [131, 4283] innerCandidates29_22 4414 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_23

def innerTail29_21 : List Nat := 4259 :: innerTail29_22
def innerCandidates29_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates29_21_checked : candidates edge 4 131 4259 innerTail29_22 best.bound = innerCandidates29_21 := by
  unfold candidates
  have row : edge 4259 = SearchData.row4259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_21 : scan edge (search edge 3) 4 [131] 131 innerTail29_21 best = best := by
  rw [innerTail29_21, scan, if_neg (by decide), innerCandidates29_21_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_22
    (if 3 ≤ innerCandidates29_21.length then search edge 3 [131, 4259] innerCandidates29_21 4390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_22

def innerTail29_20 : List Nat := 4217 :: innerTail29_21
def innerCandidates29_20 : List Nat := [4673]
set_option maxRecDepth 4096 in
theorem innerCandidates29_20_checked : candidates edge 4 131 4217 innerTail29_21 best.bound = innerCandidates29_20 := by
  unfold candidates
  have row : edge 4217 = SearchData.row4217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_20 : scan edge (search edge 3) 4 [131] 131 innerTail29_20 best = best := by
  rw [innerTail29_20, scan, if_neg (by decide), innerCandidates29_20_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_21
    (if 3 ≤ innerCandidates29_20.length then search edge 3 [131, 4217] innerCandidates29_20 4348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_21

def innerTail29_19 : List Nat := 4133 :: innerTail29_20
def innerCandidates29_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates29_19_checked : candidates edge 4 131 4133 innerTail29_20 best.bound = innerCandidates29_19 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_19 : scan edge (search edge 3) 4 [131] 131 innerTail29_19 best = best := by
  rw [innerTail29_19, scan, if_neg (by decide), innerCandidates29_19_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_20
    (if 3 ≤ innerCandidates29_19.length then search edge 3 [131, 4133] innerCandidates29_19 4264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_20

def innerTail29_18 : List Nat := 4127 :: innerTail29_19
def innerCandidates29_18 : List Nat := [5519]
set_option maxRecDepth 4096 in
theorem innerCandidates29_18_checked : candidates edge 4 131 4127 innerTail29_19 best.bound = innerCandidates29_18 := by
  unfold candidates
  have row : edge 4127 = SearchData.row4127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_18 : scan edge (search edge 3) 4 [131] 131 innerTail29_18 best = best := by
  rw [innerTail29_18, scan, if_neg (by decide), innerCandidates29_18_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_19
    (if 3 ≤ innerCandidates29_18.length then search edge 3 [131, 4127] innerCandidates29_18 4258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_19

def innerTail29_17 : List Nat := 3929 :: innerTail29_18
def innerCandidates29_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates29_17_checked : candidates edge 4 131 3929 innerTail29_18 best.bound = innerCandidates29_17 := by
  unfold candidates
  have row : edge 3929 = SearchData.row3929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_17 : scan edge (search edge 3) 4 [131] 131 innerTail29_17 best = best := by
  rw [innerTail29_17, scan, if_neg (by decide), innerCandidates29_17_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_18
    (if 3 ≤ innerCandidates29_17.length then search edge 3 [131, 3929] innerCandidates29_17 4060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_18

def innerTail29_16 : List Nat := 3881 :: innerTail29_17
def innerCandidates29_16 : List Nat := [5519]
set_option maxRecDepth 4096 in
theorem innerCandidates29_16_checked : candidates edge 4 131 3881 innerTail29_17 best.bound = innerCandidates29_16 := by
  unfold candidates
  have row : edge 3881 = SearchData.row3881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_16 : scan edge (search edge 3) 4 [131] 131 innerTail29_16 best = best := by
  rw [innerTail29_16, scan, if_neg (by decide), innerCandidates29_16_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_17
    (if 3 ≤ innerCandidates29_16.length then search edge 3 [131, 3881] innerCandidates29_16 4012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_17

def innerTail29_15 : List Nat := 3467 :: innerTail29_16
def innerCandidates29_15 : List Nat := [5231]
set_option maxRecDepth 4096 in
theorem innerCandidates29_15_checked : candidates edge 4 131 3467 innerTail29_16 best.bound = innerCandidates29_15 := by
  unfold candidates
  have row : edge 3467 = SearchData.row3467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_15 : scan edge (search edge 3) 4 [131] 131 innerTail29_15 best = best := by
  rw [innerTail29_15, scan, if_neg (by decide), innerCandidates29_15_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_16
    (if 3 ≤ innerCandidates29_15.length then search edge 3 [131, 3467] innerCandidates29_15 3598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_16

def innerTail29_14 : List Nat := 3083 :: innerTail29_15
def innerCandidates29_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates29_14_checked : candidates edge 4 131 3083 innerTail29_15 best.bound = innerCandidates29_14 := by
  unfold candidates
  have row : edge 3083 = SearchData.row3083.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_14 : scan edge (search edge 3) 4 [131] 131 innerTail29_14 best = best := by
  rw [innerTail29_14, scan, if_neg (by decide), innerCandidates29_14_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_15
    (if 3 ≤ innerCandidates29_14.length then search edge 3 [131, 3083] innerCandidates29_14 3214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_15

def innerTail29_13 : List Nat := 2963 :: innerTail29_14
def innerCandidates29_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates29_13_checked : candidates edge 4 131 2963 innerTail29_14 best.bound = innerCandidates29_13 := by
  unfold candidates
  have row : edge 2963 = SearchData.row2963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_13 : scan edge (search edge 3) 4 [131] 131 innerTail29_13 best = best := by
  rw [innerTail29_13, scan, if_neg (by decide), innerCandidates29_13_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_14
    (if 3 ≤ innerCandidates29_13.length then search edge 3 [131, 2963] innerCandidates29_13 3094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_14

def innerTail29_12 : List Nat := 2411 :: innerTail29_13
def innerCandidates29_12 : List Nat := [5519]
set_option maxRecDepth 4096 in
theorem innerCandidates29_12_checked : candidates edge 4 131 2411 innerTail29_13 best.bound = innerCandidates29_12 := by
  unfold candidates
  have row : edge 2411 = PrimePairCache.smallRow2411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_12 : scan edge (search edge 3) 4 [131] 131 innerTail29_12 best = best := by
  rw [innerTail29_12, scan, if_neg (by decide), innerCandidates29_12_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_13
    (if 3 ≤ innerCandidates29_12.length then search edge 3 [131, 2411] innerCandidates29_12 2542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_13

def innerTail29_11 : List Nat := 1901 :: innerTail29_12
def innerCandidates29_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates29_11_checked : candidates edge 4 131 1901 innerTail29_12 best.bound = innerCandidates29_11 := by
  unfold candidates
  have row : edge 1901 = PrimePairCache.smallRow1901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_11 : scan edge (search edge 3) 4 [131] 131 innerTail29_11 best = best := by
  rw [innerTail29_11, scan, if_neg (by decide), innerCandidates29_11_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_12
    (if 3 ≤ innerCandidates29_11.length then search edge 3 [131, 1901] innerCandidates29_11 2032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_12

def innerTail29_10 : List Nat := 1523 :: innerTail29_11
def innerCandidates29_10 : List Nat := [3929]
set_option maxRecDepth 4096 in
theorem innerCandidates29_10_checked : candidates edge 4 131 1523 innerTail29_11 best.bound = innerCandidates29_10 := by
  unfold candidates
  have row : edge 1523 = PrimePairCache.smallRow1523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_10 : scan edge (search edge 3) 4 [131] 131 innerTail29_10 best = best := by
  rw [innerTail29_10, scan, if_neg (by decide), innerCandidates29_10_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_11
    (if 3 ≤ innerCandidates29_10.length then search edge 3 [131, 1523] innerCandidates29_10 1654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_11

def innerTail29_9 : List Nat := 1229 :: innerTail29_10
def innerCandidates29_9 : List Nat := [2963, 3881, 4127, 5399]
set_option maxRecDepth 4096 in
theorem innerCandidates29_9_checked : candidates edge 4 131 1229 innerTail29_10 best.bound = innerCandidates29_9 := by
  unfold candidates
  have row : edge 1229 = PrimePairCache.smallRow1229.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild29_9 : search edge 3 [131, 1229] innerCandidates29_9 1360 best = best := by decide

theorem innerScan29_9 : scan edge (search edge 3) 4 [131] 131 innerTail29_9 best = best := by
  rw [innerTail29_9, scan, if_neg (by decide), innerCandidates29_9_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_10
    (if 3 ≤ innerCandidates29_9.length then search edge 3 [131, 1229] innerCandidates29_9 1360 best else best) = best
  rw [if_pos (by decide), innerChild29_9]
  exact innerScan29_10

def innerTail29_8 : List Nat := 1217 :: innerTail29_9
def innerCandidates29_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates29_8_checked : candidates edge 4 131 1217 innerTail29_9 best.bound = innerCandidates29_8 := by
  unfold candidates
  have row : edge 1217 = PrimePairCache.smallRow1217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_8 : scan edge (search edge 3) 4 [131] 131 innerTail29_8 best = best := by
  rw [innerTail29_8, scan, if_neg (by decide), innerCandidates29_8_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_9
    (if 3 ≤ innerCandidates29_8.length then search edge 3 [131, 1217] innerCandidates29_8 1348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_9

def innerTail29_7 : List Nat := 941 :: innerTail29_8
def innerCandidates29_7 : List Nat := [1229, 6299, 6971]
set_option maxRecDepth 4096 in
theorem innerCandidates29_7_checked : candidates edge 4 131 941 innerTail29_8 best.bound = innerCandidates29_7 := by
  unfold candidates
  have row : edge 941 = PrimePairCache.smallRow941.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild29_7 : search edge 3 [131, 941] innerCandidates29_7 1072 best = best := by decide

theorem innerScan29_7 : scan edge (search edge 3) 4 [131] 131 innerTail29_7 best = best := by
  rw [innerTail29_7, scan, if_neg (by decide), innerCandidates29_7_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_8
    (if 3 ≤ innerCandidates29_7.length then search edge 3 [131, 941] innerCandidates29_7 1072 best else best) = best
  rw [if_pos (by decide), innerChild29_7]
  exact innerScan29_8

def innerTail29_6 : List Nat := 839 :: innerTail29_7
def innerCandidates29_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates29_6_checked : candidates edge 4 131 839 innerTail29_7 best.bound = innerCandidates29_6 := by
  unfold candidates
  have row : edge 839 = PrimePairCache.smallRow839.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_6 : scan edge (search edge 3) 4 [131] 131 innerTail29_6 best = best := by
  rw [innerTail29_6, scan, if_neg (by decide), innerCandidates29_6_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_7
    (if 3 ≤ innerCandidates29_6.length then search edge 3 [131, 839] innerCandidates29_6 970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_7

def innerTail29_5 : List Nat := 797 :: innerTail29_6
def innerCandidates29_5 : List Nat := [2411, 3929, 4259, 5519]
set_option maxRecDepth 4096 in
theorem innerCandidates29_5_checked : candidates edge 4 131 797 innerTail29_6 best.bound = innerCandidates29_5 := by
  unfold candidates
  have row : edge 797 = PrimePairCache.smallRow797.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild29_5 : search edge 3 [131, 797] innerCandidates29_5 928 best = best := by decide

theorem innerScan29_5 : scan edge (search edge 3) 4 [131] 131 innerTail29_5 best = best := by
  rw [innerTail29_5, scan, if_neg (by decide), innerCandidates29_5_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_6
    (if 3 ≤ innerCandidates29_5.length then search edge 3 [131, 797] innerCandidates29_5 928 best else best) = best
  rw [if_pos (by decide), innerChild29_5]
  exact innerScan29_6

def innerTail29_4 : List Nat := 743 :: innerTail29_5
def innerCandidates29_4 : List Nat := [4283]
set_option maxRecDepth 4096 in
theorem innerCandidates29_4_checked : candidates edge 4 131 743 innerTail29_5 best.bound = innerCandidates29_4 := by
  unfold candidates
  have row : edge 743 = PrimePairCache.smallRow743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_4 : scan edge (search edge 3) 4 [131] 131 innerTail29_4 best = best := by
  rw [innerTail29_4, scan, if_neg (by decide), innerCandidates29_4_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_5
    (if 3 ≤ innerCandidates29_4.length then search edge 3 [131, 743] innerCandidates29_4 874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_5

def innerTail29_3 : List Nat := 641 :: innerTail29_4
def innerCandidates29_3 : List Nat := [5441]
set_option maxRecDepth 4096 in
theorem innerCandidates29_3_checked : candidates edge 4 131 641 innerTail29_4 best.bound = innerCandidates29_3 := by
  unfold candidates
  have row : edge 641 = PrimePairCache.smallRow641.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_3 : scan edge (search edge 3) 4 [131] 131 innerTail29_3 best = best := by
  rw [innerTail29_3, scan, if_neg (by decide), innerCandidates29_3_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_4
    (if 3 ≤ innerCandidates29_3.length then search edge 3 [131, 641] innerCandidates29_3 772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_4

def innerTail29_2 : List Nat := 617 :: innerTail29_3
def innerCandidates29_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates29_2_checked : candidates edge 4 131 617 innerTail29_3 best.bound = innerCandidates29_2 := by
  unfold candidates
  have row : edge 617 = PrimePairCache.smallRow617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_2 : scan edge (search edge 3) 4 [131] 131 innerTail29_2 best = best := by
  rw [innerTail29_2, scan, if_neg (by decide), innerCandidates29_2_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_3
    (if 3 ≤ innerCandidates29_2.length then search edge 3 [131, 617] innerCandidates29_2 748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_3

def innerTail29_1 : List Nat := 479 :: innerTail29_2
def innerCandidates29_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates29_1_checked : candidates edge 4 131 479 innerTail29_2 best.bound = innerCandidates29_1 := by
  unfold candidates
  have row : edge 479 = PrimePairCache.smallRow479.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_1 : scan edge (search edge 3) 4 [131] 131 innerTail29_1 best = best := by
  rw [innerTail29_1, scan, if_neg (by decide), innerCandidates29_1_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_2
    (if 3 ≤ innerCandidates29_1.length then search edge 3 [131, 479] innerCandidates29_1 610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_2

def innerTail29_0 : List Nat := 449 :: innerTail29_1
def innerCandidates29_0 : List Nat := [941]
set_option maxRecDepth 4096 in
theorem innerCandidates29_0_checked : candidates edge 4 131 449 innerTail29_1 best.bound = innerCandidates29_0 := by
  unfold candidates
  have row : edge 449 = PrimePairCache.smallRow449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan29_0 : scan edge (search edge 3) 4 [131] 131 innerTail29_0 best = best := by
  rw [innerTail29_0, scan, if_neg (by decide), innerCandidates29_0_checked]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_1
    (if 3 ≤ innerCandidates29_0.length then search edge 3 [131, 449] innerCandidates29_0 580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan29_1

theorem rootBranch29 : search edge 4 [131] rootCandidates29 131 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [131] 131 innerTail29_0 best = best
  exact innerScan29_0

end ProjectEulerSolutions.PrimePairUpper.Search
