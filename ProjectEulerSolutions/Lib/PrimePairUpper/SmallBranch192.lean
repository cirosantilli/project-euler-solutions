import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates120 : List Nat := [827, 1217, 1283, 1409, 1451, 1493, 1871, 2309, 2687, 2837, 3203, 3863, 4073, 4409, 4451, 5273, 5387, 5393, 5399, 5519, 5807, 5861, 5987, 6101, 6173, 6257, 6491, 6971, 7349, 7451, 7517, 7949, 8039, 8147]
set_option maxRecDepth 4096 in
theorem rootCandidates120_checked : candidates edge 5 0 677 tail121 best.bound = rootCandidates120 := by
  unfold candidates
  have row : edge 677 = PrimePairCache.smallRow677.contains := by funext q; rfl
  rw [row]
  decide

def innerTail120_34 : List Nat := []
theorem innerScan120_34 : scan edge (search edge 3) 4 [677] 677 innerTail120_34 best = best := rfl

def innerTail120_33 : List Nat := 8147 :: innerTail120_34
def innerCandidates120_33 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates120_33_checked : candidates edge 4 677 8147 innerTail120_34 best.bound = innerCandidates120_33 := by
  unfold candidates
  have row : edge 8147 = SearchData.row8147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_33 : scan edge (search edge 3) 4 [677] 677 innerTail120_33 best = best := by
  rw [innerTail120_33, scan, if_neg (by decide), innerCandidates120_33_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_34
    (if 3 ≤ innerCandidates120_33.length then search edge 3 [677, 8147] innerCandidates120_33 8824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_34

def innerTail120_32 : List Nat := 8039 :: innerTail120_33
def innerCandidates120_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates120_32_checked : candidates edge 4 677 8039 innerTail120_33 best.bound = innerCandidates120_32 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_32 : scan edge (search edge 3) 4 [677] 677 innerTail120_32 best = best := by
  rw [innerTail120_32, scan, if_neg (by decide), innerCandidates120_32_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_33
    (if 3 ≤ innerCandidates120_32.length then search edge 3 [677, 8039] innerCandidates120_32 8716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_33

def innerTail120_31 : List Nat := 7949 :: innerTail120_32
def innerCandidates120_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates120_31_checked : candidates edge 4 677 7949 innerTail120_32 best.bound = innerCandidates120_31 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_31 : scan edge (search edge 3) 4 [677] 677 innerTail120_31 best = best := by
  rw [innerTail120_31, scan, if_neg (by decide), innerCandidates120_31_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_32
    (if 3 ≤ innerCandidates120_31.length then search edge 3 [677, 7949] innerCandidates120_31 8626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_32

def innerTail120_30 : List Nat := 7517 :: innerTail120_31
def innerCandidates120_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates120_30_checked : candidates edge 4 677 7517 innerTail120_31 best.bound = innerCandidates120_30 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_30 : scan edge (search edge 3) 4 [677] 677 innerTail120_30 best = best := by
  rw [innerTail120_30, scan, if_neg (by decide), innerCandidates120_30_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_31
    (if 3 ≤ innerCandidates120_30.length then search edge 3 [677, 7517] innerCandidates120_30 8194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_31

def innerTail120_29 : List Nat := 7451 :: innerTail120_30
def innerCandidates120_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates120_29_checked : candidates edge 4 677 7451 innerTail120_30 best.bound = innerCandidates120_29 := by
  unfold candidates
  have row : edge 7451 = SearchData.row7451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_29 : scan edge (search edge 3) 4 [677] 677 innerTail120_29 best = best := by
  rw [innerTail120_29, scan, if_neg (by decide), innerCandidates120_29_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_30
    (if 3 ≤ innerCandidates120_29.length then search edge 3 [677, 7451] innerCandidates120_29 8128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_30

def innerTail120_28 : List Nat := 7349 :: innerTail120_29
def innerCandidates120_28 : List Nat := [8147]
set_option maxRecDepth 4096 in
theorem innerCandidates120_28_checked : candidates edge 4 677 7349 innerTail120_29 best.bound = innerCandidates120_28 := by
  unfold candidates
  have row : edge 7349 = SearchData.row7349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_28 : scan edge (search edge 3) 4 [677] 677 innerTail120_28 best = best := by
  rw [innerTail120_28, scan, if_neg (by decide), innerCandidates120_28_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_29
    (if 3 ≤ innerCandidates120_28.length then search edge 3 [677, 7349] innerCandidates120_28 8026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_29

def innerTail120_27 : List Nat := 6971 :: innerTail120_28
def innerCandidates120_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates120_27_checked : candidates edge 4 677 6971 innerTail120_28 best.bound = innerCandidates120_27 := by
  unfold candidates
  have row : edge 6971 = SearchData.row6971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_27 : scan edge (search edge 3) 4 [677] 677 innerTail120_27 best = best := by
  rw [innerTail120_27, scan, if_neg (by decide), innerCandidates120_27_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_28
    (if 3 ≤ innerCandidates120_27.length then search edge 3 [677, 6971] innerCandidates120_27 7648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_28

def innerTail120_26 : List Nat := 6491 :: innerTail120_27
def innerCandidates120_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates120_26_checked : candidates edge 4 677 6491 innerTail120_27 best.bound = innerCandidates120_26 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_26 : scan edge (search edge 3) 4 [677] 677 innerTail120_26 best = best := by
  rw [innerTail120_26, scan, if_neg (by decide), innerCandidates120_26_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_27
    (if 3 ≤ innerCandidates120_26.length then search edge 3 [677, 6491] innerCandidates120_26 7168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_27

def innerTail120_25 : List Nat := 6257 :: innerTail120_26
def innerCandidates120_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates120_25_checked : candidates edge 4 677 6257 innerTail120_26 best.bound = innerCandidates120_25 := by
  unfold candidates
  have row : edge 6257 = SearchData.row6257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_25 : scan edge (search edge 3) 4 [677] 677 innerTail120_25 best = best := by
  rw [innerTail120_25, scan, if_neg (by decide), innerCandidates120_25_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_26
    (if 3 ≤ innerCandidates120_25.length then search edge 3 [677, 6257] innerCandidates120_25 6934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_26

def innerTail120_24 : List Nat := 6173 :: innerTail120_25
def innerCandidates120_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates120_24_checked : candidates edge 4 677 6173 innerTail120_25 best.bound = innerCandidates120_24 := by
  unfold candidates
  have row : edge 6173 = SearchData.row6173.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_24 : scan edge (search edge 3) 4 [677] 677 innerTail120_24 best = best := by
  rw [innerTail120_24, scan, if_neg (by decide), innerCandidates120_24_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_25
    (if 3 ≤ innerCandidates120_24.length then search edge 3 [677, 6173] innerCandidates120_24 6850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_25

def innerTail120_23 : List Nat := 6101 :: innerTail120_24
def innerCandidates120_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates120_23_checked : candidates edge 4 677 6101 innerTail120_24 best.bound = innerCandidates120_23 := by
  unfold candidates
  have row : edge 6101 = SearchData.row6101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_23 : scan edge (search edge 3) 4 [677] 677 innerTail120_23 best = best := by
  rw [innerTail120_23, scan, if_neg (by decide), innerCandidates120_23_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_24
    (if 3 ≤ innerCandidates120_23.length then search edge 3 [677, 6101] innerCandidates120_23 6778 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_24

def innerTail120_22 : List Nat := 5987 :: innerTail120_23
def innerCandidates120_22 : List Nat := [6491]
set_option maxRecDepth 4096 in
theorem innerCandidates120_22_checked : candidates edge 4 677 5987 innerTail120_23 best.bound = innerCandidates120_22 := by
  unfold candidates
  have row : edge 5987 = SearchData.row5987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_22 : scan edge (search edge 3) 4 [677] 677 innerTail120_22 best = best := by
  rw [innerTail120_22, scan, if_neg (by decide), innerCandidates120_22_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_23
    (if 3 ≤ innerCandidates120_22.length then search edge 3 [677, 5987] innerCandidates120_22 6664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_23

def innerTail120_21 : List Nat := 5861 :: innerTail120_22
def innerCandidates120_21 : List Nat := [5987]
set_option maxRecDepth 4096 in
theorem innerCandidates120_21_checked : candidates edge 4 677 5861 innerTail120_22 best.bound = innerCandidates120_21 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_21 : scan edge (search edge 3) 4 [677] 677 innerTail120_21 best = best := by
  rw [innerTail120_21, scan, if_neg (by decide), innerCandidates120_21_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_22
    (if 3 ≤ innerCandidates120_21.length then search edge 3 [677, 5861] innerCandidates120_21 6538 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_22

def innerTail120_20 : List Nat := 5807 :: innerTail120_21
def innerCandidates120_20 : List Nat := [7949]
set_option maxRecDepth 4096 in
theorem innerCandidates120_20_checked : candidates edge 4 677 5807 innerTail120_21 best.bound = innerCandidates120_20 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_20 : scan edge (search edge 3) 4 [677] 677 innerTail120_20 best = best := by
  rw [innerTail120_20, scan, if_neg (by decide), innerCandidates120_20_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_21
    (if 3 ≤ innerCandidates120_20.length then search edge 3 [677, 5807] innerCandidates120_20 6484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_21

def innerTail120_19 : List Nat := 5519 :: innerTail120_20
def innerCandidates120_19 : List Nat := [7517]
set_option maxRecDepth 4096 in
theorem innerCandidates120_19_checked : candidates edge 4 677 5519 innerTail120_20 best.bound = innerCandidates120_19 := by
  unfold candidates
  have row : edge 5519 = SearchData.row5519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_19 : scan edge (search edge 3) 4 [677] 677 innerTail120_19 best = best := by
  rw [innerTail120_19, scan, if_neg (by decide), innerCandidates120_19_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_20
    (if 3 ≤ innerCandidates120_19.length then search edge 3 [677, 5519] innerCandidates120_19 6196 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_20

def innerTail120_18 : List Nat := 5399 :: innerTail120_19
def innerCandidates120_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates120_18_checked : candidates edge 4 677 5399 innerTail120_19 best.bound = innerCandidates120_18 := by
  unfold candidates
  have row : edge 5399 = SearchData.row5399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_18 : scan edge (search edge 3) 4 [677] 677 innerTail120_18 best = best := by
  rw [innerTail120_18, scan, if_neg (by decide), innerCandidates120_18_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_19
    (if 3 ≤ innerCandidates120_18.length then search edge 3 [677, 5399] innerCandidates120_18 6076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_19

def innerTail120_17 : List Nat := 5393 :: innerTail120_18
def innerCandidates120_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates120_17_checked : candidates edge 4 677 5393 innerTail120_18 best.bound = innerCandidates120_17 := by
  unfold candidates
  have row : edge 5393 = SearchData.row5393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_17 : scan edge (search edge 3) 4 [677] 677 innerTail120_17 best = best := by
  rw [innerTail120_17, scan, if_neg (by decide), innerCandidates120_17_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_18
    (if 3 ≤ innerCandidates120_17.length then search edge 3 [677, 5393] innerCandidates120_17 6070 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_18

def innerTail120_16 : List Nat := 5387 :: innerTail120_17
def innerCandidates120_16 : List Nat := [5399, 5807]
set_option maxRecDepth 4096 in
theorem innerCandidates120_16_checked : candidates edge 4 677 5387 innerTail120_17 best.bound = innerCandidates120_16 := by
  unfold candidates
  have row : edge 5387 = SearchData.row5387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_16 : scan edge (search edge 3) 4 [677] 677 innerTail120_16 best = best := by
  rw [innerTail120_16, scan, if_neg (by decide), innerCandidates120_16_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_17
    (if 3 ≤ innerCandidates120_16.length then search edge 3 [677, 5387] innerCandidates120_16 6064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_17

def innerTail120_15 : List Nat := 5273 :: innerTail120_16
def innerCandidates120_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates120_15_checked : candidates edge 4 677 5273 innerTail120_16 best.bound = innerCandidates120_15 := by
  unfold candidates
  have row : edge 5273 = SearchData.row5273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_15 : scan edge (search edge 3) 4 [677] 677 innerTail120_15 best = best := by
  rw [innerTail120_15, scan, if_neg (by decide), innerCandidates120_15_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_16
    (if 3 ≤ innerCandidates120_15.length then search edge 3 [677, 5273] innerCandidates120_15 5950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_16

def innerTail120_14 : List Nat := 4451 :: innerTail120_15
def innerCandidates120_14 : List Nat := [6257]
set_option maxRecDepth 4096 in
theorem innerCandidates120_14_checked : candidates edge 4 677 4451 innerTail120_15 best.bound = innerCandidates120_14 := by
  unfold candidates
  have row : edge 4451 = SearchData.row4451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_14 : scan edge (search edge 3) 4 [677] 677 innerTail120_14 best = best := by
  rw [innerTail120_14, scan, if_neg (by decide), innerCandidates120_14_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_15
    (if 3 ≤ innerCandidates120_14.length then search edge 3 [677, 4451] innerCandidates120_14 5128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_15

def innerTail120_13 : List Nat := 4409 :: innerTail120_14
def innerCandidates120_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates120_13_checked : candidates edge 4 677 4409 innerTail120_14 best.bound = innerCandidates120_13 := by
  unfold candidates
  have row : edge 4409 = SearchData.row4409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_13 : scan edge (search edge 3) 4 [677] 677 innerTail120_13 best = best := by
  rw [innerTail120_13, scan, if_neg (by decide), innerCandidates120_13_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_14
    (if 3 ≤ innerCandidates120_13.length then search edge 3 [677, 4409] innerCandidates120_13 5086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_14

def innerTail120_12 : List Nat := 4073 :: innerTail120_13
def innerCandidates120_12 : List Nat := [5393]
set_option maxRecDepth 4096 in
theorem innerCandidates120_12_checked : candidates edge 4 677 4073 innerTail120_13 best.bound = innerCandidates120_12 := by
  unfold candidates
  have row : edge 4073 = SearchData.row4073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_12 : scan edge (search edge 3) 4 [677] 677 innerTail120_12 best = best := by
  rw [innerTail120_12, scan, if_neg (by decide), innerCandidates120_12_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_13
    (if 3 ≤ innerCandidates120_12.length then search edge 3 [677, 4073] innerCandidates120_12 4750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_13

def innerTail120_11 : List Nat := 3863 :: innerTail120_12
def innerCandidates120_11 : List Nat := [4451]
set_option maxRecDepth 4096 in
theorem innerCandidates120_11_checked : candidates edge 4 677 3863 innerTail120_12 best.bound = innerCandidates120_11 := by
  unfold candidates
  have row : edge 3863 = SearchData.row3863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_11 : scan edge (search edge 3) 4 [677] 677 innerTail120_11 best = best := by
  rw [innerTail120_11, scan, if_neg (by decide), innerCandidates120_11_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_12
    (if 3 ≤ innerCandidates120_11.length then search edge 3 [677, 3863] innerCandidates120_11 4540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_12

def innerTail120_10 : List Nat := 3203 :: innerTail120_11
def innerCandidates120_10 : List Nat := [6101, 7949]
set_option maxRecDepth 4096 in
theorem innerCandidates120_10_checked : candidates edge 4 677 3203 innerTail120_11 best.bound = innerCandidates120_10 := by
  unfold candidates
  have row : edge 3203 = SearchData.row3203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_10 : scan edge (search edge 3) 4 [677] 677 innerTail120_10 best = best := by
  rw [innerTail120_10, scan, if_neg (by decide), innerCandidates120_10_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_11
    (if 3 ≤ innerCandidates120_10.length then search edge 3 [677, 3203] innerCandidates120_10 3880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_11

def innerTail120_9 : List Nat := 2837 :: innerTail120_10
def innerCandidates120_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates120_9_checked : candidates edge 4 677 2837 innerTail120_10 best.bound = innerCandidates120_9 := by
  unfold candidates
  have row : edge 2837 = SearchData.row2837.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_9 : scan edge (search edge 3) 4 [677] 677 innerTail120_9 best = best := by
  rw [innerTail120_9, scan, if_neg (by decide), innerCandidates120_9_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_10
    (if 3 ≤ innerCandidates120_9.length then search edge 3 [677, 2837] innerCandidates120_9 3514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_10

def innerTail120_8 : List Nat := 2687 :: innerTail120_9
def innerCandidates120_8 : List Nat := [8039]
set_option maxRecDepth 4096 in
theorem innerCandidates120_8_checked : candidates edge 4 677 2687 innerTail120_9 best.bound = innerCandidates120_8 := by
  unfold candidates
  have row : edge 2687 = SearchData.row2687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_8 : scan edge (search edge 3) 4 [677] 677 innerTail120_8 best = best := by
  rw [innerTail120_8, scan, if_neg (by decide), innerCandidates120_8_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_9
    (if 3 ≤ innerCandidates120_8.length then search edge 3 [677, 2687] innerCandidates120_8 3364 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_9

def innerTail120_7 : List Nat := 2309 :: innerTail120_8
def innerCandidates120_7 : List Nat := [2687, 4451, 6257]
set_option maxRecDepth 4096 in
theorem innerCandidates120_7_checked : candidates edge 4 677 2309 innerTail120_8 best.bound = innerCandidates120_7 := by
  unfold candidates
  have row : edge 2309 = PrimePairCache.smallRow2309.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild120_7 : search edge 3 [677, 2309] innerCandidates120_7 2986 best = best := by decide

theorem innerScan120_7 : scan edge (search edge 3) 4 [677] 677 innerTail120_7 best = best := by
  rw [innerTail120_7, scan, if_neg (by decide), innerCandidates120_7_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_8
    (if 3 ≤ innerCandidates120_7.length then search edge 3 [677, 2309] innerCandidates120_7 2986 best else best) = best
  rw [if_pos (by decide), innerChild120_7]
  exact innerScan120_8

def innerTail120_6 : List Nat := 1871 :: innerTail120_7
def innerCandidates120_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates120_6_checked : candidates edge 4 677 1871 innerTail120_7 best.bound = innerCandidates120_6 := by
  unfold candidates
  have row : edge 1871 = PrimePairCache.smallRow1871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_6 : scan edge (search edge 3) 4 [677] 677 innerTail120_6 best = best := by
  rw [innerTail120_6, scan, if_neg (by decide), innerCandidates120_6_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_7
    (if 3 ≤ innerCandidates120_6.length then search edge 3 [677, 1871] innerCandidates120_6 2548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_7

def innerTail120_5 : List Nat := 1493 :: innerTail120_6
def innerCandidates120_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates120_5_checked : candidates edge 4 677 1493 innerTail120_6 best.bound = innerCandidates120_5 := by
  unfold candidates
  have row : edge 1493 = PrimePairCache.smallRow1493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_5 : scan edge (search edge 3) 4 [677] 677 innerTail120_5 best = best := by
  rw [innerTail120_5, scan, if_neg (by decide), innerCandidates120_5_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_6
    (if 3 ≤ innerCandidates120_5.length then search edge 3 [677, 1493] innerCandidates120_5 2170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_6

def innerTail120_4 : List Nat := 1451 :: innerTail120_5
def innerCandidates120_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates120_4_checked : candidates edge 4 677 1451 innerTail120_5 best.bound = innerCandidates120_4 := by
  unfold candidates
  have row : edge 1451 = PrimePairCache.smallRow1451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_4 : scan edge (search edge 3) 4 [677] 677 innerTail120_4 best = best := by
  rw [innerTail120_4, scan, if_neg (by decide), innerCandidates120_4_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_5
    (if 3 ≤ innerCandidates120_4.length then search edge 3 [677, 1451] innerCandidates120_4 2128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_5

def innerTail120_3 : List Nat := 1409 :: innerTail120_4
def innerCandidates120_3 : List Nat := [5387]
set_option maxRecDepth 4096 in
theorem innerCandidates120_3_checked : candidates edge 4 677 1409 innerTail120_4 best.bound = innerCandidates120_3 := by
  unfold candidates
  have row : edge 1409 = PrimePairCache.smallRow1409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_3 : scan edge (search edge 3) 4 [677] 677 innerTail120_3 best = best := by
  rw [innerTail120_3, scan, if_neg (by decide), innerCandidates120_3_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_4
    (if 3 ≤ innerCandidates120_3.length then search edge 3 [677, 1409] innerCandidates120_3 2086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_4

def innerTail120_2 : List Nat := 1283 :: innerTail120_3
def innerCandidates120_2 : List Nat := [5807, 7949]
set_option maxRecDepth 4096 in
theorem innerCandidates120_2_checked : candidates edge 4 677 1283 innerTail120_3 best.bound = innerCandidates120_2 := by
  unfold candidates
  have row : edge 1283 = PrimePairCache.smallRow1283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_2 : scan edge (search edge 3) 4 [677] 677 innerTail120_2 best = best := by
  rw [innerTail120_2, scan, if_neg (by decide), innerCandidates120_2_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_3
    (if 3 ≤ innerCandidates120_2.length then search edge 3 [677, 1283] innerCandidates120_2 1960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_3

def innerTail120_1 : List Nat := 1217 :: innerTail120_2
def innerCandidates120_1 : List Nat := [2309, 5393, 6173]
set_option maxRecDepth 4096 in
theorem innerCandidates120_1_checked : candidates edge 4 677 1217 innerTail120_2 best.bound = innerCandidates120_1 := by
  unfold candidates
  have row : edge 1217 = PrimePairCache.smallRow1217.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild120_1 : search edge 3 [677, 1217] innerCandidates120_1 1894 best = best := by decide

theorem innerScan120_1 : scan edge (search edge 3) 4 [677] 677 innerTail120_1 best = best := by
  rw [innerTail120_1, scan, if_neg (by decide), innerCandidates120_1_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_2
    (if 3 ≤ innerCandidates120_1.length then search edge 3 [677, 1217] innerCandidates120_1 1894 best else best) = best
  rw [if_pos (by decide), innerChild120_1]
  exact innerScan120_2

def innerTail120_0 : List Nat := 827 :: innerTail120_1
def innerCandidates120_0 : List Nat := [1871]
set_option maxRecDepth 4096 in
theorem innerCandidates120_0_checked : candidates edge 4 677 827 innerTail120_1 best.bound = innerCandidates120_0 := by
  unfold candidates
  have row : edge 827 = PrimePairCache.smallRow827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan120_0 : scan edge (search edge 3) 4 [677] 677 innerTail120_0 best = best := by
  rw [innerTail120_0, scan, if_neg (by decide), innerCandidates120_0_checked]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_1
    (if 3 ≤ innerCandidates120_0.length then search edge 3 [677, 827] innerCandidates120_0 1504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan120_1

theorem rootBranch120 : search edge 4 [677] rootCandidates120 677 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [677] 677 innerTail120_0 best = best
  exact innerScan120_0

def rootCandidates119 : List Nat := [769, 1117, 1279, 1327, 1489, 1579, 1657, 1759, 2281, 2293, 2659, 3271, 4297, 4597, 4813, 4993, 6271, 6841, 7177, 7417, 8089]
set_option maxRecDepth 4096 in
theorem rootCandidates119_checked : candidates edge 5 0 673 tail120 best.bound = rootCandidates119 := by
  unfold candidates
  have row : edge 673 = PrimePairCache.smallRow673.contains := by funext q; rfl
  rw [row]
  decide

def innerTail119_21 : List Nat := []
theorem innerScan119_21 : scan edge (search edge 3) 4 [673] 673 innerTail119_21 best = best := rfl

def innerTail119_20 : List Nat := 8089 :: innerTail119_21
def innerCandidates119_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates119_20_checked : candidates edge 4 673 8089 innerTail119_21 best.bound = innerCandidates119_20 := by
  unfold candidates
  have row : edge 8089 = SearchData.row8089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan119_20 : scan edge (search edge 3) 4 [673] 673 innerTail119_20 best = best := by
  rw [innerTail119_20, scan, if_neg (by decide), innerCandidates119_20_checked]
  change scan edge (search edge 3) 4 [673] 673 innerTail119_21
    (if 3 ≤ innerCandidates119_20.length then search edge 3 [673, 8089] innerCandidates119_20 8762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan119_21

def innerTail119_19 : List Nat := 7417 :: innerTail119_20
def innerCandidates119_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates119_19_checked : candidates edge 4 673 7417 innerTail119_20 best.bound = innerCandidates119_19 := by
  unfold candidates
  have row : edge 7417 = SearchData.row7417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan119_19 : scan edge (search edge 3) 4 [673] 673 innerTail119_19 best = best := by
  rw [innerTail119_19, scan, if_neg (by decide), innerCandidates119_19_checked]
  change scan edge (search edge 3) 4 [673] 673 innerTail119_20
    (if 3 ≤ innerCandidates119_19.length then search edge 3 [673, 7417] innerCandidates119_19 8090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan119_20

def innerTail119_18 : List Nat := 7177 :: innerTail119_19
def innerCandidates119_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates119_18_checked : candidates edge 4 673 7177 innerTail119_19 best.bound = innerCandidates119_18 := by
  unfold candidates
  have row : edge 7177 = SearchData.row7177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan119_18 : scan edge (search edge 3) 4 [673] 673 innerTail119_18 best = best := by
  rw [innerTail119_18, scan, if_neg (by decide), innerCandidates119_18_checked]
  change scan edge (search edge 3) 4 [673] 673 innerTail119_19
    (if 3 ≤ innerCandidates119_18.length then search edge 3 [673, 7177] innerCandidates119_18 7850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan119_19

def innerTail119_17 : List Nat := 6841 :: innerTail119_18
def innerCandidates119_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates119_17_checked : candidates edge 4 673 6841 innerTail119_18 best.bound = innerCandidates119_17 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan119_17 : scan edge (search edge 3) 4 [673] 673 innerTail119_17 best = best := by
  rw [innerTail119_17, scan, if_neg (by decide), innerCandidates119_17_checked]
  change scan edge (search edge 3) 4 [673] 673 innerTail119_18
    (if 3 ≤ innerCandidates119_17.length then search edge 3 [673, 6841] innerCandidates119_17 7514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan119_18

def innerTail119_16 : List Nat := 6271 :: innerTail119_17
def innerCandidates119_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates119_16_checked : candidates edge 4 673 6271 innerTail119_17 best.bound = innerCandidates119_16 := by
  unfold candidates
  have row : edge 6271 = SearchData.row6271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan119_16 : scan edge (search edge 3) 4 [673] 673 innerTail119_16 best = best := by
  rw [innerTail119_16, scan, if_neg (by decide), innerCandidates119_16_checked]
  change scan edge (search edge 3) 4 [673] 673 innerTail119_17
    (if 3 ≤ innerCandidates119_16.length then search edge 3 [673, 6271] innerCandidates119_16 6944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan119_17

def innerTail119_15 : List Nat := 4993 :: innerTail119_16
def innerCandidates119_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates119_15_checked : candidates edge 4 673 4993 innerTail119_16 best.bound = innerCandidates119_15 := by
  unfold candidates
  have row : edge 4993 = SearchData.row4993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan119_15 : scan edge (search edge 3) 4 [673] 673 innerTail119_15 best = best := by
  rw [innerTail119_15, scan, if_neg (by decide), innerCandidates119_15_checked]
  change scan edge (search edge 3) 4 [673] 673 innerTail119_16
    (if 3 ≤ innerCandidates119_15.length then search edge 3 [673, 4993] innerCandidates119_15 5666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan119_16

def innerTail119_14 : List Nat := 4813 :: innerTail119_15
def innerCandidates119_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates119_14_checked : candidates edge 4 673 4813 innerTail119_15 best.bound = innerCandidates119_14 := by
  unfold candidates
  have row : edge 4813 = SearchData.row4813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan119_14 : scan edge (search edge 3) 4 [673] 673 innerTail119_14 best = best := by
  rw [innerTail119_14, scan, if_neg (by decide), innerCandidates119_14_checked]
  change scan edge (search edge 3) 4 [673] 673 innerTail119_15
    (if 3 ≤ innerCandidates119_14.length then search edge 3 [673, 4813] innerCandidates119_14 5486 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan119_15

def innerTail119_13 : List Nat := 4597 :: innerTail119_14
def innerCandidates119_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates119_13_checked : candidates edge 4 673 4597 innerTail119_14 best.bound = innerCandidates119_13 := by
  unfold candidates
  have row : edge 4597 = SearchData.row4597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan119_13 : scan edge (search edge 3) 4 [673] 673 innerTail119_13 best = best := by
  rw [innerTail119_13, scan, if_neg (by decide), innerCandidates119_13_checked]
  change scan edge (search edge 3) 4 [673] 673 innerTail119_14
    (if 3 ≤ innerCandidates119_13.length then search edge 3 [673, 4597] innerCandidates119_13 5270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan119_14

def innerTail119_12 : List Nat := 4297 :: innerTail119_13
def innerCandidates119_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates119_12_checked : candidates edge 4 673 4297 innerTail119_13 best.bound = innerCandidates119_12 := by
  unfold candidates
  have row : edge 4297 = SearchData.row4297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan119_12 : scan edge (search edge 3) 4 [673] 673 innerTail119_12 best = best := by
  rw [innerTail119_12, scan, if_neg (by decide), innerCandidates119_12_checked]
  change scan edge (search edge 3) 4 [673] 673 innerTail119_13
    (if 3 ≤ innerCandidates119_12.length then search edge 3 [673, 4297] innerCandidates119_12 4970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan119_13

def innerTail119_11 : List Nat := 3271 :: innerTail119_12
def innerCandidates119_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates119_11_checked : candidates edge 4 673 3271 innerTail119_12 best.bound = innerCandidates119_11 := by
  unfold candidates
  have row : edge 3271 = SearchData.row3271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan119_11 : scan edge (search edge 3) 4 [673] 673 innerTail119_11 best = best := by
  rw [innerTail119_11, scan, if_neg (by decide), innerCandidates119_11_checked]
  change scan edge (search edge 3) 4 [673] 673 innerTail119_12
    (if 3 ≤ innerCandidates119_11.length then search edge 3 [673, 3271] innerCandidates119_11 3944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan119_12

def innerTail119_10 : List Nat := 2659 :: innerTail119_11
def innerCandidates119_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates119_10_checked : candidates edge 4 673 2659 innerTail119_11 best.bound = innerCandidates119_10 := by
  unfold candidates
  have row : edge 2659 = PrimePairCache.smallRow2659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan119_10 : scan edge (search edge 3) 4 [673] 673 innerTail119_10 best = best := by
  rw [innerTail119_10, scan, if_neg (by decide), innerCandidates119_10_checked]
  change scan edge (search edge 3) 4 [673] 673 innerTail119_11
    (if 3 ≤ innerCandidates119_10.length then search edge 3 [673, 2659] innerCandidates119_10 3332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan119_11

def innerTail119_9 : List Nat := 2293 :: innerTail119_10
def innerCandidates119_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates119_9_checked : candidates edge 4 673 2293 innerTail119_10 best.bound = innerCandidates119_9 := by
  unfold candidates
  have row : edge 2293 = PrimePairCache.smallRow2293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan119_9 : scan edge (search edge 3) 4 [673] 673 innerTail119_9 best = best := by
  rw [innerTail119_9, scan, if_neg (by decide), innerCandidates119_9_checked]
  change scan edge (search edge 3) 4 [673] 673 innerTail119_10
    (if 3 ≤ innerCandidates119_9.length then search edge 3 [673, 2293] innerCandidates119_9 2966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan119_10

def innerTail119_8 : List Nat := 2281 :: innerTail119_9
def innerCandidates119_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates119_8_checked : candidates edge 4 673 2281 innerTail119_9 best.bound = innerCandidates119_8 := by
  unfold candidates
  have row : edge 2281 = PrimePairCache.smallRow2281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan119_8 : scan edge (search edge 3) 4 [673] 673 innerTail119_8 best = best := by
  rw [innerTail119_8, scan, if_neg (by decide), innerCandidates119_8_checked]
  change scan edge (search edge 3) 4 [673] 673 innerTail119_9
    (if 3 ≤ innerCandidates119_8.length then search edge 3 [673, 2281] innerCandidates119_8 2954 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan119_9

def innerTail119_7 : List Nat := 1759 :: innerTail119_8
def innerCandidates119_7 : List Nat := [7417]
set_option maxRecDepth 4096 in
theorem innerCandidates119_7_checked : candidates edge 4 673 1759 innerTail119_8 best.bound = innerCandidates119_7 := by
  unfold candidates
  have row : edge 1759 = PrimePairCache.smallRow1759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan119_7 : scan edge (search edge 3) 4 [673] 673 innerTail119_7 best = best := by
  rw [innerTail119_7, scan, if_neg (by decide), innerCandidates119_7_checked]
  change scan edge (search edge 3) 4 [673] 673 innerTail119_8
    (if 3 ≤ innerCandidates119_7.length then search edge 3 [673, 1759] innerCandidates119_7 2432 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan119_8

def innerTail119_6 : List Nat := 1657 :: innerTail119_7
def innerCandidates119_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates119_6_checked : candidates edge 4 673 1657 innerTail119_7 best.bound = innerCandidates119_6 := by
  unfold candidates
  have row : edge 1657 = PrimePairCache.smallRow1657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan119_6 : scan edge (search edge 3) 4 [673] 673 innerTail119_6 best = best := by
  rw [innerTail119_6, scan, if_neg (by decide), innerCandidates119_6_checked]
  change scan edge (search edge 3) 4 [673] 673 innerTail119_7
    (if 3 ≤ innerCandidates119_6.length then search edge 3 [673, 1657] innerCandidates119_6 2330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan119_7

def innerTail119_5 : List Nat := 1579 :: innerTail119_6
def innerCandidates119_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates119_5_checked : candidates edge 4 673 1579 innerTail119_6 best.bound = innerCandidates119_5 := by
  unfold candidates
  have row : edge 1579 = PrimePairCache.smallRow1579.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan119_5 : scan edge (search edge 3) 4 [673] 673 innerTail119_5 best = best := by
  rw [innerTail119_5, scan, if_neg (by decide), innerCandidates119_5_checked]
  change scan edge (search edge 3) 4 [673] 673 innerTail119_6
    (if 3 ≤ innerCandidates119_5.length then search edge 3 [673, 1579] innerCandidates119_5 2252 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan119_6

def innerTail119_4 : List Nat := 1489 :: innerTail119_5
def innerCandidates119_4 : List Nat := [4597]
set_option maxRecDepth 4096 in
theorem innerCandidates119_4_checked : candidates edge 4 673 1489 innerTail119_5 best.bound = innerCandidates119_4 := by
  unfold candidates
  have row : edge 1489 = PrimePairCache.smallRow1489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan119_4 : scan edge (search edge 3) 4 [673] 673 innerTail119_4 best = best := by
  rw [innerTail119_4, scan, if_neg (by decide), innerCandidates119_4_checked]
  change scan edge (search edge 3) 4 [673] 673 innerTail119_5
    (if 3 ≤ innerCandidates119_4.length then search edge 3 [673, 1489] innerCandidates119_4 2162 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan119_5

def innerTail119_3 : List Nat := 1327 :: innerTail119_4
def innerCandidates119_3 : List Nat := [2293]
set_option maxRecDepth 4096 in
theorem innerCandidates119_3_checked : candidates edge 4 673 1327 innerTail119_4 best.bound = innerCandidates119_3 := by
  unfold candidates
  have row : edge 1327 = PrimePairCache.smallRow1327.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan119_3 : scan edge (search edge 3) 4 [673] 673 innerTail119_3 best = best := by
  rw [innerTail119_3, scan, if_neg (by decide), innerCandidates119_3_checked]
  change scan edge (search edge 3) 4 [673] 673 innerTail119_4
    (if 3 ≤ innerCandidates119_3.length then search edge 3 [673, 1327] innerCandidates119_3 2000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan119_4

def innerTail119_2 : List Nat := 1279 :: innerTail119_3
def innerCandidates119_2 : List Nat := [6271]
set_option maxRecDepth 4096 in
theorem innerCandidates119_2_checked : candidates edge 4 673 1279 innerTail119_3 best.bound = innerCandidates119_2 := by
  unfold candidates
  have row : edge 1279 = PrimePairCache.smallRow1279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan119_2 : scan edge (search edge 3) 4 [673] 673 innerTail119_2 best = best := by
  rw [innerTail119_2, scan, if_neg (by decide), innerCandidates119_2_checked]
  change scan edge (search edge 3) 4 [673] 673 innerTail119_3
    (if 3 ≤ innerCandidates119_2.length then search edge 3 [673, 1279] innerCandidates119_2 1952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan119_3

def innerTail119_1 : List Nat := 1117 :: innerTail119_2
def innerCandidates119_1 : List Nat := [1327, 4993]
set_option maxRecDepth 4096 in
theorem innerCandidates119_1_checked : candidates edge 4 673 1117 innerTail119_2 best.bound = innerCandidates119_1 := by
  unfold candidates
  have row : edge 1117 = PrimePairCache.smallRow1117.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan119_1 : scan edge (search edge 3) 4 [673] 673 innerTail119_1 best = best := by
  rw [innerTail119_1, scan, if_neg (by decide), innerCandidates119_1_checked]
  change scan edge (search edge 3) 4 [673] 673 innerTail119_2
    (if 3 ≤ innerCandidates119_1.length then search edge 3 [673, 1117] innerCandidates119_1 1790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan119_2

def innerTail119_0 : List Nat := 769 :: innerTail119_1
def innerCandidates119_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates119_0_checked : candidates edge 4 673 769 innerTail119_1 best.bound = innerCandidates119_0 := by
  unfold candidates
  have row : edge 769 = PrimePairCache.smallRow769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan119_0 : scan edge (search edge 3) 4 [673] 673 innerTail119_0 best = best := by
  rw [innerTail119_0, scan, if_neg (by decide), innerCandidates119_0_checked]
  change scan edge (search edge 3) 4 [673] 673 innerTail119_1
    (if 3 ≤ innerCandidates119_0.length then search edge 3 [673, 769] innerCandidates119_0 1442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan119_1

theorem rootBranch119 : search edge 4 [673] rootCandidates119 673 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [673] 673 innerTail119_0 best = best
  exact innerScan119_0

end ProjectEulerSolutions.PrimePairUpper.Search
