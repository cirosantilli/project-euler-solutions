import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates72 : List Nat := [397, 751, 811, 877, 997, 1063, 1069, 1087, 1213, 1231, 1321, 1423, 1741, 1873, 2221, 2281, 2791, 3061, 3079, 3121, 3259, 3931, 4051, 4423, 4951, 5563, 5569, 5581, 6367, 6379, 6673, 6733, 6997, 7699, 7723, 7873, 7879, 7933]
set_option maxRecDepth 4096 in
theorem rootCandidates72_checked : candidates edge 5 0 379 tail73 best.bound = rootCandidates72 := by
  unfold candidates
  have row : edge 379 = PrimePairCache.smallRow379.contains := by funext q; rfl
  rw [row]
  decide

def innerTail72_38 : List Nat := []
theorem innerScan72_38 : scan edge (search edge 3) 4 [379] 379 innerTail72_38 best = best := rfl

def innerTail72_37 : List Nat := 7933 :: innerTail72_38
def innerCandidates72_37 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates72_37_checked : candidates edge 4 379 7933 innerTail72_38 best.bound = innerCandidates72_37 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_37 : scan edge (search edge 3) 4 [379] 379 innerTail72_37 best = best := by
  rw [innerTail72_37, scan, if_neg (by decide), innerCandidates72_37_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_38
    (if 3 ≤ innerCandidates72_37.length then search edge 3 [379, 7933] innerCandidates72_37 8312 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_38

def innerTail72_36 : List Nat := 7879 :: innerTail72_37
def innerCandidates72_36 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates72_36_checked : candidates edge 4 379 7879 innerTail72_37 best.bound = innerCandidates72_36 := by
  unfold candidates
  have row : edge 7879 = SearchData.row7879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_36 : scan edge (search edge 3) 4 [379] 379 innerTail72_36 best = best := by
  rw [innerTail72_36, scan, if_neg (by decide), innerCandidates72_36_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_37
    (if 3 ≤ innerCandidates72_36.length then search edge 3 [379, 7879] innerCandidates72_36 8258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_37

def innerTail72_35 : List Nat := 7873 :: innerTail72_36
def innerCandidates72_35 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates72_35_checked : candidates edge 4 379 7873 innerTail72_36 best.bound = innerCandidates72_35 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_35 : scan edge (search edge 3) 4 [379] 379 innerTail72_35 best = best := by
  rw [innerTail72_35, scan, if_neg (by decide), innerCandidates72_35_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_36
    (if 3 ≤ innerCandidates72_35.length then search edge 3 [379, 7873] innerCandidates72_35 8252 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_36

def innerTail72_34 : List Nat := 7723 :: innerTail72_35
def innerCandidates72_34 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates72_34_checked : candidates edge 4 379 7723 innerTail72_35 best.bound = innerCandidates72_34 := by
  unfold candidates
  have row : edge 7723 = SearchData.row7723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_34 : scan edge (search edge 3) 4 [379] 379 innerTail72_34 best = best := by
  rw [innerTail72_34, scan, if_neg (by decide), innerCandidates72_34_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_35
    (if 3 ≤ innerCandidates72_34.length then search edge 3 [379, 7723] innerCandidates72_34 8102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_35

def innerTail72_33 : List Nat := 7699 :: innerTail72_34
def innerCandidates72_33 : List Nat := [7873]
set_option maxRecDepth 4096 in
theorem innerCandidates72_33_checked : candidates edge 4 379 7699 innerTail72_34 best.bound = innerCandidates72_33 := by
  unfold candidates
  have row : edge 7699 = SearchData.row7699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_33 : scan edge (search edge 3) 4 [379] 379 innerTail72_33 best = best := by
  rw [innerTail72_33, scan, if_neg (by decide), innerCandidates72_33_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_34
    (if 3 ≤ innerCandidates72_33.length then search edge 3 [379, 7699] innerCandidates72_33 8078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_34

def innerTail72_32 : List Nat := 6997 :: innerTail72_33
def innerCandidates72_32 : List Nat := [7879]
set_option maxRecDepth 4096 in
theorem innerCandidates72_32_checked : candidates edge 4 379 6997 innerTail72_33 best.bound = innerCandidates72_32 := by
  unfold candidates
  have row : edge 6997 = SearchData.row6997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_32 : scan edge (search edge 3) 4 [379] 379 innerTail72_32 best = best := by
  rw [innerTail72_32, scan, if_neg (by decide), innerCandidates72_32_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_33
    (if 3 ≤ innerCandidates72_32.length then search edge 3 [379, 6997] innerCandidates72_32 7376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_33

def innerTail72_31 : List Nat := 6733 :: innerTail72_32
def innerCandidates72_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates72_31_checked : candidates edge 4 379 6733 innerTail72_32 best.bound = innerCandidates72_31 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_31 : scan edge (search edge 3) 4 [379] 379 innerTail72_31 best = best := by
  rw [innerTail72_31, scan, if_neg (by decide), innerCandidates72_31_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_32
    (if 3 ≤ innerCandidates72_31.length then search edge 3 [379, 6733] innerCandidates72_31 7112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_32

def innerTail72_30 : List Nat := 6673 :: innerTail72_31
def innerCandidates72_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates72_30_checked : candidates edge 4 379 6673 innerTail72_31 best.bound = innerCandidates72_30 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_30 : scan edge (search edge 3) 4 [379] 379 innerTail72_30 best = best := by
  rw [innerTail72_30, scan, if_neg (by decide), innerCandidates72_30_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_31
    (if 3 ≤ innerCandidates72_30.length then search edge 3 [379, 6673] innerCandidates72_30 7052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_31

def innerTail72_29 : List Nat := 6379 :: innerTail72_30
def innerCandidates72_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates72_29_checked : candidates edge 4 379 6379 innerTail72_30 best.bound = innerCandidates72_29 := by
  unfold candidates
  have row : edge 6379 = SearchData.row6379.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_29 : scan edge (search edge 3) 4 [379] 379 innerTail72_29 best = best := by
  rw [innerTail72_29, scan, if_neg (by decide), innerCandidates72_29_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_30
    (if 3 ≤ innerCandidates72_29.length then search edge 3 [379, 6379] innerCandidates72_29 6758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_30

def innerTail72_28 : List Nat := 6367 :: innerTail72_29
def innerCandidates72_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates72_28_checked : candidates edge 4 379 6367 innerTail72_29 best.bound = innerCandidates72_28 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_28 : scan edge (search edge 3) 4 [379] 379 innerTail72_28 best = best := by
  rw [innerTail72_28, scan, if_neg (by decide), innerCandidates72_28_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_29
    (if 3 ≤ innerCandidates72_28.length then search edge 3 [379, 6367] innerCandidates72_28 6746 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_29

def innerTail72_27 : List Nat := 5581 :: innerTail72_28
def innerCandidates72_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates72_27_checked : candidates edge 4 379 5581 innerTail72_28 best.bound = innerCandidates72_27 := by
  unfold candidates
  have row : edge 5581 = SearchData.row5581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_27 : scan edge (search edge 3) 4 [379] 379 innerTail72_27 best = best := by
  rw [innerTail72_27, scan, if_neg (by decide), innerCandidates72_27_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_28
    (if 3 ≤ innerCandidates72_27.length then search edge 3 [379, 5581] innerCandidates72_27 5960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_28

def innerTail72_26 : List Nat := 5569 :: innerTail72_27
def innerCandidates72_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates72_26_checked : candidates edge 4 379 5569 innerTail72_27 best.bound = innerCandidates72_26 := by
  unfold candidates
  have row : edge 5569 = SearchData.row5569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_26 : scan edge (search edge 3) 4 [379] 379 innerTail72_26 best = best := by
  rw [innerTail72_26, scan, if_neg (by decide), innerCandidates72_26_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_27
    (if 3 ≤ innerCandidates72_26.length then search edge 3 [379, 5569] innerCandidates72_26 5948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_27

def innerTail72_25 : List Nat := 5563 :: innerTail72_26
def innerCandidates72_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates72_25_checked : candidates edge 4 379 5563 innerTail72_26 best.bound = innerCandidates72_25 := by
  unfold candidates
  have row : edge 5563 = SearchData.row5563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_25 : scan edge (search edge 3) 4 [379] 379 innerTail72_25 best = best := by
  rw [innerTail72_25, scan, if_neg (by decide), innerCandidates72_25_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_26
    (if 3 ≤ innerCandidates72_25.length then search edge 3 [379, 5563] innerCandidates72_25 5942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_26

def innerTail72_24 : List Nat := 4951 :: innerTail72_25
def innerCandidates72_24 : List Nat := [5563]
set_option maxRecDepth 4096 in
theorem innerCandidates72_24_checked : candidates edge 4 379 4951 innerTail72_25 best.bound = innerCandidates72_24 := by
  unfold candidates
  have row : edge 4951 = SearchData.row4951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_24 : scan edge (search edge 3) 4 [379] 379 innerTail72_24 best = best := by
  rw [innerTail72_24, scan, if_neg (by decide), innerCandidates72_24_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_25
    (if 3 ≤ innerCandidates72_24.length then search edge 3 [379, 4951] innerCandidates72_24 5330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_25

def innerTail72_23 : List Nat := 4423 :: innerTail72_24
def innerCandidates72_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates72_23_checked : candidates edge 4 379 4423 innerTail72_24 best.bound = innerCandidates72_23 := by
  unfold candidates
  have row : edge 4423 = SearchData.row4423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_23 : scan edge (search edge 3) 4 [379] 379 innerTail72_23 best = best := by
  rw [innerTail72_23, scan, if_neg (by decide), innerCandidates72_23_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_24
    (if 3 ≤ innerCandidates72_23.length then search edge 3 [379, 4423] innerCandidates72_23 4802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_24

def innerTail72_22 : List Nat := 4051 :: innerTail72_23
def innerCandidates72_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates72_22_checked : candidates edge 4 379 4051 innerTail72_23 best.bound = innerCandidates72_22 := by
  unfold candidates
  have row : edge 4051 = SearchData.row4051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_22 : scan edge (search edge 3) 4 [379] 379 innerTail72_22 best = best := by
  rw [innerTail72_22, scan, if_neg (by decide), innerCandidates72_22_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_23
    (if 3 ≤ innerCandidates72_22.length then search edge 3 [379, 4051] innerCandidates72_22 4430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_23

def innerTail72_21 : List Nat := 3931 :: innerTail72_22
def innerCandidates72_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates72_21_checked : candidates edge 4 379 3931 innerTail72_22 best.bound = innerCandidates72_21 := by
  unfold candidates
  have row : edge 3931 = SearchData.row3931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_21 : scan edge (search edge 3) 4 [379] 379 innerTail72_21 best = best := by
  rw [innerTail72_21, scan, if_neg (by decide), innerCandidates72_21_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_22
    (if 3 ≤ innerCandidates72_21.length then search edge 3 [379, 3931] innerCandidates72_21 4310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_22

def innerTail72_20 : List Nat := 3259 :: innerTail72_21
def innerCandidates72_20 : List Nat := [5569]
set_option maxRecDepth 4096 in
theorem innerCandidates72_20_checked : candidates edge 4 379 3259 innerTail72_21 best.bound = innerCandidates72_20 := by
  unfold candidates
  have row : edge 3259 = SearchData.row3259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_20 : scan edge (search edge 3) 4 [379] 379 innerTail72_20 best = best := by
  rw [innerTail72_20, scan, if_neg (by decide), innerCandidates72_20_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_21
    (if 3 ≤ innerCandidates72_20.length then search edge 3 [379, 3259] innerCandidates72_20 3638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_21

def innerTail72_19 : List Nat := 3121 :: innerTail72_20
def innerCandidates72_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates72_19_checked : candidates edge 4 379 3121 innerTail72_20 best.bound = innerCandidates72_19 := by
  unfold candidates
  have row : edge 3121 = SearchData.row3121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_19 : scan edge (search edge 3) 4 [379] 379 innerTail72_19 best = best := by
  rw [innerTail72_19, scan, if_neg (by decide), innerCandidates72_19_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_20
    (if 3 ≤ innerCandidates72_19.length then search edge 3 [379, 3121] innerCandidates72_19 3500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_20

def innerTail72_18 : List Nat := 3079 :: innerTail72_19
def innerCandidates72_18 : List Nat := [7699]
set_option maxRecDepth 4096 in
theorem innerCandidates72_18_checked : candidates edge 4 379 3079 innerTail72_19 best.bound = innerCandidates72_18 := by
  unfold candidates
  have row : edge 3079 = SearchData.row3079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_18 : scan edge (search edge 3) 4 [379] 379 innerTail72_18 best = best := by
  rw [innerTail72_18, scan, if_neg (by decide), innerCandidates72_18_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_19
    (if 3 ≤ innerCandidates72_18.length then search edge 3 [379, 3079] innerCandidates72_18 3458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_19

def innerTail72_17 : List Nat := 3061 :: innerTail72_18
def innerCandidates72_17 : List Nat := [6673]
set_option maxRecDepth 4096 in
theorem innerCandidates72_17_checked : candidates edge 4 379 3061 innerTail72_18 best.bound = innerCandidates72_17 := by
  unfold candidates
  have row : edge 3061 = SearchData.row3061.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_17 : scan edge (search edge 3) 4 [379] 379 innerTail72_17 best = best := by
  rw [innerTail72_17, scan, if_neg (by decide), innerCandidates72_17_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_18
    (if 3 ≤ innerCandidates72_17.length then search edge 3 [379, 3061] innerCandidates72_17 3440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_18

def innerTail72_16 : List Nat := 2791 :: innerTail72_17
def innerCandidates72_16 : List Nat := [6379]
set_option maxRecDepth 4096 in
theorem innerCandidates72_16_checked : candidates edge 4 379 2791 innerTail72_17 best.bound = innerCandidates72_16 := by
  unfold candidates
  have row : edge 2791 = SearchData.row2791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_16 : scan edge (search edge 3) 4 [379] 379 innerTail72_16 best = best := by
  rw [innerTail72_16, scan, if_neg (by decide), innerCandidates72_16_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_17
    (if 3 ≤ innerCandidates72_16.length then search edge 3 [379, 2791] innerCandidates72_16 3170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_17

def innerTail72_15 : List Nat := 2281 :: innerTail72_16
def innerCandidates72_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates72_15_checked : candidates edge 4 379 2281 innerTail72_16 best.bound = innerCandidates72_15 := by
  unfold candidates
  have row : edge 2281 = PrimePairCache.smallRow2281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_15 : scan edge (search edge 3) 4 [379] 379 innerTail72_15 best = best := by
  rw [innerTail72_15, scan, if_neg (by decide), innerCandidates72_15_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_16
    (if 3 ≤ innerCandidates72_15.length then search edge 3 [379, 2281] innerCandidates72_15 2660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_16

def innerTail72_14 : List Nat := 2221 :: innerTail72_15
def innerCandidates72_14 : List Nat := [5581, 6673, 7873]
set_option maxRecDepth 4096 in
theorem innerCandidates72_14_checked : candidates edge 4 379 2221 innerTail72_15 best.bound = innerCandidates72_14 := by
  unfold candidates
  have row : edge 2221 = PrimePairCache.smallRow2221.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild72_14 : search edge 3 [379, 2221] innerCandidates72_14 2600 best = best := by decide

theorem innerScan72_14 : scan edge (search edge 3) 4 [379] 379 innerTail72_14 best = best := by
  rw [innerTail72_14, scan, if_neg (by decide), innerCandidates72_14_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_15
    (if 3 ≤ innerCandidates72_14.length then search edge 3 [379, 2221] innerCandidates72_14 2600 best else best) = best
  rw [if_pos (by decide), innerChild72_14]
  exact innerScan72_15

def innerTail72_13 : List Nat := 1873 :: innerTail72_14
def innerCandidates72_13 : List Nat := [5569, 6997]
set_option maxRecDepth 4096 in
theorem innerCandidates72_13_checked : candidates edge 4 379 1873 innerTail72_14 best.bound = innerCandidates72_13 := by
  unfold candidates
  have row : edge 1873 = PrimePairCache.smallRow1873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_13 : scan edge (search edge 3) 4 [379] 379 innerTail72_13 best = best := by
  rw [innerTail72_13, scan, if_neg (by decide), innerCandidates72_13_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_14
    (if 3 ≤ innerCandidates72_13.length then search edge 3 [379, 1873] innerCandidates72_13 2252 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_14

def innerTail72_12 : List Nat := 1741 :: innerTail72_13
def innerCandidates72_12 : List Nat := [2281, 5581]
set_option maxRecDepth 4096 in
theorem innerCandidates72_12_checked : candidates edge 4 379 1741 innerTail72_13 best.bound = innerCandidates72_12 := by
  unfold candidates
  have row : edge 1741 = PrimePairCache.smallRow1741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_12 : scan edge (search edge 3) 4 [379] 379 innerTail72_12 best = best := by
  rw [innerTail72_12, scan, if_neg (by decide), innerCandidates72_12_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_13
    (if 3 ≤ innerCandidates72_12.length then search edge 3 [379, 1741] innerCandidates72_12 2120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_13

def innerTail72_11 : List Nat := 1423 :: innerTail72_12
def innerCandidates72_11 : List Nat := [6367]
set_option maxRecDepth 4096 in
theorem innerCandidates72_11_checked : candidates edge 4 379 1423 innerTail72_12 best.bound = innerCandidates72_11 := by
  unfold candidates
  have row : edge 1423 = PrimePairCache.smallRow1423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_11 : scan edge (search edge 3) 4 [379] 379 innerTail72_11 best = best := by
  rw [innerTail72_11, scan, if_neg (by decide), innerCandidates72_11_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_12
    (if 3 ≤ innerCandidates72_11.length then search edge 3 [379, 1423] innerCandidates72_11 1802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_12

def innerTail72_10 : List Nat := 1321 :: innerTail72_11
def innerCandidates72_10 : List Nat := [5581]
set_option maxRecDepth 4096 in
theorem innerCandidates72_10_checked : candidates edge 4 379 1321 innerTail72_11 best.bound = innerCandidates72_10 := by
  unfold candidates
  have row : edge 1321 = PrimePairCache.smallRow1321.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_10 : scan edge (search edge 3) 4 [379] 379 innerTail72_10 best = best := by
  rw [innerTail72_10, scan, if_neg (by decide), innerCandidates72_10_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_11
    (if 3 ≤ innerCandidates72_10.length then search edge 3 [379, 1321] innerCandidates72_10 1700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_11

def innerTail72_9 : List Nat := 1231 :: innerTail72_10
def innerCandidates72_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates72_9_checked : candidates edge 4 379 1231 innerTail72_10 best.bound = innerCandidates72_9 := by
  unfold candidates
  have row : edge 1231 = PrimePairCache.smallRow1231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_9 : scan edge (search edge 3) 4 [379] 379 innerTail72_9 best = best := by
  rw [innerTail72_9, scan, if_neg (by decide), innerCandidates72_9_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_10
    (if 3 ≤ innerCandidates72_9.length then search edge 3 [379, 1231] innerCandidates72_9 1610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_10

def innerTail72_8 : List Nat := 1213 :: innerTail72_9
def innerCandidates72_8 : List Nat := [3061]
set_option maxRecDepth 4096 in
theorem innerCandidates72_8_checked : candidates edge 4 379 1213 innerTail72_9 best.bound = innerCandidates72_8 := by
  unfold candidates
  have row : edge 1213 = PrimePairCache.smallRow1213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_8 : scan edge (search edge 3) 4 [379] 379 innerTail72_8 best = best := by
  rw [innerTail72_8, scan, if_neg (by decide), innerCandidates72_8_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_9
    (if 3 ≤ innerCandidates72_8.length then search edge 3 [379, 1213] innerCandidates72_8 1592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_9

def innerTail72_7 : List Nat := 1087 :: innerTail72_8
def innerCandidates72_7 : List Nat := [5569]
set_option maxRecDepth 4096 in
theorem innerCandidates72_7_checked : candidates edge 4 379 1087 innerTail72_8 best.bound = innerCandidates72_7 := by
  unfold candidates
  have row : edge 1087 = PrimePairCache.smallRow1087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_7 : scan edge (search edge 3) 4 [379] 379 innerTail72_7 best = best := by
  rw [innerTail72_7, scan, if_neg (by decide), innerCandidates72_7_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_8
    (if 3 ≤ innerCandidates72_7.length then search edge 3 [379, 1087] innerCandidates72_7 1466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_8

def innerTail72_6 : List Nat := 1069 :: innerTail72_7
def innerCandidates72_6 : List Nat := [1321, 2281, 3061]
set_option maxRecDepth 4096 in
theorem innerCandidates72_6_checked : candidates edge 4 379 1069 innerTail72_7 best.bound = innerCandidates72_6 := by
  unfold candidates
  have row : edge 1069 = PrimePairCache.smallRow1069.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild72_6 : search edge 3 [379, 1069] innerCandidates72_6 1448 best = best := by decide

theorem innerScan72_6 : scan edge (search edge 3) 4 [379] 379 innerTail72_6 best = best := by
  rw [innerTail72_6, scan, if_neg (by decide), innerCandidates72_6_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_7
    (if 3 ≤ innerCandidates72_6.length then search edge 3 [379, 1069] innerCandidates72_6 1448 best else best) = best
  rw [if_pos (by decide), innerChild72_6]
  exact innerScan72_7

def innerTail72_5 : List Nat := 1063 :: innerTail72_6
def innerCandidates72_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates72_5_checked : candidates edge 4 379 1063 innerTail72_6 best.bound = innerCandidates72_5 := by
  unfold candidates
  have row : edge 1063 = PrimePairCache.smallRow1063.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_5 : scan edge (search edge 3) 4 [379] 379 innerTail72_5 best = best := by
  rw [innerTail72_5, scan, if_neg (by decide), innerCandidates72_5_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_6
    (if 3 ≤ innerCandidates72_5.length then search edge 3 [379, 1063] innerCandidates72_5 1442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_6

def innerTail72_4 : List Nat := 997 :: innerTail72_5
def innerCandidates72_4 : List Nat := [5569]
set_option maxRecDepth 4096 in
theorem innerCandidates72_4_checked : candidates edge 4 379 997 innerTail72_5 best.bound = innerCandidates72_4 := by
  unfold candidates
  have row : edge 997 = PrimePairCache.smallRow997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_4 : scan edge (search edge 3) 4 [379] 379 innerTail72_4 best = best := by
  rw [innerTail72_4, scan, if_neg (by decide), innerCandidates72_4_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_5
    (if 3 ≤ innerCandidates72_4.length then search edge 3 [379, 997] innerCandidates72_4 1376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_5

def innerTail72_3 : List Nat := 877 :: innerTail72_4
def innerCandidates72_3 : List Nat := [997]
set_option maxRecDepth 4096 in
theorem innerCandidates72_3_checked : candidates edge 4 379 877 innerTail72_4 best.bound = innerCandidates72_3 := by
  unfold candidates
  have row : edge 877 = PrimePairCache.smallRow877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_3 : scan edge (search edge 3) 4 [379] 379 innerTail72_3 best = best := by
  rw [innerTail72_3, scan, if_neg (by decide), innerCandidates72_3_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_4
    (if 3 ≤ innerCandidates72_3.length then search edge 3 [379, 877] innerCandidates72_3 1256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_4

def innerTail72_2 : List Nat := 811 :: innerTail72_3
def innerCandidates72_2 : List Nat := [997]
set_option maxRecDepth 4096 in
theorem innerCandidates72_2_checked : candidates edge 4 379 811 innerTail72_3 best.bound = innerCandidates72_2 := by
  unfold candidates
  have row : edge 811 = PrimePairCache.smallRow811.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_2 : scan edge (search edge 3) 4 [379] 379 innerTail72_2 best = best := by
  rw [innerTail72_2, scan, if_neg (by decide), innerCandidates72_2_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_3
    (if 3 ≤ innerCandidates72_2.length then search edge 3 [379, 811] innerCandidates72_2 1190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_3

def innerTail72_1 : List Nat := 751 :: innerTail72_2
def innerCandidates72_1 : List Nat := [997]
set_option maxRecDepth 4096 in
theorem innerCandidates72_1_checked : candidates edge 4 379 751 innerTail72_2 best.bound = innerCandidates72_1 := by
  unfold candidates
  have row : edge 751 = PrimePairCache.smallRow751.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_1 : scan edge (search edge 3) 4 [379] 379 innerTail72_1 best = best := by
  rw [innerTail72_1, scan, if_neg (by decide), innerCandidates72_1_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_2
    (if 3 ≤ innerCandidates72_1.length then search edge 3 [379, 751] innerCandidates72_1 1130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_2

def innerTail72_0 : List Nat := 397 :: innerTail72_1
def innerCandidates72_0 : List Nat := [1063, 7933]
set_option maxRecDepth 4096 in
theorem innerCandidates72_0_checked : candidates edge 4 379 397 innerTail72_1 best.bound = innerCandidates72_0 := by
  unfold candidates
  have row : edge 397 = PrimePairCache.smallRow397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan72_0 : scan edge (search edge 3) 4 [379] 379 innerTail72_0 best = best := by
  rw [innerTail72_0, scan, if_neg (by decide), innerCandidates72_0_checked]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_1
    (if 3 ≤ innerCandidates72_0.length then search edge 3 [379, 397] innerCandidates72_0 776 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan72_1

theorem rootBranch72 : search edge 4 [379] rootCandidates72 379 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [379] 379 innerTail72_0 best = best
  exact innerScan72_0

def rootCandidates71 : List Nat := [661, 823, 859, 937, 1009, 1237, 1447, 1531, 1867, 2161, 2467, 3967, 4519, 4903, 4957, 5101, 5449, 5881, 6133, 6163, 6793, 6823, 7057, 7351, 7417]
set_option maxRecDepth 4096 in
theorem rootCandidates71_checked : candidates edge 5 0 373 tail72 best.bound = rootCandidates71 := by
  unfold candidates
  have row : edge 373 = PrimePairCache.smallRow373.contains := by funext q; rfl
  rw [row]
  decide

def innerTail71_25 : List Nat := []
theorem innerScan71_25 : scan edge (search edge 3) 4 [373] 373 innerTail71_25 best = best := rfl

def innerTail71_24 : List Nat := 7417 :: innerTail71_25
def innerCandidates71_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates71_24_checked : candidates edge 4 373 7417 innerTail71_25 best.bound = innerCandidates71_24 := by
  unfold candidates
  have row : edge 7417 = SearchData.row7417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_24 : scan edge (search edge 3) 4 [373] 373 innerTail71_24 best = best := by
  rw [innerTail71_24, scan, if_neg (by decide), innerCandidates71_24_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_25
    (if 3 ≤ innerCandidates71_24.length then search edge 3 [373, 7417] innerCandidates71_24 7790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_25

def innerTail71_23 : List Nat := 7351 :: innerTail71_24
def innerCandidates71_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates71_23_checked : candidates edge 4 373 7351 innerTail71_24 best.bound = innerCandidates71_23 := by
  unfold candidates
  have row : edge 7351 = SearchData.row7351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_23 : scan edge (search edge 3) 4 [373] 373 innerTail71_23 best = best := by
  rw [innerTail71_23, scan, if_neg (by decide), innerCandidates71_23_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_24
    (if 3 ≤ innerCandidates71_23.length then search edge 3 [373, 7351] innerCandidates71_23 7724 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_24

def innerTail71_22 : List Nat := 7057 :: innerTail71_23
def innerCandidates71_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates71_22_checked : candidates edge 4 373 7057 innerTail71_23 best.bound = innerCandidates71_22 := by
  unfold candidates
  have row : edge 7057 = SearchData.row7057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_22 : scan edge (search edge 3) 4 [373] 373 innerTail71_22 best = best := by
  rw [innerTail71_22, scan, if_neg (by decide), innerCandidates71_22_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_23
    (if 3 ≤ innerCandidates71_22.length then search edge 3 [373, 7057] innerCandidates71_22 7430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_23

def innerTail71_21 : List Nat := 6823 :: innerTail71_22
def innerCandidates71_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates71_21_checked : candidates edge 4 373 6823 innerTail71_22 best.bound = innerCandidates71_21 := by
  unfold candidates
  have row : edge 6823 = SearchData.row6823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_21 : scan edge (search edge 3) 4 [373] 373 innerTail71_21 best = best := by
  rw [innerTail71_21, scan, if_neg (by decide), innerCandidates71_21_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_22
    (if 3 ≤ innerCandidates71_21.length then search edge 3 [373, 6823] innerCandidates71_21 7196 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_22

def innerTail71_20 : List Nat := 6793 :: innerTail71_21
def innerCandidates71_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates71_20_checked : candidates edge 4 373 6793 innerTail71_21 best.bound = innerCandidates71_20 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_20 : scan edge (search edge 3) 4 [373] 373 innerTail71_20 best = best := by
  rw [innerTail71_20, scan, if_neg (by decide), innerCandidates71_20_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_21
    (if 3 ≤ innerCandidates71_20.length then search edge 3 [373, 6793] innerCandidates71_20 7166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_21

def innerTail71_19 : List Nat := 6163 :: innerTail71_20
def innerCandidates71_19 : List Nat := [6793]
set_option maxRecDepth 4096 in
theorem innerCandidates71_19_checked : candidates edge 4 373 6163 innerTail71_20 best.bound = innerCandidates71_19 := by
  unfold candidates
  have row : edge 6163 = SearchData.row6163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_19 : scan edge (search edge 3) 4 [373] 373 innerTail71_19 best = best := by
  rw [innerTail71_19, scan, if_neg (by decide), innerCandidates71_19_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_20
    (if 3 ≤ innerCandidates71_19.length then search edge 3 [373, 6163] innerCandidates71_19 6536 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_20

def innerTail71_18 : List Nat := 6133 :: innerTail71_19
def innerCandidates71_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates71_18_checked : candidates edge 4 373 6133 innerTail71_19 best.bound = innerCandidates71_18 := by
  unfold candidates
  have row : edge 6133 = SearchData.row6133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_18 : scan edge (search edge 3) 4 [373] 373 innerTail71_18 best = best := by
  rw [innerTail71_18, scan, if_neg (by decide), innerCandidates71_18_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_19
    (if 3 ≤ innerCandidates71_18.length then search edge 3 [373, 6133] innerCandidates71_18 6506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_19

def innerTail71_17 : List Nat := 5881 :: innerTail71_18
def innerCandidates71_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates71_17_checked : candidates edge 4 373 5881 innerTail71_18 best.bound = innerCandidates71_17 := by
  unfold candidates
  have row : edge 5881 = SearchData.row5881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_17 : scan edge (search edge 3) 4 [373] 373 innerTail71_17 best = best := by
  rw [innerTail71_17, scan, if_neg (by decide), innerCandidates71_17_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_18
    (if 3 ≤ innerCandidates71_17.length then search edge 3 [373, 5881] innerCandidates71_17 6254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_18

def innerTail71_16 : List Nat := 5449 :: innerTail71_17
def innerCandidates71_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates71_16_checked : candidates edge 4 373 5449 innerTail71_17 best.bound = innerCandidates71_16 := by
  unfold candidates
  have row : edge 5449 = SearchData.row5449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_16 : scan edge (search edge 3) 4 [373] 373 innerTail71_16 best = best := by
  rw [innerTail71_16, scan, if_neg (by decide), innerCandidates71_16_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_17
    (if 3 ≤ innerCandidates71_16.length then search edge 3 [373, 5449] innerCandidates71_16 5822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_17

def innerTail71_15 : List Nat := 5101 :: innerTail71_16
def innerCandidates71_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates71_15_checked : candidates edge 4 373 5101 innerTail71_16 best.bound = innerCandidates71_15 := by
  unfold candidates
  have row : edge 5101 = SearchData.row5101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_15 : scan edge (search edge 3) 4 [373] 373 innerTail71_15 best = best := by
  rw [innerTail71_15, scan, if_neg (by decide), innerCandidates71_15_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_16
    (if 3 ≤ innerCandidates71_15.length then search edge 3 [373, 5101] innerCandidates71_15 5474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_16

def innerTail71_14 : List Nat := 4957 :: innerTail71_15
def innerCandidates71_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates71_14_checked : candidates edge 4 373 4957 innerTail71_15 best.bound = innerCandidates71_14 := by
  unfold candidates
  have row : edge 4957 = SearchData.row4957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_14 : scan edge (search edge 3) 4 [373] 373 innerTail71_14 best = best := by
  rw [innerTail71_14, scan, if_neg (by decide), innerCandidates71_14_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_15
    (if 3 ≤ innerCandidates71_14.length then search edge 3 [373, 4957] innerCandidates71_14 5330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_15

def innerTail71_13 : List Nat := 4903 :: innerTail71_14
def innerCandidates71_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates71_13_checked : candidates edge 4 373 4903 innerTail71_14 best.bound = innerCandidates71_13 := by
  unfold candidates
  have row : edge 4903 = SearchData.row4903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_13 : scan edge (search edge 3) 4 [373] 373 innerTail71_13 best = best := by
  rw [innerTail71_13, scan, if_neg (by decide), innerCandidates71_13_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_14
    (if 3 ≤ innerCandidates71_13.length then search edge 3 [373, 4903] innerCandidates71_13 5276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_14

def innerTail71_12 : List Nat := 4519 :: innerTail71_13
def innerCandidates71_12 : List Nat := [6133]
set_option maxRecDepth 4096 in
theorem innerCandidates71_12_checked : candidates edge 4 373 4519 innerTail71_13 best.bound = innerCandidates71_12 := by
  unfold candidates
  have row : edge 4519 = SearchData.row4519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_12 : scan edge (search edge 3) 4 [373] 373 innerTail71_12 best = best := by
  rw [innerTail71_12, scan, if_neg (by decide), innerCandidates71_12_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_13
    (if 3 ≤ innerCandidates71_12.length then search edge 3 [373, 4519] innerCandidates71_12 4892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_13

def innerTail71_11 : List Nat := 3967 :: innerTail71_12
def innerCandidates71_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates71_11_checked : candidates edge 4 373 3967 innerTail71_12 best.bound = innerCandidates71_11 := by
  unfold candidates
  have row : edge 3967 = SearchData.row3967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_11 : scan edge (search edge 3) 4 [373] 373 innerTail71_11 best = best := by
  rw [innerTail71_11, scan, if_neg (by decide), innerCandidates71_11_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_12
    (if 3 ≤ innerCandidates71_11.length then search edge 3 [373, 3967] innerCandidates71_11 4340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_12

def innerTail71_10 : List Nat := 2467 :: innerTail71_11
def innerCandidates71_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates71_10_checked : candidates edge 4 373 2467 innerTail71_11 best.bound = innerCandidates71_10 := by
  unfold candidates
  have row : edge 2467 = PrimePairCache.smallRow2467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_10 : scan edge (search edge 3) 4 [373] 373 innerTail71_10 best = best := by
  rw [innerTail71_10, scan, if_neg (by decide), innerCandidates71_10_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_11
    (if 3 ≤ innerCandidates71_10.length then search edge 3 [373, 2467] innerCandidates71_10 2840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_11

def innerTail71_9 : List Nat := 2161 :: innerTail71_10
def innerCandidates71_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates71_9_checked : candidates edge 4 373 2161 innerTail71_10 best.bound = innerCandidates71_9 := by
  unfold candidates
  have row : edge 2161 = PrimePairCache.smallRow2161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_9 : scan edge (search edge 3) 4 [373] 373 innerTail71_9 best = best := by
  rw [innerTail71_9, scan, if_neg (by decide), innerCandidates71_9_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_10
    (if 3 ≤ innerCandidates71_9.length then search edge 3 [373, 2161] innerCandidates71_9 2534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_10

def innerTail71_8 : List Nat := 1867 :: innerTail71_9
def innerCandidates71_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates71_8_checked : candidates edge 4 373 1867 innerTail71_9 best.bound = innerCandidates71_8 := by
  unfold candidates
  have row : edge 1867 = PrimePairCache.smallRow1867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_8 : scan edge (search edge 3) 4 [373] 373 innerTail71_8 best = best := by
  rw [innerTail71_8, scan, if_neg (by decide), innerCandidates71_8_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_9
    (if 3 ≤ innerCandidates71_8.length then search edge 3 [373, 1867] innerCandidates71_8 2240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_9

def innerTail71_7 : List Nat := 1531 :: innerTail71_8
def innerCandidates71_7 : List Nat := [2467]
set_option maxRecDepth 4096 in
theorem innerCandidates71_7_checked : candidates edge 4 373 1531 innerTail71_8 best.bound = innerCandidates71_7 := by
  unfold candidates
  have row : edge 1531 = PrimePairCache.smallRow1531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_7 : scan edge (search edge 3) 4 [373] 373 innerTail71_7 best = best := by
  rw [innerTail71_7, scan, if_neg (by decide), innerCandidates71_7_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_8
    (if 3 ≤ innerCandidates71_7.length then search edge 3 [373, 1531] innerCandidates71_7 1904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_8

def innerTail71_6 : List Nat := 1447 :: innerTail71_7
def innerCandidates71_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates71_6_checked : candidates edge 4 373 1447 innerTail71_7 best.bound = innerCandidates71_6 := by
  unfold candidates
  have row : edge 1447 = PrimePairCache.smallRow1447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_6 : scan edge (search edge 3) 4 [373] 373 innerTail71_6 best = best := by
  rw [innerTail71_6, scan, if_neg (by decide), innerCandidates71_6_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_7
    (if 3 ≤ innerCandidates71_6.length then search edge 3 [373, 1447] innerCandidates71_6 1820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_7

def innerTail71_5 : List Nat := 1237 :: innerTail71_6
def innerCandidates71_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates71_5_checked : candidates edge 4 373 1237 innerTail71_6 best.bound = innerCandidates71_5 := by
  unfold candidates
  have row : edge 1237 = PrimePairCache.smallRow1237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_5 : scan edge (search edge 3) 4 [373] 373 innerTail71_5 best = best := by
  rw [innerTail71_5, scan, if_neg (by decide), innerCandidates71_5_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_6
    (if 3 ≤ innerCandidates71_5.length then search edge 3 [373, 1237] innerCandidates71_5 1610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_6

def innerTail71_4 : List Nat := 1009 :: innerTail71_5
def innerCandidates71_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates71_4_checked : candidates edge 4 373 1009 innerTail71_5 best.bound = innerCandidates71_4 := by
  unfold candidates
  have row : edge 1009 = PrimePairCache.smallRow1009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_4 : scan edge (search edge 3) 4 [373] 373 innerTail71_4 best = best := by
  rw [innerTail71_4, scan, if_neg (by decide), innerCandidates71_4_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_5
    (if 3 ≤ innerCandidates71_4.length then search edge 3 [373, 1009] innerCandidates71_4 1382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_5

def innerTail71_3 : List Nat := 937 :: innerTail71_4
def innerCandidates71_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates71_3_checked : candidates edge 4 373 937 innerTail71_4 best.bound = innerCandidates71_3 := by
  unfold candidates
  have row : edge 937 = PrimePairCache.smallRow937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_3 : scan edge (search edge 3) 4 [373] 373 innerTail71_3 best = best := by
  rw [innerTail71_3, scan, if_neg (by decide), innerCandidates71_3_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_4
    (if 3 ≤ innerCandidates71_3.length then search edge 3 [373, 937] innerCandidates71_3 1310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_4

def innerTail71_2 : List Nat := 859 :: innerTail71_3
def innerCandidates71_2 : List Nat := [7417]
set_option maxRecDepth 4096 in
theorem innerCandidates71_2_checked : candidates edge 4 373 859 innerTail71_3 best.bound = innerCandidates71_2 := by
  unfold candidates
  have row : edge 859 = PrimePairCache.smallRow859.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_2 : scan edge (search edge 3) 4 [373] 373 innerTail71_2 best = best := by
  rw [innerTail71_2, scan, if_neg (by decide), innerCandidates71_2_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_3
    (if 3 ≤ innerCandidates71_2.length then search edge 3 [373, 859] innerCandidates71_2 1232 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_3

def innerTail71_1 : List Nat := 823 :: innerTail71_2
def innerCandidates71_1 : List Nat := [5881]
set_option maxRecDepth 4096 in
theorem innerCandidates71_1_checked : candidates edge 4 373 823 innerTail71_2 best.bound = innerCandidates71_1 := by
  unfold candidates
  have row : edge 823 = PrimePairCache.smallRow823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_1 : scan edge (search edge 3) 4 [373] 373 innerTail71_1 best = best := by
  rw [innerTail71_1, scan, if_neg (by decide), innerCandidates71_1_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_2
    (if 3 ≤ innerCandidates71_1.length then search edge 3 [373, 823] innerCandidates71_1 1196 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_2

def innerTail71_0 : List Nat := 661 :: innerTail71_1
def innerCandidates71_0 : List Nat := [1237, 1447]
set_option maxRecDepth 4096 in
theorem innerCandidates71_0_checked : candidates edge 4 373 661 innerTail71_1 best.bound = innerCandidates71_0 := by
  unfold candidates
  have row : edge 661 = PrimePairCache.smallRow661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan71_0 : scan edge (search edge 3) 4 [373] 373 innerTail71_0 best = best := by
  rw [innerTail71_0, scan, if_neg (by decide), innerCandidates71_0_checked]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_1
    (if 3 ≤ innerCandidates71_0.length then search edge 3 [373, 661] innerCandidates71_0 1034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan71_1

theorem rootBranch71 : search edge 4 [373] rootCandidates71 373 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [373] 373 innerTail71_0 best = best
  exact innerScan71_0

end ProjectEulerSolutions.PrimePairUpper.Search
