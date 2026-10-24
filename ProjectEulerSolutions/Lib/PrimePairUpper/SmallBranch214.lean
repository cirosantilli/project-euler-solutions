import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates74 : List Nat := [761, 797, 947, 971, 1259, 1493, 1913, 2087, 2309, 2333, 2393, 2459, 2969, 3137, 3191, 3299, 3719, 4049, 4409, 4637, 4703, 4733, 5231, 5351, 6029, 6089, 6257, 6947, 7499, 7703, 7727, 7853, 7937, 8087, 8123]
set_option maxRecDepth 4096 in
theorem rootCandidates74_checked : candidates edge 5 0 389 tail75 best.bound = rootCandidates74 := by
  unfold candidates
  have row : edge 389 = PrimePairCache.smallRow389.contains := by funext q; rfl
  rw [row]
  decide

def innerTail74_35 : List Nat := []
theorem innerScan74_35 : scan edge (search edge 3) 4 [389] 389 innerTail74_35 best = best := rfl

def innerTail74_34 : List Nat := 8123 :: innerTail74_35
def innerCandidates74_34 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates74_34_checked : candidates edge 4 389 8123 innerTail74_35 best.bound = innerCandidates74_34 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_34 : scan edge (search edge 3) 4 [389] 389 innerTail74_34 best = best := by
  rw [innerTail74_34, scan, if_neg (by decide), innerCandidates74_34_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_35
    (if 3 ≤ innerCandidates74_34.length then search edge 3 [389, 8123] innerCandidates74_34 8512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_35

def innerTail74_33 : List Nat := 8087 :: innerTail74_34
def innerCandidates74_33 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates74_33_checked : candidates edge 4 389 8087 innerTail74_34 best.bound = innerCandidates74_33 := by
  unfold candidates
  have row : edge 8087 = SearchData.row8087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_33 : scan edge (search edge 3) 4 [389] 389 innerTail74_33 best = best := by
  rw [innerTail74_33, scan, if_neg (by decide), innerCandidates74_33_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_34
    (if 3 ≤ innerCandidates74_33.length then search edge 3 [389, 8087] innerCandidates74_33 8476 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_34

def innerTail74_32 : List Nat := 7937 :: innerTail74_33
def innerCandidates74_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates74_32_checked : candidates edge 4 389 7937 innerTail74_33 best.bound = innerCandidates74_32 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_32 : scan edge (search edge 3) 4 [389] 389 innerTail74_32 best = best := by
  rw [innerTail74_32, scan, if_neg (by decide), innerCandidates74_32_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_33
    (if 3 ≤ innerCandidates74_32.length then search edge 3 [389, 7937] innerCandidates74_32 8326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_33

def innerTail74_31 : List Nat := 7853 :: innerTail74_32
def innerCandidates74_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates74_31_checked : candidates edge 4 389 7853 innerTail74_32 best.bound = innerCandidates74_31 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_31 : scan edge (search edge 3) 4 [389] 389 innerTail74_31 best = best := by
  rw [innerTail74_31, scan, if_neg (by decide), innerCandidates74_31_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_32
    (if 3 ≤ innerCandidates74_31.length then search edge 3 [389, 7853] innerCandidates74_31 8242 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_32

def innerTail74_30 : List Nat := 7727 :: innerTail74_31
def innerCandidates74_30 : List Nat := [7853]
set_option maxRecDepth 4096 in
theorem innerCandidates74_30_checked : candidates edge 4 389 7727 innerTail74_31 best.bound = innerCandidates74_30 := by
  unfold candidates
  have row : edge 7727 = SearchData.row7727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_30 : scan edge (search edge 3) 4 [389] 389 innerTail74_30 best = best := by
  rw [innerTail74_30, scan, if_neg (by decide), innerCandidates74_30_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_31
    (if 3 ≤ innerCandidates74_30.length then search edge 3 [389, 7727] innerCandidates74_30 8116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_31

def innerTail74_29 : List Nat := 7703 :: innerTail74_30
def innerCandidates74_29 : List Nat := [7853]
set_option maxRecDepth 4096 in
theorem innerCandidates74_29_checked : candidates edge 4 389 7703 innerTail74_30 best.bound = innerCandidates74_29 := by
  unfold candidates
  have row : edge 7703 = SearchData.row7703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_29 : scan edge (search edge 3) 4 [389] 389 innerTail74_29 best = best := by
  rw [innerTail74_29, scan, if_neg (by decide), innerCandidates74_29_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_30
    (if 3 ≤ innerCandidates74_29.length then search edge 3 [389, 7703] innerCandidates74_29 8092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_30

def innerTail74_28 : List Nat := 7499 :: innerTail74_29
def innerCandidates74_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates74_28_checked : candidates edge 4 389 7499 innerTail74_29 best.bound = innerCandidates74_28 := by
  unfold candidates
  have row : edge 7499 = SearchData.row7499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_28 : scan edge (search edge 3) 4 [389] 389 innerTail74_28 best = best := by
  rw [innerTail74_28, scan, if_neg (by decide), innerCandidates74_28_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_29
    (if 3 ≤ innerCandidates74_28.length then search edge 3 [389, 7499] innerCandidates74_28 7888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_29

def innerTail74_27 : List Nat := 6947 :: innerTail74_28
def innerCandidates74_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates74_27_checked : candidates edge 4 389 6947 innerTail74_28 best.bound = innerCandidates74_27 := by
  unfold candidates
  have row : edge 6947 = SearchData.row6947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_27 : scan edge (search edge 3) 4 [389] 389 innerTail74_27 best = best := by
  rw [innerTail74_27, scan, if_neg (by decide), innerCandidates74_27_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_28
    (if 3 ≤ innerCandidates74_27.length then search edge 3 [389, 6947] innerCandidates74_27 7336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_28

def innerTail74_26 : List Nat := 6257 :: innerTail74_27
def innerCandidates74_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates74_26_checked : candidates edge 4 389 6257 innerTail74_27 best.bound = innerCandidates74_26 := by
  unfold candidates
  have row : edge 6257 = SearchData.row6257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_26 : scan edge (search edge 3) 4 [389] 389 innerTail74_26 best = best := by
  rw [innerTail74_26, scan, if_neg (by decide), innerCandidates74_26_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_27
    (if 3 ≤ innerCandidates74_26.length then search edge 3 [389, 6257] innerCandidates74_26 6646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_27

def innerTail74_25 : List Nat := 6089 :: innerTail74_26
def innerCandidates74_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates74_25_checked : candidates edge 4 389 6089 innerTail74_26 best.bound = innerCandidates74_25 := by
  unfold candidates
  have row : edge 6089 = SearchData.row6089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_25 : scan edge (search edge 3) 4 [389] 389 innerTail74_25 best = best := by
  rw [innerTail74_25, scan, if_neg (by decide), innerCandidates74_25_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_26
    (if 3 ≤ innerCandidates74_25.length then search edge 3 [389, 6089] innerCandidates74_25 6478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_26

def innerTail74_24 : List Nat := 6029 :: innerTail74_25
def innerCandidates74_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates74_24_checked : candidates edge 4 389 6029 innerTail74_25 best.bound = innerCandidates74_24 := by
  unfold candidates
  have row : edge 6029 = SearchData.row6029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_24 : scan edge (search edge 3) 4 [389] 389 innerTail74_24 best = best := by
  rw [innerTail74_24, scan, if_neg (by decide), innerCandidates74_24_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_25
    (if 3 ≤ innerCandidates74_24.length then search edge 3 [389, 6029] innerCandidates74_24 6418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_25

def innerTail74_23 : List Nat := 5351 :: innerTail74_24
def innerCandidates74_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates74_23_checked : candidates edge 4 389 5351 innerTail74_24 best.bound = innerCandidates74_23 := by
  unfold candidates
  have row : edge 5351 = SearchData.row5351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_23 : scan edge (search edge 3) 4 [389] 389 innerTail74_23 best = best := by
  rw [innerTail74_23, scan, if_neg (by decide), innerCandidates74_23_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_24
    (if 3 ≤ innerCandidates74_23.length then search edge 3 [389, 5351] innerCandidates74_23 5740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_24

def innerTail74_22 : List Nat := 5231 :: innerTail74_23
def innerCandidates74_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates74_22_checked : candidates edge 4 389 5231 innerTail74_23 best.bound = innerCandidates74_22 := by
  unfold candidates
  have row : edge 5231 = SearchData.row5231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_22 : scan edge (search edge 3) 4 [389] 389 innerTail74_22 best = best := by
  rw [innerTail74_22, scan, if_neg (by decide), innerCandidates74_22_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_23
    (if 3 ≤ innerCandidates74_22.length then search edge 3 [389, 5231] innerCandidates74_22 5620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_23

def innerTail74_21 : List Nat := 4733 :: innerTail74_22
def innerCandidates74_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates74_21_checked : candidates edge 4 389 4733 innerTail74_22 best.bound = innerCandidates74_21 := by
  unfold candidates
  have row : edge 4733 = SearchData.row4733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_21 : scan edge (search edge 3) 4 [389] 389 innerTail74_21 best = best := by
  rw [innerTail74_21, scan, if_neg (by decide), innerCandidates74_21_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_22
    (if 3 ≤ innerCandidates74_21.length then search edge 3 [389, 4733] innerCandidates74_21 5122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_22

def innerTail74_20 : List Nat := 4703 :: innerTail74_21
def innerCandidates74_20 : List Nat := [6257]
set_option maxRecDepth 4096 in
theorem innerCandidates74_20_checked : candidates edge 4 389 4703 innerTail74_21 best.bound = innerCandidates74_20 := by
  unfold candidates
  have row : edge 4703 = SearchData.row4703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_20 : scan edge (search edge 3) 4 [389] 389 innerTail74_20 best = best := by
  rw [innerTail74_20, scan, if_neg (by decide), innerCandidates74_20_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_21
    (if 3 ≤ innerCandidates74_20.length then search edge 3 [389, 4703] innerCandidates74_20 5092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_21

def innerTail74_19 : List Nat := 4637 :: innerTail74_20
def innerCandidates74_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates74_19_checked : candidates edge 4 389 4637 innerTail74_20 best.bound = innerCandidates74_19 := by
  unfold candidates
  have row : edge 4637 = SearchData.row4637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_19 : scan edge (search edge 3) 4 [389] 389 innerTail74_19 best = best := by
  rw [innerTail74_19, scan, if_neg (by decide), innerCandidates74_19_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_20
    (if 3 ≤ innerCandidates74_19.length then search edge 3 [389, 4637] innerCandidates74_19 5026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_20

def innerTail74_18 : List Nat := 4409 :: innerTail74_19
def innerCandidates74_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates74_18_checked : candidates edge 4 389 4409 innerTail74_19 best.bound = innerCandidates74_18 := by
  unfold candidates
  have row : edge 4409 = SearchData.row4409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_18 : scan edge (search edge 3) 4 [389] 389 innerTail74_18 best = best := by
  rw [innerTail74_18, scan, if_neg (by decide), innerCandidates74_18_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_19
    (if 3 ≤ innerCandidates74_18.length then search edge 3 [389, 4409] innerCandidates74_18 4798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_19

def innerTail74_17 : List Nat := 4049 :: innerTail74_18
def innerCandidates74_17 : List Nat := [7703]
set_option maxRecDepth 4096 in
theorem innerCandidates74_17_checked : candidates edge 4 389 4049 innerTail74_18 best.bound = innerCandidates74_17 := by
  unfold candidates
  have row : edge 4049 = SearchData.row4049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_17 : scan edge (search edge 3) 4 [389] 389 innerTail74_17 best = best := by
  rw [innerTail74_17, scan, if_neg (by decide), innerCandidates74_17_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_18
    (if 3 ≤ innerCandidates74_17.length then search edge 3 [389, 4049] innerCandidates74_17 4438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_18

def innerTail74_16 : List Nat := 3719 :: innerTail74_17
def innerCandidates74_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates74_16_checked : candidates edge 4 389 3719 innerTail74_17 best.bound = innerCandidates74_16 := by
  unfold candidates
  have row : edge 3719 = SearchData.row3719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_16 : scan edge (search edge 3) 4 [389] 389 innerTail74_16 best = best := by
  rw [innerTail74_16, scan, if_neg (by decide), innerCandidates74_16_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_17
    (if 3 ≤ innerCandidates74_16.length then search edge 3 [389, 3719] innerCandidates74_16 4108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_17

def innerTail74_15 : List Nat := 3299 :: innerTail74_16
def innerCandidates74_15 : List Nat := [7499]
set_option maxRecDepth 4096 in
theorem innerCandidates74_15_checked : candidates edge 4 389 3299 innerTail74_16 best.bound = innerCandidates74_15 := by
  unfold candidates
  have row : edge 3299 = SearchData.row3299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_15 : scan edge (search edge 3) 4 [389] 389 innerTail74_15 best = best := by
  rw [innerTail74_15, scan, if_neg (by decide), innerCandidates74_15_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_16
    (if 3 ≤ innerCandidates74_15.length then search edge 3 [389, 3299] innerCandidates74_15 3688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_16

def innerTail74_14 : List Nat := 3191 :: innerTail74_15
def innerCandidates74_14 : List Nat := [7703, 7937]
set_option maxRecDepth 4096 in
theorem innerCandidates74_14_checked : candidates edge 4 389 3191 innerTail74_15 best.bound = innerCandidates74_14 := by
  unfold candidates
  have row : edge 3191 = SearchData.row3191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_14 : scan edge (search edge 3) 4 [389] 389 innerTail74_14 best = best := by
  rw [innerTail74_14, scan, if_neg (by decide), innerCandidates74_14_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_15
    (if 3 ≤ innerCandidates74_14.length then search edge 3 [389, 3191] innerCandidates74_14 3580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_15

def innerTail74_13 : List Nat := 3137 :: innerTail74_14
def innerCandidates74_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates74_13_checked : candidates edge 4 389 3137 innerTail74_14 best.bound = innerCandidates74_13 := by
  unfold candidates
  have row : edge 3137 = SearchData.row3137.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_13 : scan edge (search edge 3) 4 [389] 389 innerTail74_13 best = best := by
  rw [innerTail74_13, scan, if_neg (by decide), innerCandidates74_13_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_14
    (if 3 ≤ innerCandidates74_13.length then search edge 3 [389, 3137] innerCandidates74_13 3526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_14

def innerTail74_12 : List Nat := 2969 :: innerTail74_13
def innerCandidates74_12 : List Nat := [4703]
set_option maxRecDepth 4096 in
theorem innerCandidates74_12_checked : candidates edge 4 389 2969 innerTail74_13 best.bound = innerCandidates74_12 := by
  unfold candidates
  have row : edge 2969 = SearchData.row2969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_12 : scan edge (search edge 3) 4 [389] 389 innerTail74_12 best = best := by
  rw [innerTail74_12, scan, if_neg (by decide), innerCandidates74_12_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_13
    (if 3 ≤ innerCandidates74_12.length then search edge 3 [389, 2969] innerCandidates74_12 3358 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_13

def innerTail74_11 : List Nat := 2459 :: innerTail74_12
def innerCandidates74_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates74_11_checked : candidates edge 4 389 2459 innerTail74_12 best.bound = innerCandidates74_11 := by
  unfold candidates
  have row : edge 2459 = PrimePairCache.smallRow2459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_11 : scan edge (search edge 3) 4 [389] 389 innerTail74_11 best = best := by
  rw [innerTail74_11, scan, if_neg (by decide), innerCandidates74_11_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_12
    (if 3 ≤ innerCandidates74_11.length then search edge 3 [389, 2459] innerCandidates74_11 2848 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_12

def innerTail74_10 : List Nat := 2393 :: innerTail74_11
def innerCandidates74_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates74_10_checked : candidates edge 4 389 2393 innerTail74_11 best.bound = innerCandidates74_10 := by
  unfold candidates
  have row : edge 2393 = PrimePairCache.smallRow2393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_10 : scan edge (search edge 3) 4 [389] 389 innerTail74_10 best = best := by
  rw [innerTail74_10, scan, if_neg (by decide), innerCandidates74_10_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_11
    (if 3 ≤ innerCandidates74_10.length then search edge 3 [389, 2393] innerCandidates74_10 2782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_11

def innerTail74_9 : List Nat := 2333 :: innerTail74_10
def innerCandidates74_9 : List Nat := [3137]
set_option maxRecDepth 4096 in
theorem innerCandidates74_9_checked : candidates edge 4 389 2333 innerTail74_10 best.bound = innerCandidates74_9 := by
  unfold candidates
  have row : edge 2333 = PrimePairCache.smallRow2333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_9 : scan edge (search edge 3) 4 [389] 389 innerTail74_9 best = best := by
  rw [innerTail74_9, scan, if_neg (by decide), innerCandidates74_9_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_10
    (if 3 ≤ innerCandidates74_9.length then search edge 3 [389, 2333] innerCandidates74_9 2722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_10

def innerTail74_8 : List Nat := 2309 :: innerTail74_9
def innerCandidates74_8 : List Nat := [6257, 6947]
set_option maxRecDepth 4096 in
theorem innerCandidates74_8_checked : candidates edge 4 389 2309 innerTail74_9 best.bound = innerCandidates74_8 := by
  unfold candidates
  have row : edge 2309 = PrimePairCache.smallRow2309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_8 : scan edge (search edge 3) 4 [389] 389 innerTail74_8 best = best := by
  rw [innerTail74_8, scan, if_neg (by decide), innerCandidates74_8_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_9
    (if 3 ≤ innerCandidates74_8.length then search edge 3 [389, 2309] innerCandidates74_8 2698 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_9

def innerTail74_7 : List Nat := 2087 :: innerTail74_8
def innerCandidates74_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates74_7_checked : candidates edge 4 389 2087 innerTail74_8 best.bound = innerCandidates74_7 := by
  unfold candidates
  have row : edge 2087 = PrimePairCache.smallRow2087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_7 : scan edge (search edge 3) 4 [389] 389 innerTail74_7 best = best := by
  rw [innerTail74_7, scan, if_neg (by decide), innerCandidates74_7_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_8
    (if 3 ≤ innerCandidates74_7.length then search edge 3 [389, 2087] innerCandidates74_7 2476 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_8

def innerTail74_6 : List Nat := 1913 :: innerTail74_7
def innerCandidates74_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates74_6_checked : candidates edge 4 389 1913 innerTail74_7 best.bound = innerCandidates74_6 := by
  unfold candidates
  have row : edge 1913 = PrimePairCache.smallRow1913.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_6 : scan edge (search edge 3) 4 [389] 389 innerTail74_6 best = best := by
  rw [innerTail74_6, scan, if_neg (by decide), innerCandidates74_6_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_7
    (if 3 ≤ innerCandidates74_6.length then search edge 3 [389, 1913] innerCandidates74_6 2302 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_7

def innerTail74_5 : List Nat := 1493 :: innerTail74_6
def innerCandidates74_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates74_5_checked : candidates edge 4 389 1493 innerTail74_6 best.bound = innerCandidates74_5 := by
  unfold candidates
  have row : edge 1493 = PrimePairCache.smallRow1493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_5 : scan edge (search edge 3) 4 [389] 389 innerTail74_5 best = best := by
  rw [innerTail74_5, scan, if_neg (by decide), innerCandidates74_5_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_6
    (if 3 ≤ innerCandidates74_5.length then search edge 3 [389, 1493] innerCandidates74_5 1882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_6

def innerTail74_4 : List Nat := 1259 :: innerTail74_5
def innerCandidates74_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates74_4_checked : candidates edge 4 389 1259 innerTail74_5 best.bound = innerCandidates74_4 := by
  unfold candidates
  have row : edge 1259 = PrimePairCache.smallRow1259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_4 : scan edge (search edge 3) 4 [389] 389 innerTail74_4 best = best := by
  rw [innerTail74_4, scan, if_neg (by decide), innerCandidates74_4_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_5
    (if 3 ≤ innerCandidates74_4.length then search edge 3 [389, 1259] innerCandidates74_4 1648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_5

def innerTail74_3 : List Nat := 971 :: innerTail74_4
def innerCandidates74_3 : List Nat := [8123]
set_option maxRecDepth 4096 in
theorem innerCandidates74_3_checked : candidates edge 4 389 971 innerTail74_4 best.bound = innerCandidates74_3 := by
  unfold candidates
  have row : edge 971 = PrimePairCache.smallRow971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_3 : scan edge (search edge 3) 4 [389] 389 innerTail74_3 best = best := by
  rw [innerTail74_3, scan, if_neg (by decide), innerCandidates74_3_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_4
    (if 3 ≤ innerCandidates74_3.length then search edge 3 [389, 971] innerCandidates74_3 1360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_4

def innerTail74_2 : List Nat := 947 :: innerTail74_3
def innerCandidates74_2 : List Nat := [4637, 6947]
set_option maxRecDepth 4096 in
theorem innerCandidates74_2_checked : candidates edge 4 389 947 innerTail74_3 best.bound = innerCandidates74_2 := by
  unfold candidates
  have row : edge 947 = PrimePairCache.smallRow947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_2 : scan edge (search edge 3) 4 [389] 389 innerTail74_2 best = best := by
  rw [innerTail74_2, scan, if_neg (by decide), innerCandidates74_2_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_3
    (if 3 ≤ innerCandidates74_2.length then search edge 3 [389, 947] innerCandidates74_2 1336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_3

def innerTail74_1 : List Nat := 797 :: innerTail74_2
def innerCandidates74_1 : List Nat := [2969, 3299]
set_option maxRecDepth 4096 in
theorem innerCandidates74_1_checked : candidates edge 4 389 797 innerTail74_2 best.bound = innerCandidates74_1 := by
  unfold candidates
  have row : edge 797 = PrimePairCache.smallRow797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_1 : scan edge (search edge 3) 4 [389] 389 innerTail74_1 best = best := by
  rw [innerTail74_1, scan, if_neg (by decide), innerCandidates74_1_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_2
    (if 3 ≤ innerCandidates74_1.length then search edge 3 [389, 797] innerCandidates74_1 1186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_2

def innerTail74_0 : List Nat := 761 :: innerTail74_1
def innerCandidates74_0 : List Nat := [5351]
set_option maxRecDepth 4096 in
theorem innerCandidates74_0_checked : candidates edge 4 389 761 innerTail74_1 best.bound = innerCandidates74_0 := by
  unfold candidates
  have row : edge 761 = PrimePairCache.smallRow761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan74_0 : scan edge (search edge 3) 4 [389] 389 innerTail74_0 best = best := by
  rw [innerTail74_0, scan, if_neg (by decide), innerCandidates74_0_checked]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_1
    (if 3 ≤ innerCandidates74_0.length then search edge 3 [389, 761] innerCandidates74_0 1150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan74_1

theorem rootBranch74 : search edge 4 [389] rootCandidates74 389 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [389] 389 innerTail74_0 best = best
  exact innerScan74_0

def rootCandidates73 : List Nat := [419, 797, 821, 941, 1427, 1493, 1721, 1733, 1871, 1889, 2267, 2687, 2729, 3257, 3359, 3527, 3989, 4091, 4139, 4547, 5303, 6317, 6359, 7331, 7703, 7853, 7877, 8117, 8123]
set_option maxRecDepth 4096 in
theorem rootCandidates73_checked : candidates edge 5 0 383 tail74 best.bound = rootCandidates73 := by
  unfold candidates
  have row : edge 383 = PrimePairCache.smallRow383.contains := by funext q; rfl
  rw [row]
  decide

def innerTail73_29 : List Nat := []
theorem innerScan73_29 : scan edge (search edge 3) 4 [383] 383 innerTail73_29 best = best := rfl

def innerTail73_28 : List Nat := 8123 :: innerTail73_29
def innerCandidates73_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates73_28_checked : candidates edge 4 383 8123 innerTail73_29 best.bound = innerCandidates73_28 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_28 : scan edge (search edge 3) 4 [383] 383 innerTail73_28 best = best := by
  rw [innerTail73_28, scan, if_neg (by decide), innerCandidates73_28_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_29
    (if 3 ≤ innerCandidates73_28.length then search edge 3 [383, 8123] innerCandidates73_28 8506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_29

def innerTail73_27 : List Nat := 8117 :: innerTail73_28
def innerCandidates73_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates73_27_checked : candidates edge 4 383 8117 innerTail73_28 best.bound = innerCandidates73_27 := by
  unfold candidates
  have row : edge 8117 = SearchData.row8117.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_27 : scan edge (search edge 3) 4 [383] 383 innerTail73_27 best = best := by
  rw [innerTail73_27, scan, if_neg (by decide), innerCandidates73_27_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_28
    (if 3 ≤ innerCandidates73_27.length then search edge 3 [383, 8117] innerCandidates73_27 8500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_28

def innerTail73_26 : List Nat := 7877 :: innerTail73_27
def innerCandidates73_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates73_26_checked : candidates edge 4 383 7877 innerTail73_27 best.bound = innerCandidates73_26 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_26 : scan edge (search edge 3) 4 [383] 383 innerTail73_26 best = best := by
  rw [innerTail73_26, scan, if_neg (by decide), innerCandidates73_26_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_27
    (if 3 ≤ innerCandidates73_26.length then search edge 3 [383, 7877] innerCandidates73_26 8260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_27

def innerTail73_25 : List Nat := 7853 :: innerTail73_26
def innerCandidates73_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates73_25_checked : candidates edge 4 383 7853 innerTail73_26 best.bound = innerCandidates73_25 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_25 : scan edge (search edge 3) 4 [383] 383 innerTail73_25 best = best := by
  rw [innerTail73_25, scan, if_neg (by decide), innerCandidates73_25_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_26
    (if 3 ≤ innerCandidates73_25.length then search edge 3 [383, 7853] innerCandidates73_25 8236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_26

def innerTail73_24 : List Nat := 7703 :: innerTail73_25
def innerCandidates73_24 : List Nat := [7853]
set_option maxRecDepth 4096 in
theorem innerCandidates73_24_checked : candidates edge 4 383 7703 innerTail73_25 best.bound = innerCandidates73_24 := by
  unfold candidates
  have row : edge 7703 = SearchData.row7703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_24 : scan edge (search edge 3) 4 [383] 383 innerTail73_24 best = best := by
  rw [innerTail73_24, scan, if_neg (by decide), innerCandidates73_24_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_25
    (if 3 ≤ innerCandidates73_24.length then search edge 3 [383, 7703] innerCandidates73_24 8086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_25

def innerTail73_23 : List Nat := 7331 :: innerTail73_24
def innerCandidates73_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates73_23_checked : candidates edge 4 383 7331 innerTail73_24 best.bound = innerCandidates73_23 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_23 : scan edge (search edge 3) 4 [383] 383 innerTail73_23 best = best := by
  rw [innerTail73_23, scan, if_neg (by decide), innerCandidates73_23_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_24
    (if 3 ≤ innerCandidates73_23.length then search edge 3 [383, 7331] innerCandidates73_23 7714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_24

def innerTail73_22 : List Nat := 6359 :: innerTail73_23
def innerCandidates73_22 : List Nat := [8123]
set_option maxRecDepth 4096 in
theorem innerCandidates73_22_checked : candidates edge 4 383 6359 innerTail73_23 best.bound = innerCandidates73_22 := by
  unfold candidates
  have row : edge 6359 = SearchData.row6359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_22 : scan edge (search edge 3) 4 [383] 383 innerTail73_22 best = best := by
  rw [innerTail73_22, scan, if_neg (by decide), innerCandidates73_22_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_23
    (if 3 ≤ innerCandidates73_22.length then search edge 3 [383, 6359] innerCandidates73_22 6742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_23

def innerTail73_21 : List Nat := 6317 :: innerTail73_22
def innerCandidates73_21 : List Nat := [7853]
set_option maxRecDepth 4096 in
theorem innerCandidates73_21_checked : candidates edge 4 383 6317 innerTail73_22 best.bound = innerCandidates73_21 := by
  unfold candidates
  have row : edge 6317 = SearchData.row6317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_21 : scan edge (search edge 3) 4 [383] 383 innerTail73_21 best = best := by
  rw [innerTail73_21, scan, if_neg (by decide), innerCandidates73_21_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_22
    (if 3 ≤ innerCandidates73_21.length then search edge 3 [383, 6317] innerCandidates73_21 6700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_22

def innerTail73_20 : List Nat := 5303 :: innerTail73_21
def innerCandidates73_20 : List Nat := [8123]
set_option maxRecDepth 4096 in
theorem innerCandidates73_20_checked : candidates edge 4 383 5303 innerTail73_21 best.bound = innerCandidates73_20 := by
  unfold candidates
  have row : edge 5303 = SearchData.row5303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_20 : scan edge (search edge 3) 4 [383] 383 innerTail73_20 best = best := by
  rw [innerTail73_20, scan, if_neg (by decide), innerCandidates73_20_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_21
    (if 3 ≤ innerCandidates73_20.length then search edge 3 [383, 5303] innerCandidates73_20 5686 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_21

def innerTail73_19 : List Nat := 4547 :: innerTail73_20
def innerCandidates73_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates73_19_checked : candidates edge 4 383 4547 innerTail73_20 best.bound = innerCandidates73_19 := by
  unfold candidates
  have row : edge 4547 = SearchData.row4547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_19 : scan edge (search edge 3) 4 [383] 383 innerTail73_19 best = best := by
  rw [innerTail73_19, scan, if_neg (by decide), innerCandidates73_19_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_20
    (if 3 ≤ innerCandidates73_19.length then search edge 3 [383, 4547] innerCandidates73_19 4930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_20

def innerTail73_18 : List Nat := 4139 :: innerTail73_19
def innerCandidates73_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates73_18_checked : candidates edge 4 383 4139 innerTail73_19 best.bound = innerCandidates73_18 := by
  unfold candidates
  have row : edge 4139 = SearchData.row4139.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_18 : scan edge (search edge 3) 4 [383] 383 innerTail73_18 best = best := by
  rw [innerTail73_18, scan, if_neg (by decide), innerCandidates73_18_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_19
    (if 3 ≤ innerCandidates73_18.length then search edge 3 [383, 4139] innerCandidates73_18 4522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_19

def innerTail73_17 : List Nat := 4091 :: innerTail73_18
def innerCandidates73_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates73_17_checked : candidates edge 4 383 4091 innerTail73_18 best.bound = innerCandidates73_17 := by
  unfold candidates
  have row : edge 4091 = SearchData.row4091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_17 : scan edge (search edge 3) 4 [383] 383 innerTail73_17 best = best := by
  rw [innerTail73_17, scan, if_neg (by decide), innerCandidates73_17_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_18
    (if 3 ≤ innerCandidates73_17.length then search edge 3 [383, 4091] innerCandidates73_17 4474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_18

def innerTail73_16 : List Nat := 3989 :: innerTail73_17
def innerCandidates73_16 : List Nat := [7703]
set_option maxRecDepth 4096 in
theorem innerCandidates73_16_checked : candidates edge 4 383 3989 innerTail73_17 best.bound = innerCandidates73_16 := by
  unfold candidates
  have row : edge 3989 = SearchData.row3989.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_16 : scan edge (search edge 3) 4 [383] 383 innerTail73_16 best = best := by
  rw [innerTail73_16, scan, if_neg (by decide), innerCandidates73_16_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_17
    (if 3 ≤ innerCandidates73_16.length then search edge 3 [383, 3989] innerCandidates73_16 4372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_17

def innerTail73_15 : List Nat := 3527 :: innerTail73_16
def innerCandidates73_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates73_15_checked : candidates edge 4 383 3527 innerTail73_16 best.bound = innerCandidates73_15 := by
  unfold candidates
  have row : edge 3527 = SearchData.row3527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_15 : scan edge (search edge 3) 4 [383] 383 innerTail73_15 best = best := by
  rw [innerTail73_15, scan, if_neg (by decide), innerCandidates73_15_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_16
    (if 3 ≤ innerCandidates73_15.length then search edge 3 [383, 3527] innerCandidates73_15 3910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_16

def innerTail73_14 : List Nat := 3359 :: innerTail73_15
def innerCandidates73_14 : List Nat := [7853]
set_option maxRecDepth 4096 in
theorem innerCandidates73_14_checked : candidates edge 4 383 3359 innerTail73_15 best.bound = innerCandidates73_14 := by
  unfold candidates
  have row : edge 3359 = SearchData.row3359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_14 : scan edge (search edge 3) 4 [383] 383 innerTail73_14 best = best := by
  rw [innerTail73_14, scan, if_neg (by decide), innerCandidates73_14_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_15
    (if 3 ≤ innerCandidates73_14.length then search edge 3 [383, 3359] innerCandidates73_14 3742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_15

def innerTail73_13 : List Nat := 3257 :: innerTail73_14
def innerCandidates73_13 : List Nat := [4547]
set_option maxRecDepth 4096 in
theorem innerCandidates73_13_checked : candidates edge 4 383 3257 innerTail73_14 best.bound = innerCandidates73_13 := by
  unfold candidates
  have row : edge 3257 = SearchData.row3257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_13 : scan edge (search edge 3) 4 [383] 383 innerTail73_13 best = best := by
  rw [innerTail73_13, scan, if_neg (by decide), innerCandidates73_13_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_14
    (if 3 ≤ innerCandidates73_13.length then search edge 3 [383, 3257] innerCandidates73_13 3640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_14

def innerTail73_12 : List Nat := 2729 :: innerTail73_13
def innerCandidates73_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates73_12_checked : candidates edge 4 383 2729 innerTail73_13 best.bound = innerCandidates73_12 := by
  unfold candidates
  have row : edge 2729 = SearchData.row2729.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_12 : scan edge (search edge 3) 4 [383] 383 innerTail73_12 best = best := by
  rw [innerTail73_12, scan, if_neg (by decide), innerCandidates73_12_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_13
    (if 3 ≤ innerCandidates73_12.length then search edge 3 [383, 2729] innerCandidates73_12 3112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_13

def innerTail73_11 : List Nat := 2687 :: innerTail73_12
def innerCandidates73_11 : List Nat := [7703]
set_option maxRecDepth 4096 in
theorem innerCandidates73_11_checked : candidates edge 4 383 2687 innerTail73_12 best.bound = innerCandidates73_11 := by
  unfold candidates
  have row : edge 2687 = SearchData.row2687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_11 : scan edge (search edge 3) 4 [383] 383 innerTail73_11 best = best := by
  rw [innerTail73_11, scan, if_neg (by decide), innerCandidates73_11_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_12
    (if 3 ≤ innerCandidates73_11.length then search edge 3 [383, 2687] innerCandidates73_11 3070 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_12

def innerTail73_10 : List Nat := 2267 :: innerTail73_11
def innerCandidates73_10 : List Nat := [2729, 3527]
set_option maxRecDepth 4096 in
theorem innerCandidates73_10_checked : candidates edge 4 383 2267 innerTail73_11 best.bound = innerCandidates73_10 := by
  unfold candidates
  have row : edge 2267 = PrimePairCache.smallRow2267.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_10 : scan edge (search edge 3) 4 [383] 383 innerTail73_10 best = best := by
  rw [innerTail73_10, scan, if_neg (by decide), innerCandidates73_10_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_11
    (if 3 ≤ innerCandidates73_10.length then search edge 3 [383, 2267] innerCandidates73_10 2650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_11

def innerTail73_9 : List Nat := 1889 :: innerTail73_10
def innerCandidates73_9 : List Nat := [8123]
set_option maxRecDepth 4096 in
theorem innerCandidates73_9_checked : candidates edge 4 383 1889 innerTail73_10 best.bound = innerCandidates73_9 := by
  unfold candidates
  have row : edge 1889 = PrimePairCache.smallRow1889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_9 : scan edge (search edge 3) 4 [383] 383 innerTail73_9 best = best := by
  rw [innerTail73_9, scan, if_neg (by decide), innerCandidates73_9_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_10
    (if 3 ≤ innerCandidates73_9.length then search edge 3 [383, 1889] innerCandidates73_9 2272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_10

def innerTail73_8 : List Nat := 1871 :: innerTail73_9
def innerCandidates73_8 : List Nat := [4139]
set_option maxRecDepth 4096 in
theorem innerCandidates73_8_checked : candidates edge 4 383 1871 innerTail73_9 best.bound = innerCandidates73_8 := by
  unfold candidates
  have row : edge 1871 = PrimePairCache.smallRow1871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_8 : scan edge (search edge 3) 4 [383] 383 innerTail73_8 best = best := by
  rw [innerTail73_8, scan, if_neg (by decide), innerCandidates73_8_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_9
    (if 3 ≤ innerCandidates73_8.length then search edge 3 [383, 1871] innerCandidates73_8 2254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_9

def innerTail73_7 : List Nat := 1733 :: innerTail73_8
def innerCandidates73_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates73_7_checked : candidates edge 4 383 1733 innerTail73_8 best.bound = innerCandidates73_7 := by
  unfold candidates
  have row : edge 1733 = PrimePairCache.smallRow1733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_7 : scan edge (search edge 3) 4 [383] 383 innerTail73_7 best = best := by
  rw [innerTail73_7, scan, if_neg (by decide), innerCandidates73_7_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_8
    (if 3 ≤ innerCandidates73_7.length then search edge 3 [383, 1733] innerCandidates73_7 2116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_8

def innerTail73_6 : List Nat := 1721 :: innerTail73_7
def innerCandidates73_6 : List Nat := [3359]
set_option maxRecDepth 4096 in
theorem innerCandidates73_6_checked : candidates edge 4 383 1721 innerTail73_7 best.bound = innerCandidates73_6 := by
  unfold candidates
  have row : edge 1721 = PrimePairCache.smallRow1721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_6 : scan edge (search edge 3) 4 [383] 383 innerTail73_6 best = best := by
  rw [innerTail73_6, scan, if_neg (by decide), innerCandidates73_6_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_7
    (if 3 ≤ innerCandidates73_6.length then search edge 3 [383, 1721] innerCandidates73_6 2104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_7

def innerTail73_5 : List Nat := 1493 :: innerTail73_6
def innerCandidates73_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates73_5_checked : candidates edge 4 383 1493 innerTail73_6 best.bound = innerCandidates73_5 := by
  unfold candidates
  have row : edge 1493 = PrimePairCache.smallRow1493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_5 : scan edge (search edge 3) 4 [383] 383 innerTail73_5 best = best := by
  rw [innerTail73_5, scan, if_neg (by decide), innerCandidates73_5_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_6
    (if 3 ≤ innerCandidates73_5.length then search edge 3 [383, 1493] innerCandidates73_5 1876 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_6

def innerTail73_4 : List Nat := 1427 :: innerTail73_5
def innerCandidates73_4 : List Nat := [3359, 3989]
set_option maxRecDepth 4096 in
theorem innerCandidates73_4_checked : candidates edge 4 383 1427 innerTail73_5 best.bound = innerCandidates73_4 := by
  unfold candidates
  have row : edge 1427 = PrimePairCache.smallRow1427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_4 : scan edge (search edge 3) 4 [383] 383 innerTail73_4 best = best := by
  rw [innerTail73_4, scan, if_neg (by decide), innerCandidates73_4_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_5
    (if 3 ≤ innerCandidates73_4.length then search edge 3 [383, 1427] innerCandidates73_4 1810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_5

def innerTail73_3 : List Nat := 941 :: innerTail73_4
def innerCandidates73_3 : List Nat := [3527]
set_option maxRecDepth 4096 in
theorem innerCandidates73_3_checked : candidates edge 4 383 941 innerTail73_4 best.bound = innerCandidates73_3 := by
  unfold candidates
  have row : edge 941 = PrimePairCache.smallRow941.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_3 : scan edge (search edge 3) 4 [383] 383 innerTail73_3 best = best := by
  rw [innerTail73_3, scan, if_neg (by decide), innerCandidates73_3_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_4
    (if 3 ≤ innerCandidates73_3.length then search edge 3 [383, 941] innerCandidates73_3 1324 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_4

def innerTail73_2 : List Nat := 821 :: innerTail73_3
def innerCandidates73_2 : List Nat := [6317]
set_option maxRecDepth 4096 in
theorem innerCandidates73_2_checked : candidates edge 4 383 821 innerTail73_3 best.bound = innerCandidates73_2 := by
  unfold candidates
  have row : edge 821 = PrimePairCache.smallRow821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_2 : scan edge (search edge 3) 4 [383] 383 innerTail73_2 best = best := by
  rw [innerTail73_2, scan, if_neg (by decide), innerCandidates73_2_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_3
    (if 3 ≤ innerCandidates73_2.length then search edge 3 [383, 821] innerCandidates73_2 1204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_3

def innerTail73_1 : List Nat := 797 :: innerTail73_2
def innerCandidates73_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates73_1_checked : candidates edge 4 383 797 innerTail73_2 best.bound = innerCandidates73_1 := by
  unfold candidates
  have row : edge 797 = PrimePairCache.smallRow797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_1 : scan edge (search edge 3) 4 [383] 383 innerTail73_1 best = best := by
  rw [innerTail73_1, scan, if_neg (by decide), innerCandidates73_1_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_2
    (if 3 ≤ innerCandidates73_1.length then search edge 3 [383, 797] innerCandidates73_1 1180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_2

def innerTail73_0 : List Nat := 419 :: innerTail73_1
def innerCandidates73_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates73_0_checked : candidates edge 4 383 419 innerTail73_1 best.bound = innerCandidates73_0 := by
  unfold candidates
  have row : edge 419 = PrimePairCache.smallRow419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan73_0 : scan edge (search edge 3) 4 [383] 383 innerTail73_0 best = best := by
  rw [innerTail73_0, scan, if_neg (by decide), innerCandidates73_0_checked]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_1
    (if 3 ≤ innerCandidates73_0.length then search edge 3 [383, 419] innerCandidates73_0 802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan73_1

theorem rootBranch73 : search edge 4 [383] rootCandidates73 383 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [383] 383 innerTail73_0 best = best
  exact innerScan73_0

end ProjectEulerSolutions.PrimePairUpper.Search
