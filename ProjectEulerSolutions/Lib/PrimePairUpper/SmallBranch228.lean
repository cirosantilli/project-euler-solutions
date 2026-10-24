import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates49 : List Nat := [263, 347, 461, 509, 641, 929, 947, 977, 1013, 1049, 1091, 1439, 1637, 1847, 2333, 2447, 2459, 2729, 2861, 3257, 3467, 3761, 4157, 4241, 4787, 5039, 5303, 5807, 5879, 5927, 6131, 6197, 6257, 6659, 7523, 7541, 7823, 8087, 8111, 8243]
set_option maxRecDepth 4096 in
theorem rootCandidates49_checked : candidates edge 5 0 239 tail50 best.bound = rootCandidates49 := by
  unfold candidates
  have row : edge 239 = PrimePairCache.smallRow239.contains := by funext q; rfl
  rw [row]
  decide

def innerTail49_40 : List Nat := []
theorem innerScan49_40 : scan edge (search edge 3) 4 [239] 239 innerTail49_40 best = best := rfl

def innerTail49_39 : List Nat := 8243 :: innerTail49_40
def innerCandidates49_39 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates49_39_checked : candidates edge 4 239 8243 innerTail49_40 best.bound = innerCandidates49_39 := by
  unfold candidates
  have row : edge 8243 = SearchData.row8243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_39 : scan edge (search edge 3) 4 [239] 239 innerTail49_39 best = best := by
  rw [innerTail49_39, scan, if_neg (by decide), innerCandidates49_39_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_40
    (if 3 ≤ innerCandidates49_39.length then search edge 3 [239, 8243] innerCandidates49_39 8482 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_40

def innerTail49_38 : List Nat := 8111 :: innerTail49_39
def innerCandidates49_38 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates49_38_checked : candidates edge 4 239 8111 innerTail49_39 best.bound = innerCandidates49_38 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_38 : scan edge (search edge 3) 4 [239] 239 innerTail49_38 best = best := by
  rw [innerTail49_38, scan, if_neg (by decide), innerCandidates49_38_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_39
    (if 3 ≤ innerCandidates49_38.length then search edge 3 [239, 8111] innerCandidates49_38 8350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_39

def innerTail49_37 : List Nat := 8087 :: innerTail49_38
def innerCandidates49_37 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates49_37_checked : candidates edge 4 239 8087 innerTail49_38 best.bound = innerCandidates49_37 := by
  unfold candidates
  have row : edge 8087 = SearchData.row8087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_37 : scan edge (search edge 3) 4 [239] 239 innerTail49_37 best = best := by
  rw [innerTail49_37, scan, if_neg (by decide), innerCandidates49_37_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_38
    (if 3 ≤ innerCandidates49_37.length then search edge 3 [239, 8087] innerCandidates49_37 8326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_38

def innerTail49_36 : List Nat := 7823 :: innerTail49_37
def innerCandidates49_36 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates49_36_checked : candidates edge 4 239 7823 innerTail49_37 best.bound = innerCandidates49_36 := by
  unfold candidates
  have row : edge 7823 = SearchData.row7823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_36 : scan edge (search edge 3) 4 [239] 239 innerTail49_36 best = best := by
  rw [innerTail49_36, scan, if_neg (by decide), innerCandidates49_36_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_37
    (if 3 ≤ innerCandidates49_36.length then search edge 3 [239, 7823] innerCandidates49_36 8062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_37

def innerTail49_35 : List Nat := 7541 :: innerTail49_36
def innerCandidates49_35 : List Nat := [8111]
set_option maxRecDepth 4096 in
theorem innerCandidates49_35_checked : candidates edge 4 239 7541 innerTail49_36 best.bound = innerCandidates49_35 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_35 : scan edge (search edge 3) 4 [239] 239 innerTail49_35 best = best := by
  rw [innerTail49_35, scan, if_neg (by decide), innerCandidates49_35_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_36
    (if 3 ≤ innerCandidates49_35.length then search edge 3 [239, 7541] innerCandidates49_35 7780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_36

def innerTail49_34 : List Nat := 7523 :: innerTail49_35
def innerCandidates49_34 : List Nat := [7541]
set_option maxRecDepth 4096 in
theorem innerCandidates49_34_checked : candidates edge 4 239 7523 innerTail49_35 best.bound = innerCandidates49_34 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_34 : scan edge (search edge 3) 4 [239] 239 innerTail49_34 best = best := by
  rw [innerTail49_34, scan, if_neg (by decide), innerCandidates49_34_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_35
    (if 3 ≤ innerCandidates49_34.length then search edge 3 [239, 7523] innerCandidates49_34 7762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_35

def innerTail49_33 : List Nat := 6659 :: innerTail49_34
def innerCandidates49_33 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates49_33_checked : candidates edge 4 239 6659 innerTail49_34 best.bound = innerCandidates49_33 := by
  unfold candidates
  have row : edge 6659 = SearchData.row6659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_33 : scan edge (search edge 3) 4 [239] 239 innerTail49_33 best = best := by
  rw [innerTail49_33, scan, if_neg (by decide), innerCandidates49_33_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_34
    (if 3 ≤ innerCandidates49_33.length then search edge 3 [239, 6659] innerCandidates49_33 6898 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_34

def innerTail49_32 : List Nat := 6257 :: innerTail49_33
def innerCandidates49_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates49_32_checked : candidates edge 4 239 6257 innerTail49_33 best.bound = innerCandidates49_32 := by
  unfold candidates
  have row : edge 6257 = SearchData.row6257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_32 : scan edge (search edge 3) 4 [239] 239 innerTail49_32 best = best := by
  rw [innerTail49_32, scan, if_neg (by decide), innerCandidates49_32_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_33
    (if 3 ≤ innerCandidates49_32.length then search edge 3 [239, 6257] innerCandidates49_32 6496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_33

def innerTail49_31 : List Nat := 6197 :: innerTail49_32
def innerCandidates49_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates49_31_checked : candidates edge 4 239 6197 innerTail49_32 best.bound = innerCandidates49_31 := by
  unfold candidates
  have row : edge 6197 = SearchData.row6197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_31 : scan edge (search edge 3) 4 [239] 239 innerTail49_31 best = best := by
  rw [innerTail49_31, scan, if_neg (by decide), innerCandidates49_31_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_32
    (if 3 ≤ innerCandidates49_31.length then search edge 3 [239, 6197] innerCandidates49_31 6436 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_32

def innerTail49_30 : List Nat := 6131 :: innerTail49_31
def innerCandidates49_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates49_30_checked : candidates edge 4 239 6131 innerTail49_31 best.bound = innerCandidates49_30 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_30 : scan edge (search edge 3) 4 [239] 239 innerTail49_30 best = best := by
  rw [innerTail49_30, scan, if_neg (by decide), innerCandidates49_30_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_31
    (if 3 ≤ innerCandidates49_30.length then search edge 3 [239, 6131] innerCandidates49_30 6370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_31

def innerTail49_29 : List Nat := 5927 :: innerTail49_30
def innerCandidates49_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates49_29_checked : candidates edge 4 239 5927 innerTail49_30 best.bound = innerCandidates49_29 := by
  unfold candidates
  have row : edge 5927 = SearchData.row5927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_29 : scan edge (search edge 3) 4 [239] 239 innerTail49_29 best = best := by
  rw [innerTail49_29, scan, if_neg (by decide), innerCandidates49_29_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_30
    (if 3 ≤ innerCandidates49_29.length then search edge 3 [239, 5927] innerCandidates49_29 6166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_30

def innerTail49_28 : List Nat := 5879 :: innerTail49_29
def innerCandidates49_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates49_28_checked : candidates edge 4 239 5879 innerTail49_29 best.bound = innerCandidates49_28 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_28 : scan edge (search edge 3) 4 [239] 239 innerTail49_28 best = best := by
  rw [innerTail49_28, scan, if_neg (by decide), innerCandidates49_28_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_29
    (if 3 ≤ innerCandidates49_28.length then search edge 3 [239, 5879] innerCandidates49_28 6118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_29

def innerTail49_27 : List Nat := 5807 :: innerTail49_28
def innerCandidates49_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates49_27_checked : candidates edge 4 239 5807 innerTail49_28 best.bound = innerCandidates49_27 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_27 : scan edge (search edge 3) 4 [239] 239 innerTail49_27 best = best := by
  rw [innerTail49_27, scan, if_neg (by decide), innerCandidates49_27_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_28
    (if 3 ≤ innerCandidates49_27.length then search edge 3 [239, 5807] innerCandidates49_27 6046 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_28

def innerTail49_26 : List Nat := 5303 :: innerTail49_27
def innerCandidates49_26 : List Nat := [7823, 8243]
set_option maxRecDepth 4096 in
theorem innerCandidates49_26_checked : candidates edge 4 239 5303 innerTail49_27 best.bound = innerCandidates49_26 := by
  unfold candidates
  have row : edge 5303 = SearchData.row5303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_26 : scan edge (search edge 3) 4 [239] 239 innerTail49_26 best = best := by
  rw [innerTail49_26, scan, if_neg (by decide), innerCandidates49_26_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_27
    (if 3 ≤ innerCandidates49_26.length then search edge 3 [239, 5303] innerCandidates49_26 5542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_27

def innerTail49_25 : List Nat := 5039 :: innerTail49_26
def innerCandidates49_25 : List Nat := [8111]
set_option maxRecDepth 4096 in
theorem innerCandidates49_25_checked : candidates edge 4 239 5039 innerTail49_26 best.bound = innerCandidates49_25 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_25 : scan edge (search edge 3) 4 [239] 239 innerTail49_25 best = best := by
  rw [innerTail49_25, scan, if_neg (by decide), innerCandidates49_25_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_26
    (if 3 ≤ innerCandidates49_25.length then search edge 3 [239, 5039] innerCandidates49_25 5278 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_26

def innerTail49_24 : List Nat := 4787 :: innerTail49_25
def innerCandidates49_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates49_24_checked : candidates edge 4 239 4787 innerTail49_25 best.bound = innerCandidates49_24 := by
  unfold candidates
  have row : edge 4787 = SearchData.row4787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_24 : scan edge (search edge 3) 4 [239] 239 innerTail49_24 best = best := by
  rw [innerTail49_24, scan, if_neg (by decide), innerCandidates49_24_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_25
    (if 3 ≤ innerCandidates49_24.length then search edge 3 [239, 4787] innerCandidates49_24 5026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_25

def innerTail49_23 : List Nat := 4241 :: innerTail49_24
def innerCandidates49_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates49_23_checked : candidates edge 4 239 4241 innerTail49_24 best.bound = innerCandidates49_23 := by
  unfold candidates
  have row : edge 4241 = SearchData.row4241.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_23 : scan edge (search edge 3) 4 [239] 239 innerTail49_23 best = best := by
  rw [innerTail49_23, scan, if_neg (by decide), innerCandidates49_23_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_24
    (if 3 ≤ innerCandidates49_23.length then search edge 3 [239, 4241] innerCandidates49_23 4480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_24

def innerTail49_22 : List Nat := 4157 :: innerTail49_23
def innerCandidates49_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates49_22_checked : candidates edge 4 239 4157 innerTail49_23 best.bound = innerCandidates49_22 := by
  unfold candidates
  have row : edge 4157 = SearchData.row4157.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_22 : scan edge (search edge 3) 4 [239] 239 innerTail49_22 best = best := by
  rw [innerTail49_22, scan, if_neg (by decide), innerCandidates49_22_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_23
    (if 3 ≤ innerCandidates49_22.length then search edge 3 [239, 4157] innerCandidates49_22 4396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_23

def innerTail49_21 : List Nat := 3761 :: innerTail49_22
def innerCandidates49_21 : List Nat := [5807, 7523]
set_option maxRecDepth 4096 in
theorem innerCandidates49_21_checked : candidates edge 4 239 3761 innerTail49_22 best.bound = innerCandidates49_21 := by
  unfold candidates
  have row : edge 3761 = SearchData.row3761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_21 : scan edge (search edge 3) 4 [239] 239 innerTail49_21 best = best := by
  rw [innerTail49_21, scan, if_neg (by decide), innerCandidates49_21_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_22
    (if 3 ≤ innerCandidates49_21.length then search edge 3 [239, 3761] innerCandidates49_21 4000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_22

def innerTail49_20 : List Nat := 3467 :: innerTail49_21
def innerCandidates49_20 : List Nat := [5807]
set_option maxRecDepth 4096 in
theorem innerCandidates49_20_checked : candidates edge 4 239 3467 innerTail49_21 best.bound = innerCandidates49_20 := by
  unfold candidates
  have row : edge 3467 = SearchData.row3467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_20 : scan edge (search edge 3) 4 [239] 239 innerTail49_20 best = best := by
  rw [innerTail49_20, scan, if_neg (by decide), innerCandidates49_20_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_21
    (if 3 ≤ innerCandidates49_20.length then search edge 3 [239, 3467] innerCandidates49_20 3706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_21

def innerTail49_19 : List Nat := 3257 :: innerTail49_20
def innerCandidates49_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates49_19_checked : candidates edge 4 239 3257 innerTail49_20 best.bound = innerCandidates49_19 := by
  unfold candidates
  have row : edge 3257 = SearchData.row3257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_19 : scan edge (search edge 3) 4 [239] 239 innerTail49_19 best = best := by
  rw [innerTail49_19, scan, if_neg (by decide), innerCandidates49_19_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_20
    (if 3 ≤ innerCandidates49_19.length then search edge 3 [239, 3257] innerCandidates49_19 3496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_20

def innerTail49_18 : List Nat := 2861 :: innerTail49_19
def innerCandidates49_18 : List Nat := [3257]
set_option maxRecDepth 4096 in
theorem innerCandidates49_18_checked : candidates edge 4 239 2861 innerTail49_19 best.bound = innerCandidates49_18 := by
  unfold candidates
  have row : edge 2861 = SearchData.row2861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_18 : scan edge (search edge 3) 4 [239] 239 innerTail49_18 best = best := by
  rw [innerTail49_18, scan, if_neg (by decide), innerCandidates49_18_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_19
    (if 3 ≤ innerCandidates49_18.length then search edge 3 [239, 2861] innerCandidates49_18 3100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_19

def innerTail49_17 : List Nat := 2729 :: innerTail49_18
def innerCandidates49_17 : List Nat := [6131]
set_option maxRecDepth 4096 in
theorem innerCandidates49_17_checked : candidates edge 4 239 2729 innerTail49_18 best.bound = innerCandidates49_17 := by
  unfold candidates
  have row : edge 2729 = SearchData.row2729.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_17 : scan edge (search edge 3) 4 [239] 239 innerTail49_17 best = best := by
  rw [innerTail49_17, scan, if_neg (by decide), innerCandidates49_17_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_18
    (if 3 ≤ innerCandidates49_17.length then search edge 3 [239, 2729] innerCandidates49_17 2968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_18

def innerTail49_16 : List Nat := 2459 :: innerTail49_17
def innerCandidates49_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates49_16_checked : candidates edge 4 239 2459 innerTail49_17 best.bound = innerCandidates49_16 := by
  unfold candidates
  have row : edge 2459 = PrimePairCache.smallRow2459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_16 : scan edge (search edge 3) 4 [239] 239 innerTail49_16 best = best := by
  rw [innerTail49_16, scan, if_neg (by decide), innerCandidates49_16_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_17
    (if 3 ≤ innerCandidates49_16.length then search edge 3 [239, 2459] innerCandidates49_16 2698 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_17

def innerTail49_15 : List Nat := 2447 :: innerTail49_16
def innerCandidates49_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates49_15_checked : candidates edge 4 239 2447 innerTail49_16 best.bound = innerCandidates49_15 := by
  unfold candidates
  have row : edge 2447 = PrimePairCache.smallRow2447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_15 : scan edge (search edge 3) 4 [239] 239 innerTail49_15 best = best := by
  rw [innerTail49_15, scan, if_neg (by decide), innerCandidates49_15_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_16
    (if 3 ≤ innerCandidates49_15.length then search edge 3 [239, 2447] innerCandidates49_15 2686 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_16

def innerTail49_14 : List Nat := 2333 :: innerTail49_15
def innerCandidates49_14 : List Nat := [5927]
set_option maxRecDepth 4096 in
theorem innerCandidates49_14_checked : candidates edge 4 239 2333 innerTail49_15 best.bound = innerCandidates49_14 := by
  unfold candidates
  have row : edge 2333 = PrimePairCache.smallRow2333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_14 : scan edge (search edge 3) 4 [239] 239 innerTail49_14 best = best := by
  rw [innerTail49_14, scan, if_neg (by decide), innerCandidates49_14_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_15
    (if 3 ≤ innerCandidates49_14.length then search edge 3 [239, 2333] innerCandidates49_14 2572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_15

def innerTail49_13 : List Nat := 1847 :: innerTail49_14
def innerCandidates49_13 : List Nat := [5927, 7523]
set_option maxRecDepth 4096 in
theorem innerCandidates49_13_checked : candidates edge 4 239 1847 innerTail49_14 best.bound = innerCandidates49_13 := by
  unfold candidates
  have row : edge 1847 = PrimePairCache.smallRow1847.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_13 : scan edge (search edge 3) 4 [239] 239 innerTail49_13 best = best := by
  rw [innerTail49_13, scan, if_neg (by decide), innerCandidates49_13_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_14
    (if 3 ≤ innerCandidates49_13.length then search edge 3 [239, 1847] innerCandidates49_13 2086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_14

def innerTail49_12 : List Nat := 1637 :: innerTail49_13
def innerCandidates49_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates49_12_checked : candidates edge 4 239 1637 innerTail49_13 best.bound = innerCandidates49_12 := by
  unfold candidates
  have row : edge 1637 = PrimePairCache.smallRow1637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_12 : scan edge (search edge 3) 4 [239] 239 innerTail49_12 best = best := by
  rw [innerTail49_12, scan, if_neg (by decide), innerCandidates49_12_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_13
    (if 3 ≤ innerCandidates49_12.length then search edge 3 [239, 1637] innerCandidates49_12 1876 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_13

def innerTail49_11 : List Nat := 1439 :: innerTail49_12
def innerCandidates49_11 : List Nat := [6659]
set_option maxRecDepth 4096 in
theorem innerCandidates49_11_checked : candidates edge 4 239 1439 innerTail49_12 best.bound = innerCandidates49_11 := by
  unfold candidates
  have row : edge 1439 = PrimePairCache.smallRow1439.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_11 : scan edge (search edge 3) 4 [239] 239 innerTail49_11 best = best := by
  rw [innerTail49_11, scan, if_neg (by decide), innerCandidates49_11_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_12
    (if 3 ≤ innerCandidates49_11.length then search edge 3 [239, 1439] innerCandidates49_11 1678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_12

def innerTail49_10 : List Nat := 1091 :: innerTail49_11
def innerCandidates49_10 : List Nat := [1847, 8111]
set_option maxRecDepth 4096 in
theorem innerCandidates49_10_checked : candidates edge 4 239 1091 innerTail49_11 best.bound = innerCandidates49_10 := by
  unfold candidates
  have row : edge 1091 = PrimePairCache.smallRow1091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_10 : scan edge (search edge 3) 4 [239] 239 innerTail49_10 best = best := by
  rw [innerTail49_10, scan, if_neg (by decide), innerCandidates49_10_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_11
    (if 3 ≤ innerCandidates49_10.length then search edge 3 [239, 1091] innerCandidates49_10 1330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_11

def innerTail49_9 : List Nat := 1049 :: innerTail49_10
def innerCandidates49_9 : List Nat := [1847, 4787, 7523]
set_option maxRecDepth 4096 in
theorem innerCandidates49_9_checked : candidates edge 4 239 1049 innerTail49_10 best.bound = innerCandidates49_9 := by
  unfold candidates
  have row : edge 1049 = PrimePairCache.smallRow1049.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild49_9 : search edge 3 [239, 1049] innerCandidates49_9 1288 best = best := by decide

theorem innerScan49_9 : scan edge (search edge 3) 4 [239] 239 innerTail49_9 best = best := by
  rw [innerTail49_9, scan, if_neg (by decide), innerCandidates49_9_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_10
    (if 3 ≤ innerCandidates49_9.length then search edge 3 [239, 1049] innerCandidates49_9 1288 best else best) = best
  rw [if_pos (by decide), innerChild49_9]
  exact innerScan49_10

def innerTail49_8 : List Nat := 1013 :: innerTail49_9
def innerCandidates49_8 : List Nat := [1091, 5039]
set_option maxRecDepth 4096 in
theorem innerCandidates49_8_checked : candidates edge 4 239 1013 innerTail49_9 best.bound = innerCandidates49_8 := by
  unfold candidates
  have row : edge 1013 = PrimePairCache.smallRow1013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_8 : scan edge (search edge 3) 4 [239] 239 innerTail49_8 best = best := by
  rw [innerTail49_8, scan, if_neg (by decide), innerCandidates49_8_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_9
    (if 3 ≤ innerCandidates49_8.length then search edge 3 [239, 1013] innerCandidates49_8 1252 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_9

def innerTail49_7 : List Nat := 977 :: innerTail49_8
def innerCandidates49_7 : List Nat := [5039]
set_option maxRecDepth 4096 in
theorem innerCandidates49_7_checked : candidates edge 4 239 977 innerTail49_8 best.bound = innerCandidates49_7 := by
  unfold candidates
  have row : edge 977 = PrimePairCache.smallRow977.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_7 : scan edge (search edge 3) 4 [239] 239 innerTail49_7 best = best := by
  rw [innerTail49_7, scan, if_neg (by decide), innerCandidates49_7_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_8
    (if 3 ≤ innerCandidates49_7.length then search edge 3 [239, 977] innerCandidates49_7 1216 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_8

def innerTail49_6 : List Nat := 947 :: innerTail49_7
def innerCandidates49_6 : List Nat := [1439, 8111]
set_option maxRecDepth 4096 in
theorem innerCandidates49_6_checked : candidates edge 4 239 947 innerTail49_7 best.bound = innerCandidates49_6 := by
  unfold candidates
  have row : edge 947 = PrimePairCache.smallRow947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_6 : scan edge (search edge 3) 4 [239] 239 innerTail49_6 best = best := by
  rw [innerTail49_6, scan, if_neg (by decide), innerCandidates49_6_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_7
    (if 3 ≤ innerCandidates49_6.length then search edge 3 [239, 947] innerCandidates49_6 1186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_7

def innerTail49_5 : List Nat := 929 :: innerTail49_6
def innerCandidates49_5 : List Nat := [3467, 3761]
set_option maxRecDepth 4096 in
theorem innerCandidates49_5_checked : candidates edge 4 239 929 innerTail49_6 best.bound = innerCandidates49_5 := by
  unfold candidates
  have row : edge 929 = PrimePairCache.smallRow929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_5 : scan edge (search edge 3) 4 [239] 239 innerTail49_5 best = best := by
  rw [innerTail49_5, scan, if_neg (by decide), innerCandidates49_5_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_6
    (if 3 ≤ innerCandidates49_5.length then search edge 3 [239, 929] innerCandidates49_5 1168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_6

def innerTail49_4 : List Nat := 641 :: innerTail49_5
def innerCandidates49_4 : List Nat := [929, 4787, 6197]
set_option maxRecDepth 4096 in
theorem innerCandidates49_4_checked : candidates edge 4 239 641 innerTail49_5 best.bound = innerCandidates49_4 := by
  unfold candidates
  have row : edge 641 = PrimePairCache.smallRow641.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild49_4 : search edge 3 [239, 641] innerCandidates49_4 880 best = best := by decide

theorem innerScan49_4 : scan edge (search edge 3) 4 [239] 239 innerTail49_4 best = best := by
  rw [innerTail49_4, scan, if_neg (by decide), innerCandidates49_4_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_5
    (if 3 ≤ innerCandidates49_4.length then search edge 3 [239, 641] innerCandidates49_4 880 best else best) = best
  rw [if_pos (by decide), innerChild49_4]
  exact innerScan49_5

def innerTail49_3 : List Nat := 509 :: innerTail49_4
def innerCandidates49_3 : List Nat := [947, 1091, 4241]
set_option maxRecDepth 4096 in
theorem innerCandidates49_3_checked : candidates edge 4 239 509 innerTail49_4 best.bound = innerCandidates49_3 := by
  unfold candidates
  have row : edge 509 = PrimePairCache.smallRow509.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild49_3 : search edge 3 [239, 509] innerCandidates49_3 748 best = best := by decide

theorem innerScan49_3 : scan edge (search edge 3) 4 [239] 239 innerTail49_3 best = best := by
  rw [innerTail49_3, scan, if_neg (by decide), innerCandidates49_3_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_4
    (if 3 ≤ innerCandidates49_3.length then search edge 3 [239, 509] innerCandidates49_3 748 best else best) = best
  rw [if_pos (by decide), innerChild49_3]
  exact innerScan49_4

def innerTail49_2 : List Nat := 461 :: innerTail49_3
def innerCandidates49_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates49_2_checked : candidates edge 4 239 461 innerTail49_3 best.bound = innerCandidates49_2 := by
  unfold candidates
  have row : edge 461 = PrimePairCache.smallRow461.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_2 : scan edge (search edge 3) 4 [239] 239 innerTail49_2 best = best := by
  rw [innerTail49_2, scan, if_neg (by decide), innerCandidates49_2_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_3
    (if 3 ≤ innerCandidates49_2.length then search edge 3 [239, 461] innerCandidates49_2 700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_3

def innerTail49_1 : List Nat := 347 :: innerTail49_2
def innerCandidates49_1 : List Nat := [1847, 6257]
set_option maxRecDepth 4096 in
theorem innerCandidates49_1_checked : candidates edge 4 239 347 innerTail49_2 best.bound = innerCandidates49_1 := by
  unfold candidates
  have row : edge 347 = PrimePairCache.smallRow347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan49_1 : scan edge (search edge 3) 4 [239] 239 innerTail49_1 best = best := by
  rw [innerTail49_1, scan, if_neg (by decide), innerCandidates49_1_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_2
    (if 3 ≤ innerCandidates49_1.length then search edge 3 [239, 347] innerCandidates49_1 586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan49_2

def innerTail49_0 : List Nat := 263 :: innerTail49_1
def innerCandidates49_0 : List Nat := [1049, 1091, 3467, 6197, 8087]
set_option maxRecDepth 4096 in
theorem innerCandidates49_0_checked : candidates edge 4 239 263 innerTail49_1 best.bound = innerCandidates49_0 := by
  unfold candidates
  have row : edge 263 = PrimePairCache.smallRow263.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild49_0 : search edge 3 [239, 263] innerCandidates49_0 502 best = best := by decide

theorem innerScan49_0 : scan edge (search edge 3) 4 [239] 239 innerTail49_0 best = best := by
  rw [innerTail49_0, scan, if_neg (by decide), innerCandidates49_0_checked]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_1
    (if 3 ≤ innerCandidates49_0.length then search edge 3 [239, 263] innerCandidates49_0 502 best else best) = best
  rw [if_pos (by decide), innerChild49_0]
  exact innerScan49_1

theorem rootBranch49 : search edge 4 [239] rootCandidates49 239 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [239] 239 innerTail49_0 best = best
  exact innerScan49_0

end ProjectEulerSolutions.PrimePairUpper.Search
