import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates20 : List Nat := [227, 311, 443, 449, 563, 701, 719, 773, 911, 1031, 1301, 1373, 1409, 1553, 1619, 1697, 1811, 1889, 2003, 2063, 2339, 2411, 2591, 2633, 2687, 2693, 2729, 2801, 2969, 3191, 3347, 3491, 3557, 3659, 3923, 4643, 5039, 5441, 5591, 5717, 5897, 5987, 6203, 6491, 7079, 7283, 7307, 7583, 8093, 8171]
set_option maxRecDepth 4096 in
theorem rootCandidates20_checked : candidates edge 5 0 83 tail21 best.bound = rootCandidates20 := by
  unfold candidates
  have row : edge 83 = PrimePairCache.smallRow83.contains := by funext q; rfl
  rw [row]
  decide

def innerTail20_50 : List Nat := []
theorem innerScan20_50 : scan edge (search edge 3) 4 [83] 83 innerTail20_50 best = best := rfl

def innerTail20_49 : List Nat := 8171 :: innerTail20_50
def innerCandidates20_49 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates20_49_checked : candidates edge 4 83 8171 innerTail20_50 best.bound = innerCandidates20_49 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_49 : scan edge (search edge 3) 4 [83] 83 innerTail20_49 best = best := by
  rw [innerTail20_49, scan, if_neg (by decide), innerCandidates20_49_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_50
    (if 3 ≤ innerCandidates20_49.length then search edge 3 [83, 8171] innerCandidates20_49 8254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_50

def innerTail20_48 : List Nat := 8093 :: innerTail20_49
def innerCandidates20_48 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates20_48_checked : candidates edge 4 83 8093 innerTail20_49 best.bound = innerCandidates20_48 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_48 : scan edge (search edge 3) 4 [83] 83 innerTail20_48 best = best := by
  rw [innerTail20_48, scan, if_neg (by decide), innerCandidates20_48_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_49
    (if 3 ≤ innerCandidates20_48.length then search edge 3 [83, 8093] innerCandidates20_48 8176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_49

def innerTail20_47 : List Nat := 7583 :: innerTail20_48
def innerCandidates20_47 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates20_47_checked : candidates edge 4 83 7583 innerTail20_48 best.bound = innerCandidates20_47 := by
  unfold candidates
  have row : edge 7583 = SearchData.row7583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_47 : scan edge (search edge 3) 4 [83] 83 innerTail20_47 best = best := by
  rw [innerTail20_47, scan, if_neg (by decide), innerCandidates20_47_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_48
    (if 3 ≤ innerCandidates20_47.length then search edge 3 [83, 7583] innerCandidates20_47 7666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_48

def innerTail20_46 : List Nat := 7307 :: innerTail20_47
def innerCandidates20_46 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates20_46_checked : candidates edge 4 83 7307 innerTail20_47 best.bound = innerCandidates20_46 := by
  unfold candidates
  have row : edge 7307 = SearchData.row7307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_46 : scan edge (search edge 3) 4 [83] 83 innerTail20_46 best = best := by
  rw [innerTail20_46, scan, if_neg (by decide), innerCandidates20_46_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_47
    (if 3 ≤ innerCandidates20_46.length then search edge 3 [83, 7307] innerCandidates20_46 7390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_47

def innerTail20_45 : List Nat := 7283 :: innerTail20_46
def innerCandidates20_45 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates20_45_checked : candidates edge 4 83 7283 innerTail20_46 best.bound = innerCandidates20_45 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_45 : scan edge (search edge 3) 4 [83] 83 innerTail20_45 best = best := by
  rw [innerTail20_45, scan, if_neg (by decide), innerCandidates20_45_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_46
    (if 3 ≤ innerCandidates20_45.length then search edge 3 [83, 7283] innerCandidates20_45 7366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_46

def innerTail20_44 : List Nat := 7079 :: innerTail20_45
def innerCandidates20_44 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates20_44_checked : candidates edge 4 83 7079 innerTail20_45 best.bound = innerCandidates20_44 := by
  unfold candidates
  have row : edge 7079 = SearchData.row7079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_44 : scan edge (search edge 3) 4 [83] 83 innerTail20_44 best = best := by
  rw [innerTail20_44, scan, if_neg (by decide), innerCandidates20_44_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_45
    (if 3 ≤ innerCandidates20_44.length then search edge 3 [83, 7079] innerCandidates20_44 7162 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_45

def innerTail20_43 : List Nat := 6491 :: innerTail20_44
def innerCandidates20_43 : List Nat := [7583]
set_option maxRecDepth 4096 in
theorem innerCandidates20_43_checked : candidates edge 4 83 6491 innerTail20_44 best.bound = innerCandidates20_43 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_43 : scan edge (search edge 3) 4 [83] 83 innerTail20_43 best = best := by
  rw [innerTail20_43, scan, if_neg (by decide), innerCandidates20_43_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_44
    (if 3 ≤ innerCandidates20_43.length then search edge 3 [83, 6491] innerCandidates20_43 6574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_44

def innerTail20_42 : List Nat := 6203 :: innerTail20_43
def innerCandidates20_42 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates20_42_checked : candidates edge 4 83 6203 innerTail20_43 best.bound = innerCandidates20_42 := by
  unfold candidates
  have row : edge 6203 = SearchData.row6203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_42 : scan edge (search edge 3) 4 [83] 83 innerTail20_42 best = best := by
  rw [innerTail20_42, scan, if_neg (by decide), innerCandidates20_42_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_43
    (if 3 ≤ innerCandidates20_42.length then search edge 3 [83, 6203] innerCandidates20_42 6286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_43

def innerTail20_41 : List Nat := 5987 :: innerTail20_42
def innerCandidates20_41 : List Nat := [6491]
set_option maxRecDepth 4096 in
theorem innerCandidates20_41_checked : candidates edge 4 83 5987 innerTail20_42 best.bound = innerCandidates20_41 := by
  unfold candidates
  have row : edge 5987 = SearchData.row5987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_41 : scan edge (search edge 3) 4 [83] 83 innerTail20_41 best = best := by
  rw [innerTail20_41, scan, if_neg (by decide), innerCandidates20_41_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_42
    (if 3 ≤ innerCandidates20_41.length then search edge 3 [83, 5987] innerCandidates20_41 6070 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_42

def innerTail20_40 : List Nat := 5897 :: innerTail20_41
def innerCandidates20_40 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates20_40_checked : candidates edge 4 83 5897 innerTail20_41 best.bound = innerCandidates20_40 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_40 : scan edge (search edge 3) 4 [83] 83 innerTail20_40 best = best := by
  rw [innerTail20_40, scan, if_neg (by decide), innerCandidates20_40_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_41
    (if 3 ≤ innerCandidates20_40.length then search edge 3 [83, 5897] innerCandidates20_40 5980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_41

def innerTail20_39 : List Nat := 5717 :: innerTail20_40
def innerCandidates20_39 : List Nat := [7079, 8171]
set_option maxRecDepth 4096 in
theorem innerCandidates20_39_checked : candidates edge 4 83 5717 innerTail20_40 best.bound = innerCandidates20_39 := by
  unfold candidates
  have row : edge 5717 = SearchData.row5717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_39 : scan edge (search edge 3) 4 [83] 83 innerTail20_39 best = best := by
  rw [innerTail20_39, scan, if_neg (by decide), innerCandidates20_39_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_40
    (if 3 ≤ innerCandidates20_39.length then search edge 3 [83, 5717] innerCandidates20_39 5800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_40

def innerTail20_38 : List Nat := 5591 :: innerTail20_39
def innerCandidates20_38 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates20_38_checked : candidates edge 4 83 5591 innerTail20_39 best.bound = innerCandidates20_38 := by
  unfold candidates
  have row : edge 5591 = SearchData.row5591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_38 : scan edge (search edge 3) 4 [83] 83 innerTail20_38 best = best := by
  rw [innerTail20_38, scan, if_neg (by decide), innerCandidates20_38_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_39
    (if 3 ≤ innerCandidates20_38.length then search edge 3 [83, 5591] innerCandidates20_38 5674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_39

def innerTail20_37 : List Nat := 5441 :: innerTail20_38
def innerCandidates20_37 : List Nat := [5717]
set_option maxRecDepth 4096 in
theorem innerCandidates20_37_checked : candidates edge 4 83 5441 innerTail20_38 best.bound = innerCandidates20_37 := by
  unfold candidates
  have row : edge 5441 = SearchData.row5441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_37 : scan edge (search edge 3) 4 [83] 83 innerTail20_37 best = best := by
  rw [innerTail20_37, scan, if_neg (by decide), innerCandidates20_37_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_38
    (if 3 ≤ innerCandidates20_37.length then search edge 3 [83, 5441] innerCandidates20_37 5524 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_38

def innerTail20_36 : List Nat := 5039 :: innerTail20_37
def innerCandidates20_36 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates20_36_checked : candidates edge 4 83 5039 innerTail20_37 best.bound = innerCandidates20_36 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_36 : scan edge (search edge 3) 4 [83] 83 innerTail20_36 best = best := by
  rw [innerTail20_36, scan, if_neg (by decide), innerCandidates20_36_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_37
    (if 3 ≤ innerCandidates20_36.length then search edge 3 [83, 5039] innerCandidates20_36 5122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_37

def innerTail20_35 : List Nat := 4643 :: innerTail20_36
def innerCandidates20_35 : List Nat := [5717]
set_option maxRecDepth 4096 in
theorem innerCandidates20_35_checked : candidates edge 4 83 4643 innerTail20_36 best.bound = innerCandidates20_35 := by
  unfold candidates
  have row : edge 4643 = SearchData.row4643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_35 : scan edge (search edge 3) 4 [83] 83 innerTail20_35 best = best := by
  rw [innerTail20_35, scan, if_neg (by decide), innerCandidates20_35_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_36
    (if 3 ≤ innerCandidates20_35.length then search edge 3 [83, 4643] innerCandidates20_35 4726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_36

def innerTail20_34 : List Nat := 3923 :: innerTail20_35
def innerCandidates20_34 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates20_34_checked : candidates edge 4 83 3923 innerTail20_35 best.bound = innerCandidates20_34 := by
  unfold candidates
  have row : edge 3923 = SearchData.row3923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_34 : scan edge (search edge 3) 4 [83] 83 innerTail20_34 best = best := by
  rw [innerTail20_34, scan, if_neg (by decide), innerCandidates20_34_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_35
    (if 3 ≤ innerCandidates20_34.length then search edge 3 [83, 3923] innerCandidates20_34 4006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_35

def innerTail20_33 : List Nat := 3659 :: innerTail20_34
def innerCandidates20_33 : List Nat := [4643]
set_option maxRecDepth 4096 in
theorem innerCandidates20_33_checked : candidates edge 4 83 3659 innerTail20_34 best.bound = innerCandidates20_33 := by
  unfold candidates
  have row : edge 3659 = SearchData.row3659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_33 : scan edge (search edge 3) 4 [83] 83 innerTail20_33 best = best := by
  rw [innerTail20_33, scan, if_neg (by decide), innerCandidates20_33_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_34
    (if 3 ≤ innerCandidates20_33.length then search edge 3 [83, 3659] innerCandidates20_33 3742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_34

def innerTail20_32 : List Nat := 3557 :: innerTail20_33
def innerCandidates20_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates20_32_checked : candidates edge 4 83 3557 innerTail20_33 best.bound = innerCandidates20_32 := by
  unfold candidates
  have row : edge 3557 = SearchData.row3557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_32 : scan edge (search edge 3) 4 [83] 83 innerTail20_32 best = best := by
  rw [innerTail20_32, scan, if_neg (by decide), innerCandidates20_32_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_33
    (if 3 ≤ innerCandidates20_32.length then search edge 3 [83, 3557] innerCandidates20_32 3640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_33

def innerTail20_31 : List Nat := 3491 :: innerTail20_32
def innerCandidates20_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates20_31_checked : candidates edge 4 83 3491 innerTail20_32 best.bound = innerCandidates20_31 := by
  unfold candidates
  have row : edge 3491 = SearchData.row3491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_31 : scan edge (search edge 3) 4 [83] 83 innerTail20_31 best = best := by
  rw [innerTail20_31, scan, if_neg (by decide), innerCandidates20_31_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_32
    (if 3 ≤ innerCandidates20_31.length then search edge 3 [83, 3491] innerCandidates20_31 3574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_32

def innerTail20_30 : List Nat := 3347 :: innerTail20_31
def innerCandidates20_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates20_30_checked : candidates edge 4 83 3347 innerTail20_31 best.bound = innerCandidates20_30 := by
  unfold candidates
  have row : edge 3347 = SearchData.row3347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_30 : scan edge (search edge 3) 4 [83] 83 innerTail20_30 best = best := by
  rw [innerTail20_30, scan, if_neg (by decide), innerCandidates20_30_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_31
    (if 3 ≤ innerCandidates20_30.length then search edge 3 [83, 3347] innerCandidates20_30 3430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_31

def innerTail20_29 : List Nat := 3191 :: innerTail20_30
def innerCandidates20_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates20_29_checked : candidates edge 4 83 3191 innerTail20_30 best.bound = innerCandidates20_29 := by
  unfold candidates
  have row : edge 3191 = SearchData.row3191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_29 : scan edge (search edge 3) 4 [83] 83 innerTail20_29 best = best := by
  rw [innerTail20_29, scan, if_neg (by decide), innerCandidates20_29_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_30
    (if 3 ≤ innerCandidates20_29.length then search edge 3 [83, 3191] innerCandidates20_29 3274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_30

def innerTail20_28 : List Nat := 2969 :: innerTail20_29
def innerCandidates20_28 : List Nat := [3557]
set_option maxRecDepth 4096 in
theorem innerCandidates20_28_checked : candidates edge 4 83 2969 innerTail20_29 best.bound = innerCandidates20_28 := by
  unfold candidates
  have row : edge 2969 = SearchData.row2969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_28 : scan edge (search edge 3) 4 [83] 83 innerTail20_28 best = best := by
  rw [innerTail20_28, scan, if_neg (by decide), innerCandidates20_28_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_29
    (if 3 ≤ innerCandidates20_28.length then search edge 3 [83, 2969] innerCandidates20_28 3052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_29

def innerTail20_27 : List Nat := 2801 :: innerTail20_28
def innerCandidates20_27 : List Nat := [5987, 8171]
set_option maxRecDepth 4096 in
theorem innerCandidates20_27_checked : candidates edge 4 83 2801 innerTail20_28 best.bound = innerCandidates20_27 := by
  unfold candidates
  have row : edge 2801 = SearchData.row2801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_27 : scan edge (search edge 3) 4 [83] 83 innerTail20_27 best = best := by
  rw [innerTail20_27, scan, if_neg (by decide), innerCandidates20_27_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_28
    (if 3 ≤ innerCandidates20_27.length then search edge 3 [83, 2801] innerCandidates20_27 2884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_28

def innerTail20_26 : List Nat := 2729 :: innerTail20_27
def innerCandidates20_26 : List Nat := [3191]
set_option maxRecDepth 4096 in
theorem innerCandidates20_26_checked : candidates edge 4 83 2729 innerTail20_27 best.bound = innerCandidates20_26 := by
  unfold candidates
  have row : edge 2729 = SearchData.row2729.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_26 : scan edge (search edge 3) 4 [83] 83 innerTail20_26 best = best := by
  rw [innerTail20_26, scan, if_neg (by decide), innerCandidates20_26_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_27
    (if 3 ≤ innerCandidates20_26.length then search edge 3 [83, 2729] innerCandidates20_26 2812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_27

def innerTail20_25 : List Nat := 2693 :: innerTail20_26
def innerCandidates20_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates20_25_checked : candidates edge 4 83 2693 innerTail20_26 best.bound = innerCandidates20_25 := by
  unfold candidates
  have row : edge 2693 = SearchData.row2693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_25 : scan edge (search edge 3) 4 [83] 83 innerTail20_25 best = best := by
  rw [innerTail20_25, scan, if_neg (by decide), innerCandidates20_25_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_26
    (if 3 ≤ innerCandidates20_25.length then search edge 3 [83, 2693] innerCandidates20_25 2776 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_26

def innerTail20_24 : List Nat := 2687 :: innerTail20_25
def innerCandidates20_24 : List Nat := [5897]
set_option maxRecDepth 4096 in
theorem innerCandidates20_24_checked : candidates edge 4 83 2687 innerTail20_25 best.bound = innerCandidates20_24 := by
  unfold candidates
  have row : edge 2687 = SearchData.row2687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_24 : scan edge (search edge 3) 4 [83] 83 innerTail20_24 best = best := by
  rw [innerTail20_24, scan, if_neg (by decide), innerCandidates20_24_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_25
    (if 3 ≤ innerCandidates20_24.length then search edge 3 [83, 2687] innerCandidates20_24 2770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_25

def innerTail20_23 : List Nat := 2633 :: innerTail20_24
def innerCandidates20_23 : List Nat := [7079]
set_option maxRecDepth 4096 in
theorem innerCandidates20_23_checked : candidates edge 4 83 2633 innerTail20_24 best.bound = innerCandidates20_23 := by
  unfold candidates
  have row : edge 2633 = PrimePairCache.smallRow2633.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_23 : scan edge (search edge 3) 4 [83] 83 innerTail20_23 best = best := by
  rw [innerTail20_23, scan, if_neg (by decide), innerCandidates20_23_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_24
    (if 3 ≤ innerCandidates20_23.length then search edge 3 [83, 2633] innerCandidates20_23 2716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_24

def innerTail20_22 : List Nat := 2591 :: innerTail20_23
def innerCandidates20_22 : List Nat := [2633]
set_option maxRecDepth 4096 in
theorem innerCandidates20_22_checked : candidates edge 4 83 2591 innerTail20_23 best.bound = innerCandidates20_22 := by
  unfold candidates
  have row : edge 2591 = PrimePairCache.smallRow2591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_22 : scan edge (search edge 3) 4 [83] 83 innerTail20_22 best = best := by
  rw [innerTail20_22, scan, if_neg (by decide), innerCandidates20_22_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_23
    (if 3 ≤ innerCandidates20_22.length then search edge 3 [83, 2591] innerCandidates20_22 2674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_23

def innerTail20_21 : List Nat := 2411 :: innerTail20_22
def innerCandidates20_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates20_21_checked : candidates edge 4 83 2411 innerTail20_22 best.bound = innerCandidates20_21 := by
  unfold candidates
  have row : edge 2411 = PrimePairCache.smallRow2411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_21 : scan edge (search edge 3) 4 [83] 83 innerTail20_21 best = best := by
  rw [innerTail20_21, scan, if_neg (by decide), innerCandidates20_21_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_22
    (if 3 ≤ innerCandidates20_21.length then search edge 3 [83, 2411] innerCandidates20_21 2494 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_22

def innerTail20_20 : List Nat := 2339 :: innerTail20_21
def innerCandidates20_20 : List Nat := [2729, 3347, 5039, 7307]
set_option maxRecDepth 4096 in
theorem innerCandidates20_20_checked : candidates edge 4 83 2339 innerTail20_21 best.bound = innerCandidates20_20 := by
  unfold candidates
  have row : edge 2339 = PrimePairCache.smallRow2339.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild20_20 : search edge 3 [83, 2339] innerCandidates20_20 2422 best = best := by decide

theorem innerScan20_20 : scan edge (search edge 3) 4 [83] 83 innerTail20_20 best = best := by
  rw [innerTail20_20, scan, if_neg (by decide), innerCandidates20_20_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_21
    (if 3 ≤ innerCandidates20_20.length then search edge 3 [83, 2339] innerCandidates20_20 2422 best else best) = best
  rw [if_pos (by decide), innerChild20_20]
  exact innerScan20_21

def innerTail20_19 : List Nat := 2063 :: innerTail20_20
def innerCandidates20_19 : List Nat := [2411]
set_option maxRecDepth 4096 in
theorem innerCandidates20_19_checked : candidates edge 4 83 2063 innerTail20_20 best.bound = innerCandidates20_19 := by
  unfold candidates
  have row : edge 2063 = PrimePairCache.smallRow2063.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_19 : scan edge (search edge 3) 4 [83] 83 innerTail20_19 best = best := by
  rw [innerTail20_19, scan, if_neg (by decide), innerCandidates20_19_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_20
    (if 3 ≤ innerCandidates20_19.length then search edge 3 [83, 2063] innerCandidates20_19 2146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_20

def innerTail20_18 : List Nat := 2003 :: innerTail20_19
def innerCandidates20_18 : List Nat := [3557, 5039, 8171]
set_option maxRecDepth 4096 in
theorem innerCandidates20_18_checked : candidates edge 4 83 2003 innerTail20_19 best.bound = innerCandidates20_18 := by
  unfold candidates
  have row : edge 2003 = PrimePairCache.smallRow2003.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild20_18 : search edge 3 [83, 2003] innerCandidates20_18 2086 best = best := by decide

theorem innerScan20_18 : scan edge (search edge 3) 4 [83] 83 innerTail20_18 best = best := by
  rw [innerTail20_18, scan, if_neg (by decide), innerCandidates20_18_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_19
    (if 3 ≤ innerCandidates20_18.length then search edge 3 [83, 2003] innerCandidates20_18 2086 best else best) = best
  rw [if_pos (by decide), innerChild20_18]
  exact innerScan20_19

def innerTail20_17 : List Nat := 1889 :: innerTail20_18
def innerCandidates20_17 : List Nat := [2633, 3491]
set_option maxRecDepth 4096 in
theorem innerCandidates20_17_checked : candidates edge 4 83 1889 innerTail20_18 best.bound = innerCandidates20_17 := by
  unfold candidates
  have row : edge 1889 = PrimePairCache.smallRow1889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_17 : scan edge (search edge 3) 4 [83] 83 innerTail20_17 best = best := by
  rw [innerTail20_17, scan, if_neg (by decide), innerCandidates20_17_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_18
    (if 3 ≤ innerCandidates20_17.length then search edge 3 [83, 1889] innerCandidates20_17 1972 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_18

def innerTail20_16 : List Nat := 1811 :: innerTail20_17
def innerCandidates20_16 : List Nat := [5039]
set_option maxRecDepth 4096 in
theorem innerCandidates20_16_checked : candidates edge 4 83 1811 innerTail20_17 best.bound = innerCandidates20_16 := by
  unfold candidates
  have row : edge 1811 = PrimePairCache.smallRow1811.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_16 : scan edge (search edge 3) 4 [83] 83 innerTail20_16 best = best := by
  rw [innerTail20_16, scan, if_neg (by decide), innerCandidates20_16_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_17
    (if 3 ≤ innerCandidates20_16.length then search edge 3 [83, 1811] innerCandidates20_16 1894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_17

def innerTail20_15 : List Nat := 1697 :: innerTail20_16
def innerCandidates20_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates20_15_checked : candidates edge 4 83 1697 innerTail20_16 best.bound = innerCandidates20_15 := by
  unfold candidates
  have row : edge 1697 = PrimePairCache.smallRow1697.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_15 : scan edge (search edge 3) 4 [83] 83 innerTail20_15 best = best := by
  rw [innerTail20_15, scan, if_neg (by decide), innerCandidates20_15_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_16
    (if 3 ≤ innerCandidates20_15.length then search edge 3 [83, 1697] innerCandidates20_15 1780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_16

def innerTail20_14 : List Nat := 1619 :: innerTail20_15
def innerCandidates20_14 : List Nat := [2003, 2339, 5987]
set_option maxRecDepth 4096 in
theorem innerCandidates20_14_checked : candidates edge 4 83 1619 innerTail20_15 best.bound = innerCandidates20_14 := by
  unfold candidates
  have row : edge 1619 = PrimePairCache.smallRow1619.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild20_14 : search edge 3 [83, 1619] innerCandidates20_14 1702 best = best := by decide

theorem innerScan20_14 : scan edge (search edge 3) 4 [83] 83 innerTail20_14 best = best := by
  rw [innerTail20_14, scan, if_neg (by decide), innerCandidates20_14_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_15
    (if 3 ≤ innerCandidates20_14.length then search edge 3 [83, 1619] innerCandidates20_14 1702 best else best) = best
  rw [if_pos (by decide), innerChild20_14]
  exact innerScan20_15

def innerTail20_13 : List Nat := 1553 :: innerTail20_14
def innerCandidates20_13 : List Nat := [1697, 1811, 2687, 6203]
set_option maxRecDepth 4096 in
theorem innerCandidates20_13_checked : candidates edge 4 83 1553 innerTail20_14 best.bound = innerCandidates20_13 := by
  unfold candidates
  have row : edge 1553 = PrimePairCache.smallRow1553.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild20_13 : search edge 3 [83, 1553] innerCandidates20_13 1636 best = best := by decide

theorem innerScan20_13 : scan edge (search edge 3) 4 [83] 83 innerTail20_13 best = best := by
  rw [innerTail20_13, scan, if_neg (by decide), innerCandidates20_13_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_14
    (if 3 ≤ innerCandidates20_13.length then search edge 3 [83, 1553] innerCandidates20_13 1636 best else best) = best
  rw [if_pos (by decide), innerChild20_13]
  exact innerScan20_14

def innerTail20_12 : List Nat := 1409 :: innerTail20_13
def innerCandidates20_12 : List Nat := [4643]
set_option maxRecDepth 4096 in
theorem innerCandidates20_12_checked : candidates edge 4 83 1409 innerTail20_13 best.bound = innerCandidates20_12 := by
  unfold candidates
  have row : edge 1409 = PrimePairCache.smallRow1409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_12 : scan edge (search edge 3) 4 [83] 83 innerTail20_12 best = best := by
  rw [innerTail20_12, scan, if_neg (by decide), innerCandidates20_12_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_13
    (if 3 ≤ innerCandidates20_12.length then search edge 3 [83, 1409] innerCandidates20_12 1492 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_13

def innerTail20_11 : List Nat := 1373 :: innerTail20_12
def innerCandidates20_11 : List Nat := [2591, 6491]
set_option maxRecDepth 4096 in
theorem innerCandidates20_11_checked : candidates edge 4 83 1373 innerTail20_12 best.bound = innerCandidates20_11 := by
  unfold candidates
  have row : edge 1373 = PrimePairCache.smallRow1373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_11 : scan edge (search edge 3) 4 [83] 83 innerTail20_11 best = best := by
  rw [innerTail20_11, scan, if_neg (by decide), innerCandidates20_11_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_12
    (if 3 ≤ innerCandidates20_11.length then search edge 3 [83, 1373] innerCandidates20_11 1456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_12

def innerTail20_10 : List Nat := 1301 :: innerTail20_11
def innerCandidates20_10 : List Nat := [2969, 7283]
set_option maxRecDepth 4096 in
theorem innerCandidates20_10_checked : candidates edge 4 83 1301 innerTail20_11 best.bound = innerCandidates20_10 := by
  unfold candidates
  have row : edge 1301 = PrimePairCache.smallRow1301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_10 : scan edge (search edge 3) 4 [83] 83 innerTail20_10 best = best := by
  rw [innerTail20_10, scan, if_neg (by decide), innerCandidates20_10_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_11
    (if 3 ≤ innerCandidates20_10.length then search edge 3 [83, 1301] innerCandidates20_10 1384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_11

def innerTail20_9 : List Nat := 1031 :: innerTail20_10
def innerCandidates20_9 : List Nat := [1811, 4643]
set_option maxRecDepth 4096 in
theorem innerCandidates20_9_checked : candidates edge 4 83 1031 innerTail20_10 best.bound = innerCandidates20_9 := by
  unfold candidates
  have row : edge 1031 = PrimePairCache.smallRow1031.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_9 : scan edge (search edge 3) 4 [83] 83 innerTail20_9 best = best := by
  rw [innerTail20_9, scan, if_neg (by decide), innerCandidates20_9_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_10
    (if 3 ≤ innerCandidates20_9.length then search edge 3 [83, 1031] innerCandidates20_9 1114 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_10

def innerTail20_8 : List Nat := 911 :: innerTail20_9
def innerCandidates20_8 : List Nat := [4643]
set_option maxRecDepth 4096 in
theorem innerCandidates20_8_checked : candidates edge 4 83 911 innerTail20_9 best.bound = innerCandidates20_8 := by
  unfold candidates
  have row : edge 911 = PrimePairCache.smallRow911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_8 : scan edge (search edge 3) 4 [83] 83 innerTail20_8 best = best := by
  rw [innerTail20_8, scan, if_neg (by decide), innerCandidates20_8_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_9
    (if 3 ≤ innerCandidates20_8.length then search edge 3 [83, 911] innerCandidates20_8 994 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_9

def innerTail20_7 : List Nat := 773 :: innerTail20_8
def innerCandidates20_7 : List Nat := [2063, 2339, 2591]
set_option maxRecDepth 4096 in
theorem innerCandidates20_7_checked : candidates edge 4 83 773 innerTail20_8 best.bound = innerCandidates20_7 := by
  unfold candidates
  have row : edge 773 = PrimePairCache.smallRow773.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild20_7 : search edge 3 [83, 773] innerCandidates20_7 856 best = best := by decide

theorem innerScan20_7 : scan edge (search edge 3) 4 [83] 83 innerTail20_7 best = best := by
  rw [innerTail20_7, scan, if_neg (by decide), innerCandidates20_7_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_8
    (if 3 ≤ innerCandidates20_7.length then search edge 3 [83, 773] innerCandidates20_7 856 best else best) = best
  rw [if_pos (by decide), innerChild20_7]
  exact innerScan20_8

def innerTail20_6 : List Nat := 719 :: innerTail20_7
def innerCandidates20_6 : List Nat := [773, 911, 5039, 7079, 7283]
set_option maxRecDepth 4096 in
theorem innerCandidates20_6_checked : candidates edge 4 83 719 innerTail20_7 best.bound = innerCandidates20_6 := by
  unfold candidates
  have row : edge 719 = PrimePairCache.smallRow719.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild20_6 : search edge 3 [83, 719] innerCandidates20_6 802 best = best := by decide

theorem innerScan20_6 : scan edge (search edge 3) 4 [83] 83 innerTail20_6 best = best := by
  rw [innerTail20_6, scan, if_neg (by decide), innerCandidates20_6_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_7
    (if 3 ≤ innerCandidates20_6.length then search edge 3 [83, 719] innerCandidates20_6 802 best else best) = best
  rw [if_pos (by decide), innerChild20_6]
  exact innerScan20_7

def innerTail20_5 : List Nat := 701 :: innerTail20_6
def innerCandidates20_5 : List Nat := [1553, 8171]
set_option maxRecDepth 4096 in
theorem innerCandidates20_5_checked : candidates edge 4 83 701 innerTail20_6 best.bound = innerCandidates20_5 := by
  unfold candidates
  have row : edge 701 = PrimePairCache.smallRow701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_5 : scan edge (search edge 3) 4 [83] 83 innerTail20_5 best = best := by
  rw [innerTail20_5, scan, if_neg (by decide), innerCandidates20_5_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_6
    (if 3 ≤ innerCandidates20_5.length then search edge 3 [83, 701] innerCandidates20_5 784 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_6

def innerTail20_4 : List Nat := 563 :: innerTail20_5
def innerCandidates20_4 : List Nat := [5717]
set_option maxRecDepth 4096 in
theorem innerCandidates20_4_checked : candidates edge 4 83 563 innerTail20_5 best.bound = innerCandidates20_4 := by
  unfold candidates
  have row : edge 563 = PrimePairCache.smallRow563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_4 : scan edge (search edge 3) 4 [83] 83 innerTail20_4 best = best := by
  rw [innerTail20_4, scan, if_neg (by decide), innerCandidates20_4_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_5
    (if 3 ≤ innerCandidates20_4.length then search edge 3 [83, 563] innerCandidates20_4 646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_5

def innerTail20_3 : List Nat := 449 :: innerTail20_4
def innerCandidates20_3 : List Nat := [563, 3659, 6491, 7583]
set_option maxRecDepth 4096 in
theorem innerCandidates20_3_checked : candidates edge 4 83 449 innerTail20_4 best.bound = innerCandidates20_3 := by
  unfold candidates
  have row : edge 449 = PrimePairCache.smallRow449.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild20_3 : search edge 3 [83, 449] innerCandidates20_3 532 best = best := by decide

theorem innerScan20_3 : scan edge (search edge 3) 4 [83] 83 innerTail20_3 best = best := by
  rw [innerTail20_3, scan, if_neg (by decide), innerCandidates20_3_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_4
    (if 3 ≤ innerCandidates20_3.length then search edge 3 [83, 449] innerCandidates20_3 532 best else best) = best
  rw [if_pos (by decide), innerChild20_3]
  exact innerScan20_4

def innerTail20_2 : List Nat := 443 :: innerTail20_3
def innerCandidates20_2 : List Nat := [701, 1811, 2693, 5897, 7283]
set_option maxRecDepth 4096 in
theorem innerCandidates20_2_checked : candidates edge 4 83 443 innerTail20_3 best.bound = innerCandidates20_2 := by
  unfold candidates
  have row : edge 443 = PrimePairCache.smallRow443.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild20_2 : search edge 3 [83, 443] innerCandidates20_2 526 best = best := by decide

theorem innerScan20_2 : scan edge (search edge 3) 4 [83] 83 innerTail20_2 best = best := by
  rw [innerTail20_2, scan, if_neg (by decide), innerCandidates20_2_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_3
    (if 3 ≤ innerCandidates20_2.length then search edge 3 [83, 443] innerCandidates20_2 526 best else best) = best
  rw [if_pos (by decide), innerChild20_2]
  exact innerScan20_3

def innerTail20_1 : List Nat := 311 :: innerTail20_2
def innerCandidates20_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates20_1_checked : candidates edge 4 83 311 innerTail20_2 best.bound = innerCandidates20_1 := by
  unfold candidates
  have row : edge 311 = PrimePairCache.smallRow311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan20_1 : scan edge (search edge 3) 4 [83] 83 innerTail20_1 best = best := by
  rw [innerTail20_1, scan, if_neg (by decide), innerCandidates20_1_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_2
    (if 3 ≤ innerCandidates20_1.length then search edge 3 [83, 311] innerCandidates20_1 394 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan20_2

def innerTail20_0 : List Nat := 227 :: innerTail20_1
def innerCandidates20_0 : List Nat := [719, 2729, 5591, 8093]
set_option maxRecDepth 4096 in
theorem innerCandidates20_0_checked : candidates edge 4 83 227 innerTail20_1 best.bound = innerCandidates20_0 := by
  unfold candidates
  have row : edge 227 = PrimePairCache.smallRow227.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild20_0 : search edge 3 [83, 227] innerCandidates20_0 310 best = best := by decide

theorem innerScan20_0 : scan edge (search edge 3) 4 [83] 83 innerTail20_0 best = best := by
  rw [innerTail20_0, scan, if_neg (by decide), innerCandidates20_0_checked]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_1
    (if 3 ≤ innerCandidates20_0.length then search edge 3 [83, 227] innerCandidates20_0 310 best else best) = best
  rw [if_pos (by decide), innerChild20_0]
  exact innerScan20_1

theorem rootBranch20 : search edge 4 [83] rootCandidates20 83 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [83] 83 innerTail20_0 best = best
  exact innerScan20_0

end ProjectEulerSolutions.PrimePairUpper.Search
