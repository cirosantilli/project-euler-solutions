import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates10 : List Nat := [227, 257, 593, 719, 863, 887, 911, 1013, 1031, 1259, 1583, 1613, 1637, 1667, 1823, 2081, 2333, 2339, 2411, 2609, 2939, 3299, 3461, 3527, 3533, 3719, 4013, 4049, 4217, 4451, 4457, 4679, 4691, 5147, 5477, 5717, 6089, 6263, 6473, 6491, 6719, 6833, 7187, 7331, 7451, 7457, 7583, 7643, 7793, 8069, 8219]
set_option maxRecDepth 4096 in
theorem rootCandidates10_checked : candidates edge 5 0 41 tail11 best.bound = rootCandidates10 := by
  unfold candidates
  have row : edge 41 = PrimePairCache.smallRow41.contains := by funext q; rfl
  rw [row]
  decide

def innerTail10_51 : List Nat := []
theorem innerScan10_51 : scan edge (search edge 3) 4 [41] 41 innerTail10_51 best = best := rfl

def innerTail10_50 : List Nat := 8219 :: innerTail10_51
def innerCandidates10_50 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates10_50_checked : candidates edge 4 41 8219 innerTail10_51 best.bound = innerCandidates10_50 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_50 : scan edge (search edge 3) 4 [41] 41 innerTail10_50 best = best := by
  rw [innerTail10_50, scan, if_neg (by decide), innerCandidates10_50_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_51
    (if 3 ≤ innerCandidates10_50.length then search edge 3 [41, 8219] innerCandidates10_50 8260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_51

def innerTail10_49 : List Nat := 8069 :: innerTail10_50
def innerCandidates10_49 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates10_49_checked : candidates edge 4 41 8069 innerTail10_50 best.bound = innerCandidates10_49 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_49 : scan edge (search edge 3) 4 [41] 41 innerTail10_49 best = best := by
  rw [innerTail10_49, scan, if_neg (by decide), innerCandidates10_49_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_50
    (if 3 ≤ innerCandidates10_49.length then search edge 3 [41, 8069] innerCandidates10_49 8110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_50

def innerTail10_48 : List Nat := 7793 :: innerTail10_49
def innerCandidates10_48 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates10_48_checked : candidates edge 4 41 7793 innerTail10_49 best.bound = innerCandidates10_48 := by
  unfold candidates
  have row : edge 7793 = SearchData.row7793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_48 : scan edge (search edge 3) 4 [41] 41 innerTail10_48 best = best := by
  rw [innerTail10_48, scan, if_neg (by decide), innerCandidates10_48_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_49
    (if 3 ≤ innerCandidates10_48.length then search edge 3 [41, 7793] innerCandidates10_48 7834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_49

def innerTail10_47 : List Nat := 7643 :: innerTail10_48
def innerCandidates10_47 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates10_47_checked : candidates edge 4 41 7643 innerTail10_48 best.bound = innerCandidates10_47 := by
  unfold candidates
  have row : edge 7643 = SearchData.row7643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_47 : scan edge (search edge 3) 4 [41] 41 innerTail10_47 best = best := by
  rw [innerTail10_47, scan, if_neg (by decide), innerCandidates10_47_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_48
    (if 3 ≤ innerCandidates10_47.length then search edge 3 [41, 7643] innerCandidates10_47 7684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_48

def innerTail10_46 : List Nat := 7583 :: innerTail10_47
def innerCandidates10_46 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates10_46_checked : candidates edge 4 41 7583 innerTail10_47 best.bound = innerCandidates10_46 := by
  unfold candidates
  have row : edge 7583 = SearchData.row7583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_46 : scan edge (search edge 3) 4 [41] 41 innerTail10_46 best = best := by
  rw [innerTail10_46, scan, if_neg (by decide), innerCandidates10_46_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_47
    (if 3 ≤ innerCandidates10_46.length then search edge 3 [41, 7583] innerCandidates10_46 7624 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_47

def innerTail10_45 : List Nat := 7457 :: innerTail10_46
def innerCandidates10_45 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates10_45_checked : candidates edge 4 41 7457 innerTail10_46 best.bound = innerCandidates10_45 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_45 : scan edge (search edge 3) 4 [41] 41 innerTail10_45 best = best := by
  rw [innerTail10_45, scan, if_neg (by decide), innerCandidates10_45_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_46
    (if 3 ≤ innerCandidates10_45.length then search edge 3 [41, 7457] innerCandidates10_45 7498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_46

def innerTail10_44 : List Nat := 7451 :: innerTail10_45
def innerCandidates10_44 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates10_44_checked : candidates edge 4 41 7451 innerTail10_45 best.bound = innerCandidates10_44 := by
  unfold candidates
  have row : edge 7451 = SearchData.row7451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_44 : scan edge (search edge 3) 4 [41] 41 innerTail10_44 best = best := by
  rw [innerTail10_44, scan, if_neg (by decide), innerCandidates10_44_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_45
    (if 3 ≤ innerCandidates10_44.length then search edge 3 [41, 7451] innerCandidates10_44 7492 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_45

def innerTail10_43 : List Nat := 7331 :: innerTail10_44
def innerCandidates10_43 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates10_43_checked : candidates edge 4 41 7331 innerTail10_44 best.bound = innerCandidates10_43 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_43 : scan edge (search edge 3) 4 [41] 41 innerTail10_43 best = best := by
  rw [innerTail10_43, scan, if_neg (by decide), innerCandidates10_43_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_44
    (if 3 ≤ innerCandidates10_43.length then search edge 3 [41, 7331] innerCandidates10_43 7372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_44

def innerTail10_42 : List Nat := 7187 :: innerTail10_43
def innerCandidates10_42 : List Nat := [7331]
set_option maxRecDepth 4096 in
theorem innerCandidates10_42_checked : candidates edge 4 41 7187 innerTail10_43 best.bound = innerCandidates10_42 := by
  unfold candidates
  have row : edge 7187 = SearchData.row7187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_42 : scan edge (search edge 3) 4 [41] 41 innerTail10_42 best = best := by
  rw [innerTail10_42, scan, if_neg (by decide), innerCandidates10_42_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_43
    (if 3 ≤ innerCandidates10_42.length then search edge 3 [41, 7187] innerCandidates10_42 7228 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_43

def innerTail10_41 : List Nat := 6833 :: innerTail10_42
def innerCandidates10_41 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates10_41_checked : candidates edge 4 41 6833 innerTail10_42 best.bound = innerCandidates10_41 := by
  unfold candidates
  have row : edge 6833 = SearchData.row6833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_41 : scan edge (search edge 3) 4 [41] 41 innerTail10_41 best = best := by
  rw [innerTail10_41, scan, if_neg (by decide), innerCandidates10_41_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_42
    (if 3 ≤ innerCandidates10_41.length then search edge 3 [41, 6833] innerCandidates10_41 6874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_42

def innerTail10_40 : List Nat := 6719 :: innerTail10_41
def innerCandidates10_40 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates10_40_checked : candidates edge 4 41 6719 innerTail10_41 best.bound = innerCandidates10_40 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_40 : scan edge (search edge 3) 4 [41] 41 innerTail10_40 best = best := by
  rw [innerTail10_40, scan, if_neg (by decide), innerCandidates10_40_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_41
    (if 3 ≤ innerCandidates10_40.length then search edge 3 [41, 6719] innerCandidates10_40 6760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_41

def innerTail10_39 : List Nat := 6491 :: innerTail10_40
def innerCandidates10_39 : List Nat := [7583]
set_option maxRecDepth 4096 in
theorem innerCandidates10_39_checked : candidates edge 4 41 6491 innerTail10_40 best.bound = innerCandidates10_39 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_39 : scan edge (search edge 3) 4 [41] 41 innerTail10_39 best = best := by
  rw [innerTail10_39, scan, if_neg (by decide), innerCandidates10_39_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_40
    (if 3 ≤ innerCandidates10_39.length then search edge 3 [41, 6491] innerCandidates10_39 6532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_40

def innerTail10_38 : List Nat := 6473 :: innerTail10_39
def innerCandidates10_38 : List Nat := [7643]
set_option maxRecDepth 4096 in
theorem innerCandidates10_38_checked : candidates edge 4 41 6473 innerTail10_39 best.bound = innerCandidates10_38 := by
  unfold candidates
  have row : edge 6473 = SearchData.row6473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_38 : scan edge (search edge 3) 4 [41] 41 innerTail10_38 best = best := by
  rw [innerTail10_38, scan, if_neg (by decide), innerCandidates10_38_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_39
    (if 3 ≤ innerCandidates10_38.length then search edge 3 [41, 6473] innerCandidates10_38 6514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_39

def innerTail10_37 : List Nat := 6263 :: innerTail10_38
def innerCandidates10_37 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates10_37_checked : candidates edge 4 41 6263 innerTail10_38 best.bound = innerCandidates10_37 := by
  unfold candidates
  have row : edge 6263 = SearchData.row6263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_37 : scan edge (search edge 3) 4 [41] 41 innerTail10_37 best = best := by
  rw [innerTail10_37, scan, if_neg (by decide), innerCandidates10_37_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_38
    (if 3 ≤ innerCandidates10_37.length then search edge 3 [41, 6263] innerCandidates10_37 6304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_38

def innerTail10_36 : List Nat := 6089 :: innerTail10_37
def innerCandidates10_36 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates10_36_checked : candidates edge 4 41 6089 innerTail10_37 best.bound = innerCandidates10_36 := by
  unfold candidates
  have row : edge 6089 = SearchData.row6089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_36 : scan edge (search edge 3) 4 [41] 41 innerTail10_36 best = best := by
  rw [innerTail10_36, scan, if_neg (by decide), innerCandidates10_36_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_37
    (if 3 ≤ innerCandidates10_36.length then search edge 3 [41, 6089] innerCandidates10_36 6130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_37

def innerTail10_35 : List Nat := 5717 :: innerTail10_36
def innerCandidates10_35 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates10_35_checked : candidates edge 4 41 5717 innerTail10_36 best.bound = innerCandidates10_35 := by
  unfold candidates
  have row : edge 5717 = SearchData.row5717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_35 : scan edge (search edge 3) 4 [41] 41 innerTail10_35 best = best := by
  rw [innerTail10_35, scan, if_neg (by decide), innerCandidates10_35_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_36
    (if 3 ≤ innerCandidates10_35.length then search edge 3 [41, 5717] innerCandidates10_35 5758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_36

def innerTail10_34 : List Nat := 5477 :: innerTail10_35
def innerCandidates10_34 : List Nat := [6089]
set_option maxRecDepth 4096 in
theorem innerCandidates10_34_checked : candidates edge 4 41 5477 innerTail10_35 best.bound = innerCandidates10_34 := by
  unfold candidates
  have row : edge 5477 = SearchData.row5477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_34 : scan edge (search edge 3) 4 [41] 41 innerTail10_34 best = best := by
  rw [innerTail10_34, scan, if_neg (by decide), innerCandidates10_34_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_35
    (if 3 ≤ innerCandidates10_34.length then search edge 3 [41, 5477] innerCandidates10_34 5518 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_35

def innerTail10_33 : List Nat := 5147 :: innerTail10_34
def innerCandidates10_33 : List Nat := [7331]
set_option maxRecDepth 4096 in
theorem innerCandidates10_33_checked : candidates edge 4 41 5147 innerTail10_34 best.bound = innerCandidates10_33 := by
  unfold candidates
  have row : edge 5147 = SearchData.row5147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_33 : scan edge (search edge 3) 4 [41] 41 innerTail10_33 best = best := by
  rw [innerTail10_33, scan, if_neg (by decide), innerCandidates10_33_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_34
    (if 3 ≤ innerCandidates10_33.length then search edge 3 [41, 5147] innerCandidates10_33 5188 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_34

def innerTail10_32 : List Nat := 4691 :: innerTail10_33
def innerCandidates10_32 : List Nat := [7457]
set_option maxRecDepth 4096 in
theorem innerCandidates10_32_checked : candidates edge 4 41 4691 innerTail10_33 best.bound = innerCandidates10_32 := by
  unfold candidates
  have row : edge 4691 = SearchData.row4691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_32 : scan edge (search edge 3) 4 [41] 41 innerTail10_32 best = best := by
  rw [innerTail10_32, scan, if_neg (by decide), innerCandidates10_32_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_33
    (if 3 ≤ innerCandidates10_32.length then search edge 3 [41, 4691] innerCandidates10_32 4732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_33

def innerTail10_31 : List Nat := 4679 :: innerTail10_32
def innerCandidates10_31 : List Nat := [5717]
set_option maxRecDepth 4096 in
theorem innerCandidates10_31_checked : candidates edge 4 41 4679 innerTail10_32 best.bound = innerCandidates10_31 := by
  unfold candidates
  have row : edge 4679 = SearchData.row4679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_31 : scan edge (search edge 3) 4 [41] 41 innerTail10_31 best = best := by
  rw [innerTail10_31, scan, if_neg (by decide), innerCandidates10_31_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_32
    (if 3 ≤ innerCandidates10_31.length then search edge 3 [41, 4679] innerCandidates10_31 4720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_32

def innerTail10_30 : List Nat := 4457 :: innerTail10_31
def innerCandidates10_30 : List Nat := [4679, 6473, 6491, 7331]
set_option maxRecDepth 4096 in
theorem innerCandidates10_30_checked : candidates edge 4 41 4457 innerTail10_31 best.bound = innerCandidates10_30 := by
  unfold candidates
  have row : edge 4457 = SearchData.row4457.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild10_30 : search edge 3 [41, 4457] innerCandidates10_30 4498 best = best := by decide

theorem innerScan10_30 : scan edge (search edge 3) 4 [41] 41 innerTail10_30 best = best := by
  rw [innerTail10_30, scan, if_neg (by decide), innerCandidates10_30_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_31
    (if 3 ≤ innerCandidates10_30.length then search edge 3 [41, 4457] innerCandidates10_30 4498 best else best) = best
  rw [if_pos (by decide), innerChild10_30]
  exact innerScan10_31

def innerTail10_29 : List Nat := 4451 :: innerTail10_30
def innerCandidates10_29 : List Nat := [6089, 6833]
set_option maxRecDepth 4096 in
theorem innerCandidates10_29_checked : candidates edge 4 41 4451 innerTail10_30 best.bound = innerCandidates10_29 := by
  unfold candidates
  have row : edge 4451 = SearchData.row4451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_29 : scan edge (search edge 3) 4 [41] 41 innerTail10_29 best = best := by
  rw [innerTail10_29, scan, if_neg (by decide), innerCandidates10_29_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_30
    (if 3 ≤ innerCandidates10_29.length then search edge 3 [41, 4451] innerCandidates10_29 4492 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_30

def innerTail10_28 : List Nat := 4217 :: innerTail10_29
def innerCandidates10_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates10_28_checked : candidates edge 4 41 4217 innerTail10_29 best.bound = innerCandidates10_28 := by
  unfold candidates
  have row : edge 4217 = SearchData.row4217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_28 : scan edge (search edge 3) 4 [41] 41 innerTail10_28 best = best := by
  rw [innerTail10_28, scan, if_neg (by decide), innerCandidates10_28_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_29
    (if 3 ≤ innerCandidates10_28.length then search edge 3 [41, 4217] innerCandidates10_28 4258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_29

def innerTail10_27 : List Nat := 4049 :: innerTail10_28
def innerCandidates10_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates10_27_checked : candidates edge 4 41 4049 innerTail10_28 best.bound = innerCandidates10_27 := by
  unfold candidates
  have row : edge 4049 = SearchData.row4049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_27 : scan edge (search edge 3) 4 [41] 41 innerTail10_27 best = best := by
  rw [innerTail10_27, scan, if_neg (by decide), innerCandidates10_27_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_28
    (if 3 ≤ innerCandidates10_27.length then search edge 3 [41, 4049] innerCandidates10_27 4090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_28

def innerTail10_26 : List Nat := 4013 :: innerTail10_27
def innerCandidates10_26 : List Nat := [8069]
set_option maxRecDepth 4096 in
theorem innerCandidates10_26_checked : candidates edge 4 41 4013 innerTail10_27 best.bound = innerCandidates10_26 := by
  unfold candidates
  have row : edge 4013 = SearchData.row4013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_26 : scan edge (search edge 3) 4 [41] 41 innerTail10_26 best = best := by
  rw [innerTail10_26, scan, if_neg (by decide), innerCandidates10_26_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_27
    (if 3 ≤ innerCandidates10_26.length then search edge 3 [41, 4013] innerCandidates10_26 4054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_27

def innerTail10_25 : List Nat := 3719 :: innerTail10_26
def innerCandidates10_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates10_25_checked : candidates edge 4 41 3719 innerTail10_26 best.bound = innerCandidates10_25 := by
  unfold candidates
  have row : edge 3719 = SearchData.row3719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_25 : scan edge (search edge 3) 4 [41] 41 innerTail10_25 best = best := by
  rw [innerTail10_25, scan, if_neg (by decide), innerCandidates10_25_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_26
    (if 3 ≤ innerCandidates10_25.length then search edge 3 [41, 3719] innerCandidates10_25 3760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_26

def innerTail10_24 : List Nat := 3533 :: innerTail10_25
def innerCandidates10_24 : List Nat := [4679]
set_option maxRecDepth 4096 in
theorem innerCandidates10_24_checked : candidates edge 4 41 3533 innerTail10_25 best.bound = innerCandidates10_24 := by
  unfold candidates
  have row : edge 3533 = SearchData.row3533.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_24 : scan edge (search edge 3) 4 [41] 41 innerTail10_24 best = best := by
  rw [innerTail10_24, scan, if_neg (by decide), innerCandidates10_24_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_25
    (if 3 ≤ innerCandidates10_24.length then search edge 3 [41, 3533] innerCandidates10_24 3574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_25

def innerTail10_23 : List Nat := 3527 :: innerTail10_24
def innerCandidates10_23 : List Nat := [3533, 4457]
set_option maxRecDepth 4096 in
theorem innerCandidates10_23_checked : candidates edge 4 41 3527 innerTail10_24 best.bound = innerCandidates10_23 := by
  unfold candidates
  have row : edge 3527 = SearchData.row3527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_23 : scan edge (search edge 3) 4 [41] 41 innerTail10_23 best = best := by
  rw [innerTail10_23, scan, if_neg (by decide), innerCandidates10_23_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_24
    (if 3 ≤ innerCandidates10_23.length then search edge 3 [41, 3527] innerCandidates10_23 3568 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_24

def innerTail10_22 : List Nat := 3461 :: innerTail10_23
def innerCandidates10_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates10_22_checked : candidates edge 4 41 3461 innerTail10_23 best.bound = innerCandidates10_22 := by
  unfold candidates
  have row : edge 3461 = SearchData.row3461.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_22 : scan edge (search edge 3) 4 [41] 41 innerTail10_22 best = best := by
  rw [innerTail10_22, scan, if_neg (by decide), innerCandidates10_22_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_23
    (if 3 ≤ innerCandidates10_22.length then search edge 3 [41, 3461] innerCandidates10_22 3502 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_23

def innerTail10_21 : List Nat := 3299 :: innerTail10_22
def innerCandidates10_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates10_21_checked : candidates edge 4 41 3299 innerTail10_22 best.bound = innerCandidates10_21 := by
  unfold candidates
  have row : edge 3299 = SearchData.row3299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_21 : scan edge (search edge 3) 4 [41] 41 innerTail10_21 best = best := by
  rw [innerTail10_21, scan, if_neg (by decide), innerCandidates10_21_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_22
    (if 3 ≤ innerCandidates10_21.length then search edge 3 [41, 3299] innerCandidates10_21 3340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_22

def innerTail10_20 : List Nat := 2939 :: innerTail10_21
def innerCandidates10_20 : List Nat := [4217, 4679]
set_option maxRecDepth 4096 in
theorem innerCandidates10_20_checked : candidates edge 4 41 2939 innerTail10_21 best.bound = innerCandidates10_20 := by
  unfold candidates
  have row : edge 2939 = SearchData.row2939.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_20 : scan edge (search edge 3) 4 [41] 41 innerTail10_20 best = best := by
  rw [innerTail10_20, scan, if_neg (by decide), innerCandidates10_20_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_21
    (if 3 ≤ innerCandidates10_20.length then search edge 3 [41, 2939] innerCandidates10_20 2980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_21

def innerTail10_19 : List Nat := 2609 :: innerTail10_20
def innerCandidates10_19 : List Nat := [7451, 8069]
set_option maxRecDepth 4096 in
theorem innerCandidates10_19_checked : candidates edge 4 41 2609 innerTail10_20 best.bound = innerCandidates10_19 := by
  unfold candidates
  have row : edge 2609 = PrimePairCache.smallRow2609.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_19 : scan edge (search edge 3) 4 [41] 41 innerTail10_19 best = best := by
  rw [innerTail10_19, scan, if_neg (by decide), innerCandidates10_19_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_20
    (if 3 ≤ innerCandidates10_19.length then search edge 3 [41, 2609] innerCandidates10_19 2650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_20

def innerTail10_18 : List Nat := 2411 :: innerTail10_19
def innerCandidates10_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates10_18_checked : candidates edge 4 41 2411 innerTail10_19 best.bound = innerCandidates10_18 := by
  unfold candidates
  have row : edge 2411 = PrimePairCache.smallRow2411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_18 : scan edge (search edge 3) 4 [41] 41 innerTail10_18 best = best := by
  rw [innerTail10_18, scan, if_neg (by decide), innerCandidates10_18_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_19
    (if 3 ≤ innerCandidates10_18.length then search edge 3 [41, 2411] innerCandidates10_18 2452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_19

def innerTail10_17 : List Nat := 2339 :: innerTail10_18
def innerCandidates10_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates10_17_checked : candidates edge 4 41 2339 innerTail10_18 best.bound = innerCandidates10_17 := by
  unfold candidates
  have row : edge 2339 = PrimePairCache.smallRow2339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_17 : scan edge (search edge 3) 4 [41] 41 innerTail10_17 best = best := by
  rw [innerTail10_17, scan, if_neg (by decide), innerCandidates10_17_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_18
    (if 3 ≤ innerCandidates10_17.length then search edge 3 [41, 2339] innerCandidates10_17 2380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_18

def innerTail10_16 : List Nat := 2333 :: innerTail10_17
def innerCandidates10_16 : List Nat := [2339, 3533]
set_option maxRecDepth 4096 in
theorem innerCandidates10_16_checked : candidates edge 4 41 2333 innerTail10_17 best.bound = innerCandidates10_16 := by
  unfold candidates
  have row : edge 2333 = PrimePairCache.smallRow2333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_16 : scan edge (search edge 3) 4 [41] 41 innerTail10_16 best = best := by
  rw [innerTail10_16, scan, if_neg (by decide), innerCandidates10_16_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_17
    (if 3 ≤ innerCandidates10_16.length then search edge 3 [41, 2333] innerCandidates10_16 2374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_17

def innerTail10_15 : List Nat := 2081 :: innerTail10_16
def innerCandidates10_15 : List Nat := [4691, 7793]
set_option maxRecDepth 4096 in
theorem innerCandidates10_15_checked : candidates edge 4 41 2081 innerTail10_16 best.bound = innerCandidates10_15 := by
  unfold candidates
  have row : edge 2081 = PrimePairCache.smallRow2081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_15 : scan edge (search edge 3) 4 [41] 41 innerTail10_15 best = best := by
  rw [innerTail10_15, scan, if_neg (by decide), innerCandidates10_15_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_16
    (if 3 ≤ innerCandidates10_15.length then search edge 3 [41, 2081] innerCandidates10_15 2122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_16

def innerTail10_14 : List Nat := 1823 :: innerTail10_15
def innerCandidates10_14 : List Nat := [3461, 6719, 8069]
set_option maxRecDepth 4096 in
theorem innerCandidates10_14_checked : candidates edge 4 41 1823 innerTail10_15 best.bound = innerCandidates10_14 := by
  unfold candidates
  have row : edge 1823 = PrimePairCache.smallRow1823.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild10_14 : search edge 3 [41, 1823] innerCandidates10_14 1864 best = best := by decide

theorem innerScan10_14 : scan edge (search edge 3) 4 [41] 41 innerTail10_14 best = best := by
  rw [innerTail10_14, scan, if_neg (by decide), innerCandidates10_14_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_15
    (if 3 ≤ innerCandidates10_14.length then search edge 3 [41, 1823] innerCandidates10_14 1864 best else best) = best
  rw [if_pos (by decide), innerChild10_14]
  exact innerScan10_15

def innerTail10_13 : List Nat := 1667 :: innerTail10_14
def innerCandidates10_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates10_13_checked : candidates edge 4 41 1667 innerTail10_14 best.bound = innerCandidates10_13 := by
  unfold candidates
  have row : edge 1667 = PrimePairCache.smallRow1667.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_13 : scan edge (search edge 3) 4 [41] 41 innerTail10_13 best = best := by
  rw [innerTail10_13, scan, if_neg (by decide), innerCandidates10_13_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_14
    (if 3 ≤ innerCandidates10_13.length then search edge 3 [41, 1667] innerCandidates10_13 1708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_14

def innerTail10_12 : List Nat := 1637 :: innerTail10_13
def innerCandidates10_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates10_12_checked : candidates edge 4 41 1637 innerTail10_13 best.bound = innerCandidates10_12 := by
  unfold candidates
  have row : edge 1637 = PrimePairCache.smallRow1637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_12 : scan edge (search edge 3) 4 [41] 41 innerTail10_12 best = best := by
  rw [innerTail10_12, scan, if_neg (by decide), innerCandidates10_12_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_13
    (if 3 ≤ innerCandidates10_12.length then search edge 3 [41, 1637] innerCandidates10_12 1678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_13

def innerTail10_11 : List Nat := 1613 :: innerTail10_12
def innerCandidates10_11 : List Nat := [1637]
set_option maxRecDepth 4096 in
theorem innerCandidates10_11_checked : candidates edge 4 41 1613 innerTail10_12 best.bound = innerCandidates10_11 := by
  unfold candidates
  have row : edge 1613 = PrimePairCache.smallRow1613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_11 : scan edge (search edge 3) 4 [41] 41 innerTail10_11 best = best := by
  rw [innerTail10_11, scan, if_neg (by decide), innerCandidates10_11_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_12
    (if 3 ≤ innerCandidates10_11.length then search edge 3 [41, 1613] innerCandidates10_11 1654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_12

def innerTail10_10 : List Nat := 1583 :: innerTail10_11
def innerCandidates10_10 : List Nat := [1637, 4451, 7643]
set_option maxRecDepth 4096 in
theorem innerCandidates10_10_checked : candidates edge 4 41 1583 innerTail10_11 best.bound = innerCandidates10_10 := by
  unfold candidates
  have row : edge 1583 = PrimePairCache.smallRow1583.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild10_10 : search edge 3 [41, 1583] innerCandidates10_10 1624 best = best := by decide

theorem innerScan10_10 : scan edge (search edge 3) 4 [41] 41 innerTail10_10 best = best := by
  rw [innerTail10_10, scan, if_neg (by decide), innerCandidates10_10_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_11
    (if 3 ≤ innerCandidates10_10.length then search edge 3 [41, 1583] innerCandidates10_10 1624 best else best) = best
  rw [if_pos (by decide), innerChild10_10]
  exact innerScan10_11

def innerTail10_9 : List Nat := 1259 :: innerTail10_10
def innerCandidates10_9 : List Nat := [2609]
set_option maxRecDepth 4096 in
theorem innerCandidates10_9_checked : candidates edge 4 41 1259 innerTail10_10 best.bound = innerCandidates10_9 := by
  unfold candidates
  have row : edge 1259 = PrimePairCache.smallRow1259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_9 : scan edge (search edge 3) 4 [41] 41 innerTail10_9 best = best := by
  rw [innerTail10_9, scan, if_neg (by decide), innerCandidates10_9_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_10
    (if 3 ≤ innerCandidates10_9.length then search edge 3 [41, 1259] innerCandidates10_9 1300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_10

def innerTail10_8 : List Nat := 1031 :: innerTail10_9
def innerCandidates10_8 : List Nat := [3533]
set_option maxRecDepth 4096 in
theorem innerCandidates10_8_checked : candidates edge 4 41 1031 innerTail10_9 best.bound = innerCandidates10_8 := by
  unfold candidates
  have row : edge 1031 = PrimePairCache.smallRow1031.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_8 : scan edge (search edge 3) 4 [41] 41 innerTail10_8 best = best := by
  rw [innerTail10_8, scan, if_neg (by decide), innerCandidates10_8_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_9
    (if 3 ≤ innerCandidates10_8.length then search edge 3 [41, 1031] innerCandidates10_8 1072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_9

def innerTail10_7 : List Nat := 1013 :: innerTail10_8
def innerCandidates10_7 : List Nat := [6719, 7187, 7331]
set_option maxRecDepth 4096 in
theorem innerCandidates10_7_checked : candidates edge 4 41 1013 innerTail10_8 best.bound = innerCandidates10_7 := by
  unfold candidates
  have row : edge 1013 = PrimePairCache.smallRow1013.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild10_7 : search edge 3 [41, 1013] innerCandidates10_7 1054 best = best := by decide

theorem innerScan10_7 : scan edge (search edge 3) 4 [41] 41 innerTail10_7 best = best := by
  rw [innerTail10_7, scan, if_neg (by decide), innerCandidates10_7_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_8
    (if 3 ≤ innerCandidates10_7.length then search edge 3 [41, 1013] innerCandidates10_7 1054 best else best) = best
  rw [if_pos (by decide), innerChild10_7]
  exact innerScan10_8

def innerTail10_6 : List Nat := 911 :: innerTail10_7
def innerCandidates10_6 : List Nat := [6089]
set_option maxRecDepth 4096 in
theorem innerCandidates10_6_checked : candidates edge 4 41 911 innerTail10_7 best.bound = innerCandidates10_6 := by
  unfold candidates
  have row : edge 911 = PrimePairCache.smallRow911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_6 : scan edge (search edge 3) 4 [41] 41 innerTail10_6 best = best := by
  rw [innerTail10_6, scan, if_neg (by decide), innerCandidates10_6_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_7
    (if 3 ≤ innerCandidates10_6.length then search edge 3 [41, 911] innerCandidates10_6 952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_7

def innerTail10_5 : List Nat := 887 :: innerTail10_6
def innerCandidates10_5 : List Nat := [4049]
set_option maxRecDepth 4096 in
theorem innerCandidates10_5_checked : candidates edge 4 41 887 innerTail10_6 best.bound = innerCandidates10_5 := by
  unfold candidates
  have row : edge 887 = PrimePairCache.smallRow887.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_5 : scan edge (search edge 3) 4 [41] 41 innerTail10_5 best = best := by
  rw [innerTail10_5, scan, if_neg (by decide), innerCandidates10_5_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_6
    (if 3 ≤ innerCandidates10_5.length then search edge 3 [41, 887] innerCandidates10_5 928 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_6

def innerTail10_4 : List Nat := 863 :: innerTail10_5
def innerCandidates10_4 : List Nat := [1823, 5717]
set_option maxRecDepth 4096 in
theorem innerCandidates10_4_checked : candidates edge 4 41 863 innerTail10_5 best.bound = innerCandidates10_4 := by
  unfold candidates
  have row : edge 863 = PrimePairCache.smallRow863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_4 : scan edge (search edge 3) 4 [41] 41 innerTail10_4 best = best := by
  rw [innerTail10_4, scan, if_neg (by decide), innerCandidates10_4_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_5
    (if 3 ≤ innerCandidates10_4.length then search edge 3 [41, 863] innerCandidates10_4 904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_5

def innerTail10_3 : List Nat := 719 :: innerTail10_4
def innerCandidates10_3 : List Nat := [911, 2333, 7793]
set_option maxRecDepth 4096 in
theorem innerCandidates10_3_checked : candidates edge 4 41 719 innerTail10_4 best.bound = innerCandidates10_3 := by
  unfold candidates
  have row : edge 719 = PrimePairCache.smallRow719.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild10_3 : search edge 3 [41, 719] innerCandidates10_3 760 best = best := by decide

theorem innerScan10_3 : scan edge (search edge 3) 4 [41] 41 innerTail10_3 best = best := by
  rw [innerTail10_3, scan, if_neg (by decide), innerCandidates10_3_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_4
    (if 3 ≤ innerCandidates10_3.length then search edge 3 [41, 719] innerCandidates10_3 760 best else best) = best
  rw [if_pos (by decide), innerChild10_3]
  exact innerScan10_4

def innerTail10_2 : List Nat := 593 :: innerTail10_3
def innerCandidates10_2 : List Nat := [863, 2411, 3461]
set_option maxRecDepth 4096 in
theorem innerCandidates10_2_checked : candidates edge 4 41 593 innerTail10_3 best.bound = innerCandidates10_2 := by
  unfold candidates
  have row : edge 593 = PrimePairCache.smallRow593.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild10_2 : search edge 3 [41, 593] innerCandidates10_2 634 best = best := by decide

theorem innerScan10_2 : scan edge (search edge 3) 4 [41] 41 innerTail10_2 best = best := by
  rw [innerTail10_2, scan, if_neg (by decide), innerCandidates10_2_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_3
    (if 3 ≤ innerCandidates10_2.length then search edge 3 [41, 593] innerCandidates10_2 634 best else best) = best
  rw [if_pos (by decide), innerChild10_2]
  exact innerScan10_3

def innerTail10_1 : List Nat := 257 :: innerTail10_2
def innerCandidates10_1 : List Nat := [1823, 2939]
set_option maxRecDepth 4096 in
theorem innerCandidates10_1_checked : candidates edge 4 41 257 innerTail10_2 best.bound = innerCandidates10_1 := by
  unfold candidates
  have row : edge 257 = PrimePairCache.smallRow257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan10_1 : scan edge (search edge 3) 4 [41] 41 innerTail10_1 best = best := by
  rw [innerTail10_1, scan, if_neg (by decide), innerCandidates10_1_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_2
    (if 3 ≤ innerCandidates10_1.length then search edge 3 [41, 257] innerCandidates10_1 298 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan10_2

def innerTail10_0 : List Nat := 227 :: innerTail10_1
def innerCandidates10_0 : List Nat := [593, 719, 2939, 4691]
set_option maxRecDepth 4096 in
theorem innerCandidates10_0_checked : candidates edge 4 41 227 innerTail10_1 best.bound = innerCandidates10_0 := by
  unfold candidates
  have row : edge 227 = PrimePairCache.smallRow227.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild10_0 : search edge 3 [41, 227] innerCandidates10_0 268 best = best := by decide

theorem innerScan10_0 : scan edge (search edge 3) 4 [41] 41 innerTail10_0 best = best := by
  rw [innerTail10_0, scan, if_neg (by decide), innerCandidates10_0_checked]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_1
    (if 3 ≤ innerCandidates10_0.length then search edge 3 [41, 227] innerCandidates10_0 268 best else best) = best
  rw [if_pos (by decide), innerChild10_0]
  exact innerScan10_1

theorem rootBranch10 : search edge 4 [41] rootCandidates10 41 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [41] 41 innerTail10_0 best = best
  exact innerScan10_0

end ProjectEulerSolutions.PrimePairUpper.Search
