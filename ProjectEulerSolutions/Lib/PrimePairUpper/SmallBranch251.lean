import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates16 : List Nat := [139, 157, 547, 601, 619, 751, 757, 1039, 1063, 1093, 1201, 1303, 1609, 1831, 1933, 1999, 2029, 2251, 2341, 2377, 2557, 2767, 3313, 3499, 3637, 3943, 4057, 4099, 4483, 4789, 4831, 4903, 4987, 5179, 5419, 5569, 5581, 5743, 5827, 5881, 5923, 6337, 6661, 6829, 7561, 7639, 7681, 7687, 8191, 8221]
set_option maxRecDepth 4096 in
theorem rootCandidates16_checked : candidates edge 5 0 67 tail17 best.bound = rootCandidates16 := by
  unfold candidates
  have row : edge 67 = PrimePairCache.smallRow67.contains := by funext q; rfl
  rw [row]
  decide

def innerTail16_50 : List Nat := []
theorem innerScan16_50 : scan edge (search edge 3) 4 [67] 67 innerTail16_50 best = best := rfl

def innerTail16_49 : List Nat := 8221 :: innerTail16_50
def innerCandidates16_49 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates16_49_checked : candidates edge 4 67 8221 innerTail16_50 best.bound = innerCandidates16_49 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_49 : scan edge (search edge 3) 4 [67] 67 innerTail16_49 best = best := by
  rw [innerTail16_49, scan, if_neg (by decide), innerCandidates16_49_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_50
    (if 3 ≤ innerCandidates16_49.length then search edge 3 [67, 8221] innerCandidates16_49 8288 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_50

def innerTail16_48 : List Nat := 8191 :: innerTail16_49
def innerCandidates16_48 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates16_48_checked : candidates edge 4 67 8191 innerTail16_49 best.bound = innerCandidates16_48 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_48 : scan edge (search edge 3) 4 [67] 67 innerTail16_48 best = best := by
  rw [innerTail16_48, scan, if_neg (by decide), innerCandidates16_48_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_49
    (if 3 ≤ innerCandidates16_48.length then search edge 3 [67, 8191] innerCandidates16_48 8258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_49

def innerTail16_47 : List Nat := 7687 :: innerTail16_48
def innerCandidates16_47 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates16_47_checked : candidates edge 4 67 7687 innerTail16_48 best.bound = innerCandidates16_47 := by
  unfold candidates
  have row : edge 7687 = SearchData.row7687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_47 : scan edge (search edge 3) 4 [67] 67 innerTail16_47 best = best := by
  rw [innerTail16_47, scan, if_neg (by decide), innerCandidates16_47_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_48
    (if 3 ≤ innerCandidates16_47.length then search edge 3 [67, 7687] innerCandidates16_47 7754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_48

def innerTail16_46 : List Nat := 7681 :: innerTail16_47
def innerCandidates16_46 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates16_46_checked : candidates edge 4 67 7681 innerTail16_47 best.bound = innerCandidates16_46 := by
  unfold candidates
  have row : edge 7681 = SearchData.row7681.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_46 : scan edge (search edge 3) 4 [67] 67 innerTail16_46 best = best := by
  rw [innerTail16_46, scan, if_neg (by decide), innerCandidates16_46_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_47
    (if 3 ≤ innerCandidates16_46.length then search edge 3 [67, 7681] innerCandidates16_46 7748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_47

def innerTail16_45 : List Nat := 7639 :: innerTail16_46
def innerCandidates16_45 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates16_45_checked : candidates edge 4 67 7639 innerTail16_46 best.bound = innerCandidates16_45 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_45 : scan edge (search edge 3) 4 [67] 67 innerTail16_45 best = best := by
  rw [innerTail16_45, scan, if_neg (by decide), innerCandidates16_45_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_46
    (if 3 ≤ innerCandidates16_45.length then search edge 3 [67, 7639] innerCandidates16_45 7706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_46

def innerTail16_44 : List Nat := 7561 :: innerTail16_45
def innerCandidates16_44 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates16_44_checked : candidates edge 4 67 7561 innerTail16_45 best.bound = innerCandidates16_44 := by
  unfold candidates
  have row : edge 7561 = SearchData.row7561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_44 : scan edge (search edge 3) 4 [67] 67 innerTail16_44 best = best := by
  rw [innerTail16_44, scan, if_neg (by decide), innerCandidates16_44_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_45
    (if 3 ≤ innerCandidates16_44.length then search edge 3 [67, 7561] innerCandidates16_44 7628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_45

def innerTail16_43 : List Nat := 6829 :: innerTail16_44
def innerCandidates16_43 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates16_43_checked : candidates edge 4 67 6829 innerTail16_44 best.bound = innerCandidates16_43 := by
  unfold candidates
  have row : edge 6829 = SearchData.row6829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_43 : scan edge (search edge 3) 4 [67] 67 innerTail16_43 best = best := by
  rw [innerTail16_43, scan, if_neg (by decide), innerCandidates16_43_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_44
    (if 3 ≤ innerCandidates16_43.length then search edge 3 [67, 6829] innerCandidates16_43 6896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_44

def innerTail16_42 : List Nat := 6661 :: innerTail16_43
def innerCandidates16_42 : List Nat := [7687]
set_option maxRecDepth 4096 in
theorem innerCandidates16_42_checked : candidates edge 4 67 6661 innerTail16_43 best.bound = innerCandidates16_42 := by
  unfold candidates
  have row : edge 6661 = SearchData.row6661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_42 : scan edge (search edge 3) 4 [67] 67 innerTail16_42 best = best := by
  rw [innerTail16_42, scan, if_neg (by decide), innerCandidates16_42_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_43
    (if 3 ≤ innerCandidates16_42.length then search edge 3 [67, 6661] innerCandidates16_42 6728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_43

def innerTail16_41 : List Nat := 6337 :: innerTail16_42
def innerCandidates16_41 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates16_41_checked : candidates edge 4 67 6337 innerTail16_42 best.bound = innerCandidates16_41 := by
  unfold candidates
  have row : edge 6337 = SearchData.row6337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_41 : scan edge (search edge 3) 4 [67] 67 innerTail16_41 best = best := by
  rw [innerTail16_41, scan, if_neg (by decide), innerCandidates16_41_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_42
    (if 3 ≤ innerCandidates16_41.length then search edge 3 [67, 6337] innerCandidates16_41 6404 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_42

def innerTail16_40 : List Nat := 5923 :: innerTail16_41
def innerCandidates16_40 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates16_40_checked : candidates edge 4 67 5923 innerTail16_41 best.bound = innerCandidates16_40 := by
  unfold candidates
  have row : edge 5923 = SearchData.row5923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_40 : scan edge (search edge 3) 4 [67] 67 innerTail16_40 best = best := by
  rw [innerTail16_40, scan, if_neg (by decide), innerCandidates16_40_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_41
    (if 3 ≤ innerCandidates16_40.length then search edge 3 [67, 5923] innerCandidates16_40 5990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_41

def innerTail16_39 : List Nat := 5881 :: innerTail16_40
def innerCandidates16_39 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates16_39_checked : candidates edge 4 67 5881 innerTail16_40 best.bound = innerCandidates16_39 := by
  unfold candidates
  have row : edge 5881 = SearchData.row5881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_39 : scan edge (search edge 3) 4 [67] 67 innerTail16_39 best = best := by
  rw [innerTail16_39, scan, if_neg (by decide), innerCandidates16_39_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_40
    (if 3 ≤ innerCandidates16_39.length then search edge 3 [67, 5881] innerCandidates16_39 5948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_40

def innerTail16_38 : List Nat := 5827 :: innerTail16_39
def innerCandidates16_38 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates16_38_checked : candidates edge 4 67 5827 innerTail16_39 best.bound = innerCandidates16_38 := by
  unfold candidates
  have row : edge 5827 = SearchData.row5827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_38 : scan edge (search edge 3) 4 [67] 67 innerTail16_38 best = best := by
  rw [innerTail16_38, scan, if_neg (by decide), innerCandidates16_38_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_39
    (if 3 ≤ innerCandidates16_38.length then search edge 3 [67, 5827] innerCandidates16_38 5894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_39

def innerTail16_37 : List Nat := 5743 :: innerTail16_38
def innerCandidates16_37 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates16_37_checked : candidates edge 4 67 5743 innerTail16_38 best.bound = innerCandidates16_37 := by
  unfold candidates
  have row : edge 5743 = SearchData.row5743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_37 : scan edge (search edge 3) 4 [67] 67 innerTail16_37 best = best := by
  rw [innerTail16_37, scan, if_neg (by decide), innerCandidates16_37_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_38
    (if 3 ≤ innerCandidates16_37.length then search edge 3 [67, 5743] innerCandidates16_37 5810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_38

def innerTail16_36 : List Nat := 5581 :: innerTail16_37
def innerCandidates16_36 : List Nat := [7561]
set_option maxRecDepth 4096 in
theorem innerCandidates16_36_checked : candidates edge 4 67 5581 innerTail16_37 best.bound = innerCandidates16_36 := by
  unfold candidates
  have row : edge 5581 = SearchData.row5581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_36 : scan edge (search edge 3) 4 [67] 67 innerTail16_36 best = best := by
  rw [innerTail16_36, scan, if_neg (by decide), innerCandidates16_36_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_37
    (if 3 ≤ innerCandidates16_36.length then search edge 3 [67, 5581] innerCandidates16_36 5648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_37

def innerTail16_35 : List Nat := 5569 :: innerTail16_36
def innerCandidates16_35 : List Nat := [5923, 7687]
set_option maxRecDepth 4096 in
theorem innerCandidates16_35_checked : candidates edge 4 67 5569 innerTail16_36 best.bound = innerCandidates16_35 := by
  unfold candidates
  have row : edge 5569 = SearchData.row5569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_35 : scan edge (search edge 3) 4 [67] 67 innerTail16_35 best = best := by
  rw [innerTail16_35, scan, if_neg (by decide), innerCandidates16_35_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_36
    (if 3 ≤ innerCandidates16_35.length then search edge 3 [67, 5569] innerCandidates16_35 5636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_36

def innerTail16_34 : List Nat := 5419 :: innerTail16_35
def innerCandidates16_34 : List Nat := [7687]
set_option maxRecDepth 4096 in
theorem innerCandidates16_34_checked : candidates edge 4 67 5419 innerTail16_35 best.bound = innerCandidates16_34 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_34 : scan edge (search edge 3) 4 [67] 67 innerTail16_34 best = best := by
  rw [innerTail16_34, scan, if_neg (by decide), innerCandidates16_34_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_35
    (if 3 ≤ innerCandidates16_34.length then search edge 3 [67, 5419] innerCandidates16_34 5486 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_35

def innerTail16_33 : List Nat := 5179 :: innerTail16_34
def innerCandidates16_33 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates16_33_checked : candidates edge 4 67 5179 innerTail16_34 best.bound = innerCandidates16_33 := by
  unfold candidates
  have row : edge 5179 = SearchData.row5179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_33 : scan edge (search edge 3) 4 [67] 67 innerTail16_33 best = best := by
  rw [innerTail16_33, scan, if_neg (by decide), innerCandidates16_33_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_34
    (if 3 ≤ innerCandidates16_33.length then search edge 3 [67, 5179] innerCandidates16_33 5246 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_34

def innerTail16_32 : List Nat := 4987 :: innerTail16_33
def innerCandidates16_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates16_32_checked : candidates edge 4 67 4987 innerTail16_33 best.bound = innerCandidates16_32 := by
  unfold candidates
  have row : edge 4987 = SearchData.row4987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_32 : scan edge (search edge 3) 4 [67] 67 innerTail16_32 best = best := by
  rw [innerTail16_32, scan, if_neg (by decide), innerCandidates16_32_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_33
    (if 3 ≤ innerCandidates16_32.length then search edge 3 [67, 4987] innerCandidates16_32 5054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_33

def innerTail16_31 : List Nat := 4903 :: innerTail16_32
def innerCandidates16_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates16_31_checked : candidates edge 4 67 4903 innerTail16_32 best.bound = innerCandidates16_31 := by
  unfold candidates
  have row : edge 4903 = SearchData.row4903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_31 : scan edge (search edge 3) 4 [67] 67 innerTail16_31 best = best := by
  rw [innerTail16_31, scan, if_neg (by decide), innerCandidates16_31_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_32
    (if 3 ≤ innerCandidates16_31.length then search edge 3 [67, 4903] innerCandidates16_31 4970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_32

def innerTail16_30 : List Nat := 4831 :: innerTail16_31
def innerCandidates16_30 : List Nat := [8191]
set_option maxRecDepth 4096 in
theorem innerCandidates16_30_checked : candidates edge 4 67 4831 innerTail16_31 best.bound = innerCandidates16_30 := by
  unfold candidates
  have row : edge 4831 = SearchData.row4831.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_30 : scan edge (search edge 3) 4 [67] 67 innerTail16_30 best = best := by
  rw [innerTail16_30, scan, if_neg (by decide), innerCandidates16_30_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_31
    (if 3 ≤ innerCandidates16_30.length then search edge 3 [67, 4831] innerCandidates16_30 4898 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_31

def innerTail16_29 : List Nat := 4789 :: innerTail16_30
def innerCandidates16_29 : List Nat := [5179, 7561]
set_option maxRecDepth 4096 in
theorem innerCandidates16_29_checked : candidates edge 4 67 4789 innerTail16_30 best.bound = innerCandidates16_29 := by
  unfold candidates
  have row : edge 4789 = SearchData.row4789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_29 : scan edge (search edge 3) 4 [67] 67 innerTail16_29 best = best := by
  rw [innerTail16_29, scan, if_neg (by decide), innerCandidates16_29_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_30
    (if 3 ≤ innerCandidates16_29.length then search edge 3 [67, 4789] innerCandidates16_29 4856 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_30

def innerTail16_28 : List Nat := 4483 :: innerTail16_29
def innerCandidates16_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates16_28_checked : candidates edge 4 67 4483 innerTail16_29 best.bound = innerCandidates16_28 := by
  unfold candidates
  have row : edge 4483 = SearchData.row4483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_28 : scan edge (search edge 3) 4 [67] 67 innerTail16_28 best = best := by
  rw [innerTail16_28, scan, if_neg (by decide), innerCandidates16_28_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_29
    (if 3 ≤ innerCandidates16_28.length then search edge 3 [67, 4483] innerCandidates16_28 4550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_29

def innerTail16_27 : List Nat := 4099 :: innerTail16_28
def innerCandidates16_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates16_27_checked : candidates edge 4 67 4099 innerTail16_28 best.bound = innerCandidates16_27 := by
  unfold candidates
  have row : edge 4099 = SearchData.row4099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_27 : scan edge (search edge 3) 4 [67] 67 innerTail16_27 best = best := by
  rw [innerTail16_27, scan, if_neg (by decide), innerCandidates16_27_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_28
    (if 3 ≤ innerCandidates16_27.length then search edge 3 [67, 4099] innerCandidates16_27 4166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_28

def innerTail16_26 : List Nat := 4057 :: innerTail16_27
def innerCandidates16_26 : List Nat := [4483, 5419]
set_option maxRecDepth 4096 in
theorem innerCandidates16_26_checked : candidates edge 4 67 4057 innerTail16_27 best.bound = innerCandidates16_26 := by
  unfold candidates
  have row : edge 4057 = SearchData.row4057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_26 : scan edge (search edge 3) 4 [67] 67 innerTail16_26 best = best := by
  rw [innerTail16_26, scan, if_neg (by decide), innerCandidates16_26_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_27
    (if 3 ≤ innerCandidates16_26.length then search edge 3 [67, 4057] innerCandidates16_26 4124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_27

def innerTail16_25 : List Nat := 3943 :: innerTail16_26
def innerCandidates16_25 : List Nat := [5569, 5581, 6829]
set_option maxRecDepth 4096 in
theorem innerCandidates16_25_checked : candidates edge 4 67 3943 innerTail16_26 best.bound = innerCandidates16_25 := by
  unfold candidates
  have row : edge 3943 = SearchData.row3943.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild16_25 : search edge 3 [67, 3943] innerCandidates16_25 4010 best = best := by decide

theorem innerScan16_25 : scan edge (search edge 3) 4 [67] 67 innerTail16_25 best = best := by
  rw [innerTail16_25, scan, if_neg (by decide), innerCandidates16_25_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_26
    (if 3 ≤ innerCandidates16_25.length then search edge 3 [67, 3943] innerCandidates16_25 4010 best else best) = best
  rw [if_pos (by decide), innerChild16_25]
  exact innerScan16_26

def innerTail16_24 : List Nat := 3637 :: innerTail16_25
def innerCandidates16_24 : List Nat := [7639]
set_option maxRecDepth 4096 in
theorem innerCandidates16_24_checked : candidates edge 4 67 3637 innerTail16_25 best.bound = innerCandidates16_24 := by
  unfold candidates
  have row : edge 3637 = SearchData.row3637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_24 : scan edge (search edge 3) 4 [67] 67 innerTail16_24 best = best := by
  rw [innerTail16_24, scan, if_neg (by decide), innerCandidates16_24_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_25
    (if 3 ≤ innerCandidates16_24.length then search edge 3 [67, 3637] innerCandidates16_24 3704 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_25

def innerTail16_23 : List Nat := 3499 :: innerTail16_24
def innerCandidates16_23 : List Nat := [5881, 8191]
set_option maxRecDepth 4096 in
theorem innerCandidates16_23_checked : candidates edge 4 67 3499 innerTail16_24 best.bound = innerCandidates16_23 := by
  unfold candidates
  have row : edge 3499 = SearchData.row3499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_23 : scan edge (search edge 3) 4 [67] 67 innerTail16_23 best = best := by
  rw [innerTail16_23, scan, if_neg (by decide), innerCandidates16_23_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_24
    (if 3 ≤ innerCandidates16_23.length then search edge 3 [67, 3499] innerCandidates16_23 3566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_24

def innerTail16_22 : List Nat := 3313 :: innerTail16_23
def innerCandidates16_22 : List Nat := [4831]
set_option maxRecDepth 4096 in
theorem innerCandidates16_22_checked : candidates edge 4 67 3313 innerTail16_23 best.bound = innerCandidates16_22 := by
  unfold candidates
  have row : edge 3313 = SearchData.row3313.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_22 : scan edge (search edge 3) 4 [67] 67 innerTail16_22 best = best := by
  rw [innerTail16_22, scan, if_neg (by decide), innerCandidates16_22_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_23
    (if 3 ≤ innerCandidates16_22.length then search edge 3 [67, 3313] innerCandidates16_22 3380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_23

def innerTail16_21 : List Nat := 2767 :: innerTail16_22
def innerCandidates16_21 : List Nat := [4099, 7561]
set_option maxRecDepth 4096 in
theorem innerCandidates16_21_checked : candidates edge 4 67 2767 innerTail16_22 best.bound = innerCandidates16_21 := by
  unfold candidates
  have row : edge 2767 = SearchData.row2767.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_21 : scan edge (search edge 3) 4 [67] 67 innerTail16_21 best = best := by
  rw [innerTail16_21, scan, if_neg (by decide), innerCandidates16_21_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_22
    (if 3 ≤ innerCandidates16_21.length then search edge 3 [67, 2767] innerCandidates16_21 2834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_22

def innerTail16_20 : List Nat := 2557 :: innerTail16_21
def innerCandidates16_20 : List Nat := [3313, 7687, 8191]
set_option maxRecDepth 4096 in
theorem innerCandidates16_20_checked : candidates edge 4 67 2557 innerTail16_21 best.bound = innerCandidates16_20 := by
  unfold candidates
  have row : edge 2557 = PrimePairCache.smallRow2557.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild16_20 : search edge 3 [67, 2557] innerCandidates16_20 2624 best = best := by decide

theorem innerScan16_20 : scan edge (search edge 3) 4 [67] 67 innerTail16_20 best = best := by
  rw [innerTail16_20, scan, if_neg (by decide), innerCandidates16_20_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_21
    (if 3 ≤ innerCandidates16_20.length then search edge 3 [67, 2557] innerCandidates16_20 2624 best else best) = best
  rw [if_pos (by decide), innerChild16_20]
  exact innerScan16_21

def innerTail16_19 : List Nat := 2377 :: innerTail16_20
def innerCandidates16_19 : List Nat := [2557, 5743]
set_option maxRecDepth 4096 in
theorem innerCandidates16_19_checked : candidates edge 4 67 2377 innerTail16_20 best.bound = innerCandidates16_19 := by
  unfold candidates
  have row : edge 2377 = PrimePairCache.smallRow2377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_19 : scan edge (search edge 3) 4 [67] 67 innerTail16_19 best = best := by
  rw [innerTail16_19, scan, if_neg (by decide), innerCandidates16_19_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_20
    (if 3 ≤ innerCandidates16_19.length then search edge 3 [67, 2377] innerCandidates16_19 2444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_20

def innerTail16_18 : List Nat := 2341 :: innerTail16_19
def innerCandidates16_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates16_18_checked : candidates edge 4 67 2341 innerTail16_19 best.bound = innerCandidates16_18 := by
  unfold candidates
  have row : edge 2341 = PrimePairCache.smallRow2341.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_18 : scan edge (search edge 3) 4 [67] 67 innerTail16_18 best = best := by
  rw [innerTail16_18, scan, if_neg (by decide), innerCandidates16_18_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_19
    (if 3 ≤ innerCandidates16_18.length then search edge 3 [67, 2341] innerCandidates16_18 2408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_19

def innerTail16_17 : List Nat := 2251 :: innerTail16_18
def innerCandidates16_17 : List Nat := [5881, 8191]
set_option maxRecDepth 4096 in
theorem innerCandidates16_17_checked : candidates edge 4 67 2251 innerTail16_18 best.bound = innerCandidates16_17 := by
  unfold candidates
  have row : edge 2251 = PrimePairCache.smallRow2251.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_17 : scan edge (search edge 3) 4 [67] 67 innerTail16_17 best = best := by
  rw [innerTail16_17, scan, if_neg (by decide), innerCandidates16_17_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_18
    (if 3 ≤ innerCandidates16_17.length then search edge 3 [67, 2251] innerCandidates16_17 2318 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_18

def innerTail16_16 : List Nat := 2029 :: innerTail16_17
def innerCandidates16_16 : List Nat := [2341, 5419]
set_option maxRecDepth 4096 in
theorem innerCandidates16_16_checked : candidates edge 4 67 2029 innerTail16_17 best.bound = innerCandidates16_16 := by
  unfold candidates
  have row : edge 2029 = PrimePairCache.smallRow2029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_16 : scan edge (search edge 3) 4 [67] 67 innerTail16_16 best = best := by
  rw [innerTail16_16, scan, if_neg (by decide), innerCandidates16_16_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_17
    (if 3 ≤ innerCandidates16_16.length then search edge 3 [67, 2029] innerCandidates16_16 2096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_17

def innerTail16_15 : List Nat := 1999 :: innerTail16_16
def innerCandidates16_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates16_15_checked : candidates edge 4 67 1999 innerTail16_16 best.bound = innerCandidates16_15 := by
  unfold candidates
  have row : edge 1999 = PrimePairCache.smallRow1999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_15 : scan edge (search edge 3) 4 [67] 67 innerTail16_15 best = best := by
  rw [innerTail16_15, scan, if_neg (by decide), innerCandidates16_15_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_16
    (if 3 ≤ innerCandidates16_15.length then search edge 3 [67, 1999] innerCandidates16_15 2066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_16

def innerTail16_14 : List Nat := 1933 :: innerTail16_15
def innerCandidates16_14 : List Nat := [3313, 7639]
set_option maxRecDepth 4096 in
theorem innerCandidates16_14_checked : candidates edge 4 67 1933 innerTail16_15 best.bound = innerCandidates16_14 := by
  unfold candidates
  have row : edge 1933 = PrimePairCache.smallRow1933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_14 : scan edge (search edge 3) 4 [67] 67 innerTail16_14 best = best := by
  rw [innerTail16_14, scan, if_neg (by decide), innerCandidates16_14_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_15
    (if 3 ≤ innerCandidates16_14.length then search edge 3 [67, 1933] innerCandidates16_14 2000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_15

def innerTail16_13 : List Nat := 1831 :: innerTail16_14
def innerCandidates16_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates16_13_checked : candidates edge 4 67 1831 innerTail16_14 best.bound = innerCandidates16_13 := by
  unfold candidates
  have row : edge 1831 = PrimePairCache.smallRow1831.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_13 : scan edge (search edge 3) 4 [67] 67 innerTail16_13 best = best := by
  rw [innerTail16_13, scan, if_neg (by decide), innerCandidates16_13_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_14
    (if 3 ≤ innerCandidates16_13.length then search edge 3 [67, 1831] innerCandidates16_13 1898 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_14

def innerTail16_12 : List Nat := 1609 :: innerTail16_13
def innerCandidates16_12 : List Nat := [5179]
set_option maxRecDepth 4096 in
theorem innerCandidates16_12_checked : candidates edge 4 67 1609 innerTail16_13 best.bound = innerCandidates16_12 := by
  unfold candidates
  have row : edge 1609 = PrimePairCache.smallRow1609.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_12 : scan edge (search edge 3) 4 [67] 67 innerTail16_12 best = best := by
  rw [innerTail16_12, scan, if_neg (by decide), innerCandidates16_12_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_13
    (if 3 ≤ innerCandidates16_12.length then search edge 3 [67, 1609] innerCandidates16_12 1676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_13

def innerTail16_11 : List Nat := 1303 :: innerTail16_12
def innerCandidates16_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates16_11_checked : candidates edge 4 67 1303 innerTail16_12 best.bound = innerCandidates16_11 := by
  unfold candidates
  have row : edge 1303 = PrimePairCache.smallRow1303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_11 : scan edge (search edge 3) 4 [67] 67 innerTail16_11 best = best := by
  rw [innerTail16_11, scan, if_neg (by decide), innerCandidates16_11_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_12
    (if 3 ≤ innerCandidates16_11.length then search edge 3 [67, 1303] innerCandidates16_11 1370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_12

def innerTail16_10 : List Nat := 1201 :: innerTail16_11
def innerCandidates16_10 : List Nat := [1303, 3637, 6337]
set_option maxRecDepth 4096 in
theorem innerCandidates16_10_checked : candidates edge 4 67 1201 innerTail16_11 best.bound = innerCandidates16_10 := by
  unfold candidates
  have row : edge 1201 = PrimePairCache.smallRow1201.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild16_10 : search edge 3 [67, 1201] innerCandidates16_10 1268 best = best := by decide

theorem innerScan16_10 : scan edge (search edge 3) 4 [67] 67 innerTail16_10 best = best := by
  rw [innerTail16_10, scan, if_neg (by decide), innerCandidates16_10_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_11
    (if 3 ≤ innerCandidates16_10.length then search edge 3 [67, 1201] innerCandidates16_10 1268 best else best) = best
  rw [if_pos (by decide), innerChild16_10]
  exact innerScan16_11

def innerTail16_9 : List Nat := 1093 :: innerTail16_10
def innerCandidates16_9 : List Nat := [5569]
set_option maxRecDepth 4096 in
theorem innerCandidates16_9_checked : candidates edge 4 67 1093 innerTail16_10 best.bound = innerCandidates16_9 := by
  unfold candidates
  have row : edge 1093 = PrimePairCache.smallRow1093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_9 : scan edge (search edge 3) 4 [67] 67 innerTail16_9 best = best := by
  rw [innerTail16_9, scan, if_neg (by decide), innerCandidates16_9_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_10
    (if 3 ≤ innerCandidates16_9.length then search edge 3 [67, 1093] innerCandidates16_9 1160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_10

def innerTail16_8 : List Nat := 1063 :: innerTail16_9
def innerCandidates16_8 : List Nat := [1609, 3499]
set_option maxRecDepth 4096 in
theorem innerCandidates16_8_checked : candidates edge 4 67 1063 innerTail16_9 best.bound = innerCandidates16_8 := by
  unfold candidates
  have row : edge 1063 = PrimePairCache.smallRow1063.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_8 : scan edge (search edge 3) 4 [67] 67 innerTail16_8 best = best := by
  rw [innerTail16_8, scan, if_neg (by decide), innerCandidates16_8_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_9
    (if 3 ≤ innerCandidates16_8.length then search edge 3 [67, 1063] innerCandidates16_8 1130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_9

def innerTail16_7 : List Nat := 1039 :: innerTail16_8
def innerCandidates16_7 : List Nat := [1999, 3637]
set_option maxRecDepth 4096 in
theorem innerCandidates16_7_checked : candidates edge 4 67 1039 innerTail16_8 best.bound = innerCandidates16_7 := by
  unfold candidates
  have row : edge 1039 = PrimePairCache.smallRow1039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_7 : scan edge (search edge 3) 4 [67] 67 innerTail16_7 best = best := by
  rw [innerTail16_7, scan, if_neg (by decide), innerCandidates16_7_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_8
    (if 3 ≤ innerCandidates16_7.length then search edge 3 [67, 1039] innerCandidates16_7 1106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_8

def innerTail16_6 : List Nat := 757 :: innerTail16_7
def innerCandidates16_6 : List Nat := [5569, 6661]
set_option maxRecDepth 4096 in
theorem innerCandidates16_6_checked : candidates edge 4 67 757 innerTail16_7 best.bound = innerCandidates16_6 := by
  unfold candidates
  have row : edge 757 = PrimePairCache.smallRow757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_6 : scan edge (search edge 3) 4 [67] 67 innerTail16_6 best = best := by
  rw [innerTail16_6, scan, if_neg (by decide), innerCandidates16_6_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_7
    (if 3 ≤ innerCandidates16_6.length then search edge 3 [67, 757] innerCandidates16_6 824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_7

def innerTail16_5 : List Nat := 751 :: innerTail16_6
def innerCandidates16_5 : List Nat := [2377]
set_option maxRecDepth 4096 in
theorem innerCandidates16_5_checked : candidates edge 4 67 751 innerTail16_6 best.bound = innerCandidates16_5 := by
  unfold candidates
  have row : edge 751 = PrimePairCache.smallRow751.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_5 : scan edge (search edge 3) 4 [67] 67 innerTail16_5 best = best := by
  rw [innerTail16_5, scan, if_neg (by decide), innerCandidates16_5_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_6
    (if 3 ≤ innerCandidates16_5.length then search edge 3 [67, 751] innerCandidates16_5 818 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_6

def innerTail16_4 : List Nat := 619 :: innerTail16_5
def innerCandidates16_4 : List Nat := [5419, 7687]
set_option maxRecDepth 4096 in
theorem innerCandidates16_4_checked : candidates edge 4 67 619 innerTail16_5 best.bound = innerCandidates16_4 := by
  unfold candidates
  have row : edge 619 = PrimePairCache.smallRow619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_4 : scan edge (search edge 3) 4 [67] 67 innerTail16_4 best = best := by
  rw [innerTail16_4, scan, if_neg (by decide), innerCandidates16_4_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_5
    (if 3 ≤ innerCandidates16_4.length then search edge 3 [67, 619] innerCandidates16_4 686 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_5

def innerTail16_3 : List Nat := 601 :: innerTail16_4
def innerCandidates16_3 : List Nat := [7639]
set_option maxRecDepth 4096 in
theorem innerCandidates16_3_checked : candidates edge 4 67 601 innerTail16_4 best.bound = innerCandidates16_3 := by
  unfold candidates
  have row : edge 601 = PrimePairCache.smallRow601.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_3 : scan edge (search edge 3) 4 [67] 67 innerTail16_3 best = best := by
  rw [innerTail16_3, scan, if_neg (by decide), innerCandidates16_3_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_4
    (if 3 ≤ innerCandidates16_3.length then search edge 3 [67, 601] innerCandidates16_3 668 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_4

def innerTail16_2 : List Nat := 547 :: innerTail16_3
def innerCandidates16_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates16_2_checked : candidates edge 4 67 547 innerTail16_3 best.bound = innerCandidates16_2 := by
  unfold candidates
  have row : edge 547 = PrimePairCache.smallRow547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan16_2 : scan edge (search edge 3) 4 [67] 67 innerTail16_2 best = best := by
  rw [innerTail16_2, scan, if_neg (by decide), innerCandidates16_2_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_3
    (if 3 ≤ innerCandidates16_2.length then search edge 3 [67, 547] innerCandidates16_2 614 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan16_3

def innerTail16_1 : List Nat := 157 :: innerTail16_2
def innerCandidates16_1 : List Nat := [2377, 6661, 7561]
set_option maxRecDepth 4096 in
theorem innerCandidates16_1_checked : candidates edge 4 67 157 innerTail16_2 best.bound = innerCandidates16_1 := by
  unfold candidates
  have row : edge 157 = PrimePairCache.smallRow157.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild16_1 : search edge 3 [67, 157] innerCandidates16_1 224 best = best := by decide

theorem innerScan16_1 : scan edge (search edge 3) 4 [67] 67 innerTail16_1 best = best := by
  rw [innerTail16_1, scan, if_neg (by decide), innerCandidates16_1_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_2
    (if 3 ≤ innerCandidates16_1.length then search edge 3 [67, 157] innerCandidates16_1 224 best else best) = best
  rw [if_pos (by decide), innerChild16_1]
  exact innerScan16_2

def innerTail16_0 : List Nat := 139 :: innerTail16_1
def innerCandidates16_0 : List Nat := [547, 619, 3313, 5743]
set_option maxRecDepth 4096 in
theorem innerCandidates16_0_checked : candidates edge 4 67 139 innerTail16_1 best.bound = innerCandidates16_0 := by
  unfold candidates
  have row : edge 139 = PrimePairCache.smallRow139.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild16_0 : search edge 3 [67, 139] innerCandidates16_0 206 best = best := by decide

theorem innerScan16_0 : scan edge (search edge 3) 4 [67] 67 innerTail16_0 best = best := by
  rw [innerTail16_0, scan, if_neg (by decide), innerCandidates16_0_checked]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_1
    (if 3 ≤ innerCandidates16_0.length then search edge 3 [67, 139] innerCandidates16_0 206 best else best) = best
  rw [if_pos (by decide), innerChild16_0]
  exact innerScan16_1

theorem rootBranch16 : search edge 4 [67] rootCandidates16 67 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [67] 67 innerTail16_0 best = best
  exact innerScan16_0

end ProjectEulerSolutions.PrimePairUpper.Search
