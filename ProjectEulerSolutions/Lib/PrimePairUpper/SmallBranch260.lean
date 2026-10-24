import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates7 : List Nat := [71, 137, 167, 179, 347, 383, 401, 569, 599, 683, 761, 881, 1901, 1997, 2027, 2081, 2141, 2309, 2393, 2441, 2477, 2549, 2711, 2801, 2969, 3257, 3659, 3767, 3803, 3863, 4013, 4157, 4241, 4337, 4397, 4523, 4673, 4787, 5039, 5081, 5237, 5441, 5861, 6047, 6221, 6269, 6299, 6563, 6827, 6911, 6983, 7079, 7247, 7487, 7589, 7607, 7757, 7793, 7841, 7853, 8237]
set_option maxRecDepth 4096 in
theorem rootCandidates7_checked : candidates edge 5 0 29 tail8 best.bound = rootCandidates7 := by
  unfold candidates
  have row : edge 29 = PrimePairCache.smallRow29.contains := by funext q; rfl
  rw [row]
  decide

def innerTail7_61 : List Nat := []
theorem innerScan7_61 : scan edge (search edge 3) 4 [29] 29 innerTail7_61 best = best := rfl

def innerTail7_60 : List Nat := 8237 :: innerTail7_61
def innerCandidates7_60 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates7_60_checked : candidates edge 4 29 8237 innerTail7_61 best.bound = innerCandidates7_60 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_60 : scan edge (search edge 3) 4 [29] 29 innerTail7_60 best = best := by
  rw [innerTail7_60, scan, if_neg (by decide), innerCandidates7_60_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_61
    (if 3 ≤ innerCandidates7_60.length then search edge 3 [29, 8237] innerCandidates7_60 8266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_61

def innerTail7_59 : List Nat := 7853 :: innerTail7_60
def innerCandidates7_59 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates7_59_checked : candidates edge 4 29 7853 innerTail7_60 best.bound = innerCandidates7_59 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_59 : scan edge (search edge 3) 4 [29] 29 innerTail7_59 best = best := by
  rw [innerTail7_59, scan, if_neg (by decide), innerCandidates7_59_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_60
    (if 3 ≤ innerCandidates7_59.length then search edge 3 [29, 7853] innerCandidates7_59 7882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_60

def innerTail7_58 : List Nat := 7841 :: innerTail7_59
def innerCandidates7_58 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates7_58_checked : candidates edge 4 29 7841 innerTail7_59 best.bound = innerCandidates7_58 := by
  unfold candidates
  have row : edge 7841 = SearchData.row7841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_58 : scan edge (search edge 3) 4 [29] 29 innerTail7_58 best = best := by
  rw [innerTail7_58, scan, if_neg (by decide), innerCandidates7_58_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_59
    (if 3 ≤ innerCandidates7_58.length then search edge 3 [29, 7841] innerCandidates7_58 7870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_59

def innerTail7_57 : List Nat := 7793 :: innerTail7_58
def innerCandidates7_57 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates7_57_checked : candidates edge 4 29 7793 innerTail7_58 best.bound = innerCandidates7_57 := by
  unfold candidates
  have row : edge 7793 = SearchData.row7793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_57 : scan edge (search edge 3) 4 [29] 29 innerTail7_57 best = best := by
  rw [innerTail7_57, scan, if_neg (by decide), innerCandidates7_57_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_58
    (if 3 ≤ innerCandidates7_57.length then search edge 3 [29, 7793] innerCandidates7_57 7822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_58

def innerTail7_56 : List Nat := 7757 :: innerTail7_57
def innerCandidates7_56 : List Nat := [7853]
set_option maxRecDepth 4096 in
theorem innerCandidates7_56_checked : candidates edge 4 29 7757 innerTail7_57 best.bound = innerCandidates7_56 := by
  unfold candidates
  have row : edge 7757 = SearchData.row7757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_56 : scan edge (search edge 3) 4 [29] 29 innerTail7_56 best = best := by
  rw [innerTail7_56, scan, if_neg (by decide), innerCandidates7_56_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_57
    (if 3 ≤ innerCandidates7_56.length then search edge 3 [29, 7757] innerCandidates7_56 7786 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_57

def innerTail7_55 : List Nat := 7607 :: innerTail7_56
def innerCandidates7_55 : List Nat := [7853]
set_option maxRecDepth 4096 in
theorem innerCandidates7_55_checked : candidates edge 4 29 7607 innerTail7_56 best.bound = innerCandidates7_55 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_55 : scan edge (search edge 3) 4 [29] 29 innerTail7_55 best = best := by
  rw [innerTail7_55, scan, if_neg (by decide), innerCandidates7_55_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_56
    (if 3 ≤ innerCandidates7_55.length then search edge 3 [29, 7607] innerCandidates7_55 7636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_56

def innerTail7_54 : List Nat := 7589 :: innerTail7_55
def innerCandidates7_54 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates7_54_checked : candidates edge 4 29 7589 innerTail7_55 best.bound = innerCandidates7_54 := by
  unfold candidates
  have row : edge 7589 = SearchData.row7589.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_54 : scan edge (search edge 3) 4 [29] 29 innerTail7_54 best = best := by
  rw [innerTail7_54, scan, if_neg (by decide), innerCandidates7_54_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_55
    (if 3 ≤ innerCandidates7_54.length then search edge 3 [29, 7589] innerCandidates7_54 7618 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_55

def innerTail7_53 : List Nat := 7487 :: innerTail7_54
def innerCandidates7_53 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates7_53_checked : candidates edge 4 29 7487 innerTail7_54 best.bound = innerCandidates7_53 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_53 : scan edge (search edge 3) 4 [29] 29 innerTail7_53 best = best := by
  rw [innerTail7_53, scan, if_neg (by decide), innerCandidates7_53_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_54
    (if 3 ≤ innerCandidates7_53.length then search edge 3 [29, 7487] innerCandidates7_53 7516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_54

def innerTail7_52 : List Nat := 7247 :: innerTail7_53
def innerCandidates7_52 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates7_52_checked : candidates edge 4 29 7247 innerTail7_53 best.bound = innerCandidates7_52 := by
  unfold candidates
  have row : edge 7247 = SearchData.row7247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_52 : scan edge (search edge 3) 4 [29] 29 innerTail7_52 best = best := by
  rw [innerTail7_52, scan, if_neg (by decide), innerCandidates7_52_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_53
    (if 3 ≤ innerCandidates7_52.length then search edge 3 [29, 7247] innerCandidates7_52 7276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_53

def innerTail7_51 : List Nat := 7079 :: innerTail7_52
def innerCandidates7_51 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates7_51_checked : candidates edge 4 29 7079 innerTail7_52 best.bound = innerCandidates7_51 := by
  unfold candidates
  have row : edge 7079 = SearchData.row7079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_51 : scan edge (search edge 3) 4 [29] 29 innerTail7_51 best = best := by
  rw [innerTail7_51, scan, if_neg (by decide), innerCandidates7_51_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_52
    (if 3 ≤ innerCandidates7_51.length then search edge 3 [29, 7079] innerCandidates7_51 7108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_52

def innerTail7_50 : List Nat := 6983 :: innerTail7_51
def innerCandidates7_50 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates7_50_checked : candidates edge 4 29 6983 innerTail7_51 best.bound = innerCandidates7_50 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_50 : scan edge (search edge 3) 4 [29] 29 innerTail7_50 best = best := by
  rw [innerTail7_50, scan, if_neg (by decide), innerCandidates7_50_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_51
    (if 3 ≤ innerCandidates7_50.length then search edge 3 [29, 6983] innerCandidates7_50 7012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_51

def innerTail7_49 : List Nat := 6911 :: innerTail7_50
def innerCandidates7_49 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates7_49_checked : candidates edge 4 29 6911 innerTail7_50 best.bound = innerCandidates7_49 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_49 : scan edge (search edge 3) 4 [29] 29 innerTail7_49 best = best := by
  rw [innerTail7_49, scan, if_neg (by decide), innerCandidates7_49_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_50
    (if 3 ≤ innerCandidates7_49.length then search edge 3 [29, 6911] innerCandidates7_49 6940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_50

def innerTail7_48 : List Nat := 6827 :: innerTail7_49
def innerCandidates7_48 : List Nat := [7487]
set_option maxRecDepth 4096 in
theorem innerCandidates7_48_checked : candidates edge 4 29 6827 innerTail7_49 best.bound = innerCandidates7_48 := by
  unfold candidates
  have row : edge 6827 = SearchData.row6827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_48 : scan edge (search edge 3) 4 [29] 29 innerTail7_48 best = best := by
  rw [innerTail7_48, scan, if_neg (by decide), innerCandidates7_48_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_49
    (if 3 ≤ innerCandidates7_48.length then search edge 3 [29, 6827] innerCandidates7_48 6856 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_49

def innerTail7_47 : List Nat := 6563 :: innerTail7_48
def innerCandidates7_47 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates7_47_checked : candidates edge 4 29 6563 innerTail7_48 best.bound = innerCandidates7_47 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_47 : scan edge (search edge 3) 4 [29] 29 innerTail7_47 best = best := by
  rw [innerTail7_47, scan, if_neg (by decide), innerCandidates7_47_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_48
    (if 3 ≤ innerCandidates7_47.length then search edge 3 [29, 6563] innerCandidates7_47 6592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_48

def innerTail7_46 : List Nat := 6299 :: innerTail7_47
def innerCandidates7_46 : List Nat := [7589]
set_option maxRecDepth 4096 in
theorem innerCandidates7_46_checked : candidates edge 4 29 6299 innerTail7_47 best.bound = innerCandidates7_46 := by
  unfold candidates
  have row : edge 6299 = SearchData.row6299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_46 : scan edge (search edge 3) 4 [29] 29 innerTail7_46 best = best := by
  rw [innerTail7_46, scan, if_neg (by decide), innerCandidates7_46_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_47
    (if 3 ≤ innerCandidates7_46.length then search edge 3 [29, 6299] innerCandidates7_46 6328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_47

def innerTail7_45 : List Nat := 6269 :: innerTail7_46
def innerCandidates7_45 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates7_45_checked : candidates edge 4 29 6269 innerTail7_46 best.bound = innerCandidates7_45 := by
  unfold candidates
  have row : edge 6269 = SearchData.row6269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_45 : scan edge (search edge 3) 4 [29] 29 innerTail7_45 best = best := by
  rw [innerTail7_45, scan, if_neg (by decide), innerCandidates7_45_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_46
    (if 3 ≤ innerCandidates7_45.length then search edge 3 [29, 6269] innerCandidates7_45 6298 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_46

def innerTail7_44 : List Nat := 6221 :: innerTail7_45
def innerCandidates7_44 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates7_44_checked : candidates edge 4 29 6221 innerTail7_45 best.bound = innerCandidates7_44 := by
  unfold candidates
  have row : edge 6221 = SearchData.row6221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_44 : scan edge (search edge 3) 4 [29] 29 innerTail7_44 best = best := by
  rw [innerTail7_44, scan, if_neg (by decide), innerCandidates7_44_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_45
    (if 3 ≤ innerCandidates7_44.length then search edge 3 [29, 6221] innerCandidates7_44 6250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_45

def innerTail7_43 : List Nat := 6047 :: innerTail7_44
def innerCandidates7_43 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates7_43_checked : candidates edge 4 29 6047 innerTail7_44 best.bound = innerCandidates7_43 := by
  unfold candidates
  have row : edge 6047 = SearchData.row6047.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_43 : scan edge (search edge 3) 4 [29] 29 innerTail7_43 best = best := by
  rw [innerTail7_43, scan, if_neg (by decide), innerCandidates7_43_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_44
    (if 3 ≤ innerCandidates7_43.length then search edge 3 [29, 6047] innerCandidates7_43 6076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_44

def innerTail7_42 : List Nat := 5861 :: innerTail7_43
def innerCandidates7_42 : List Nat := [6827]
set_option maxRecDepth 4096 in
theorem innerCandidates7_42_checked : candidates edge 4 29 5861 innerTail7_43 best.bound = innerCandidates7_42 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_42 : scan edge (search edge 3) 4 [29] 29 innerTail7_42 best = best := by
  rw [innerTail7_42, scan, if_neg (by decide), innerCandidates7_42_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_43
    (if 3 ≤ innerCandidates7_42.length then search edge 3 [29, 5861] innerCandidates7_42 5890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_43

def innerTail7_41 : List Nat := 5441 :: innerTail7_42
def innerCandidates7_41 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates7_41_checked : candidates edge 4 29 5441 innerTail7_42 best.bound = innerCandidates7_41 := by
  unfold candidates
  have row : edge 5441 = SearchData.row5441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_41 : scan edge (search edge 3) 4 [29] 29 innerTail7_41 best = best := by
  rw [innerTail7_41, scan, if_neg (by decide), innerCandidates7_41_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_42
    (if 3 ≤ innerCandidates7_41.length then search edge 3 [29, 5441] innerCandidates7_41 5470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_42

def innerTail7_40 : List Nat := 5237 :: innerTail7_41
def innerCandidates7_40 : List Nat := [6563, 7487]
set_option maxRecDepth 4096 in
theorem innerCandidates7_40_checked : candidates edge 4 29 5237 innerTail7_41 best.bound = innerCandidates7_40 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_40 : scan edge (search edge 3) 4 [29] 29 innerTail7_40 best = best := by
  rw [innerTail7_40, scan, if_neg (by decide), innerCandidates7_40_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_41
    (if 3 ≤ innerCandidates7_40.length then search edge 3 [29, 5237] innerCandidates7_40 5266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_41

def innerTail7_39 : List Nat := 5081 :: innerTail7_40
def innerCandidates7_39 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates7_39_checked : candidates edge 4 29 5081 innerTail7_40 best.bound = innerCandidates7_39 := by
  unfold candidates
  have row : edge 5081 = SearchData.row5081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_39 : scan edge (search edge 3) 4 [29] 29 innerTail7_39 best = best := by
  rw [innerTail7_39, scan, if_neg (by decide), innerCandidates7_39_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_40
    (if 3 ≤ innerCandidates7_39.length then search edge 3 [29, 5081] innerCandidates7_39 5110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_40

def innerTail7_38 : List Nat := 5039 :: innerTail7_39
def innerCandidates7_38 : List Nat := [6299]
set_option maxRecDepth 4096 in
theorem innerCandidates7_38_checked : candidates edge 4 29 5039 innerTail7_39 best.bound = innerCandidates7_38 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_38 : scan edge (search edge 3) 4 [29] 29 innerTail7_38 best = best := by
  rw [innerTail7_38, scan, if_neg (by decide), innerCandidates7_38_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_39
    (if 3 ≤ innerCandidates7_38.length then search edge 3 [29, 5039] innerCandidates7_38 5068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_39

def innerTail7_37 : List Nat := 4787 :: innerTail7_38
def innerCandidates7_37 : List Nat := [5237, 6047]
set_option maxRecDepth 4096 in
theorem innerCandidates7_37_checked : candidates edge 4 29 4787 innerTail7_38 best.bound = innerCandidates7_37 := by
  unfold candidates
  have row : edge 4787 = SearchData.row4787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_37 : scan edge (search edge 3) 4 [29] 29 innerTail7_37 best = best := by
  rw [innerTail7_37, scan, if_neg (by decide), innerCandidates7_37_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_38
    (if 3 ≤ innerCandidates7_37.length then search edge 3 [29, 4787] innerCandidates7_37 4816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_38

def innerTail7_36 : List Nat := 4673 :: innerTail7_37
def innerCandidates7_36 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates7_36_checked : candidates edge 4 29 4673 innerTail7_37 best.bound = innerCandidates7_36 := by
  unfold candidates
  have row : edge 4673 = SearchData.row4673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_36 : scan edge (search edge 3) 4 [29] 29 innerTail7_36 best = best := by
  rw [innerTail7_36, scan, if_neg (by decide), innerCandidates7_36_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_37
    (if 3 ≤ innerCandidates7_36.length then search edge 3 [29, 4673] innerCandidates7_36 4702 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_37

def innerTail7_35 : List Nat := 4523 :: innerTail7_36
def innerCandidates7_35 : List Nat := [5039, 5081, 6911]
set_option maxRecDepth 4096 in
theorem innerCandidates7_35_checked : candidates edge 4 29 4523 innerTail7_36 best.bound = innerCandidates7_35 := by
  unfold candidates
  have row : edge 4523 = SearchData.row4523.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild7_35 : search edge 3 [29, 4523] innerCandidates7_35 4552 best = best := by decide

theorem innerScan7_35 : scan edge (search edge 3) 4 [29] 29 innerTail7_35 best = best := by
  rw [innerTail7_35, scan, if_neg (by decide), innerCandidates7_35_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_36
    (if 3 ≤ innerCandidates7_35.length then search edge 3 [29, 4523] innerCandidates7_35 4552 best else best) = best
  rw [if_pos (by decide), innerChild7_35]
  exact innerScan7_36

def innerTail7_34 : List Nat := 4397 :: innerTail7_35
def innerCandidates7_34 : List Nat := [6563, 7841]
set_option maxRecDepth 4096 in
theorem innerCandidates7_34_checked : candidates edge 4 29 4397 innerTail7_35 best.bound = innerCandidates7_34 := by
  unfold candidates
  have row : edge 4397 = SearchData.row4397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_34 : scan edge (search edge 3) 4 [29] 29 innerTail7_34 best = best := by
  rw [innerTail7_34, scan, if_neg (by decide), innerCandidates7_34_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_35
    (if 3 ≤ innerCandidates7_34.length then search edge 3 [29, 4397] innerCandidates7_34 4426 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_35

def innerTail7_33 : List Nat := 4337 :: innerTail7_34
def innerCandidates7_33 : List Nat := [4523, 5237]
set_option maxRecDepth 4096 in
theorem innerCandidates7_33_checked : candidates edge 4 29 4337 innerTail7_34 best.bound = innerCandidates7_33 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_33 : scan edge (search edge 3) 4 [29] 29 innerTail7_33 best = best := by
  rw [innerTail7_33, scan, if_neg (by decide), innerCandidates7_33_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_34
    (if 3 ≤ innerCandidates7_33.length then search edge 3 [29, 4337] innerCandidates7_33 4366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_34

def innerTail7_32 : List Nat := 4241 :: innerTail7_33
def innerCandidates7_32 : List Nat := [7757]
set_option maxRecDepth 4096 in
theorem innerCandidates7_32_checked : candidates edge 4 29 4241 innerTail7_33 best.bound = innerCandidates7_32 := by
  unfold candidates
  have row : edge 4241 = SearchData.row4241.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_32 : scan edge (search edge 3) 4 [29] 29 innerTail7_32 best = best := by
  rw [innerTail7_32, scan, if_neg (by decide), innerCandidates7_32_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_33
    (if 3 ≤ innerCandidates7_32.length then search edge 3 [29, 4241] innerCandidates7_32 4270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_33

def innerTail7_31 : List Nat := 4157 :: innerTail7_32
def innerCandidates7_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates7_31_checked : candidates edge 4 29 4157 innerTail7_32 best.bound = innerCandidates7_31 := by
  unfold candidates
  have row : edge 4157 = SearchData.row4157.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_31 : scan edge (search edge 3) 4 [29] 29 innerTail7_31 best = best := by
  rw [innerTail7_31, scan, if_neg (by decide), innerCandidates7_31_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_32
    (if 3 ≤ innerCandidates7_31.length then search edge 3 [29, 4157] innerCandidates7_31 4186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_32

def innerTail7_30 : List Nat := 4013 :: innerTail7_31
def innerCandidates7_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates7_30_checked : candidates edge 4 29 4013 innerTail7_31 best.bound = innerCandidates7_30 := by
  unfold candidates
  have row : edge 4013 = SearchData.row4013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_30 : scan edge (search edge 3) 4 [29] 29 innerTail7_30 best = best := by
  rw [innerTail7_30, scan, if_neg (by decide), innerCandidates7_30_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_31
    (if 3 ≤ innerCandidates7_30.length then search edge 3 [29, 4013] innerCandidates7_30 4042 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_31

def innerTail7_29 : List Nat := 3863 :: innerTail7_30
def innerCandidates7_29 : List Nat := [7757]
set_option maxRecDepth 4096 in
theorem innerCandidates7_29_checked : candidates edge 4 29 3863 innerTail7_30 best.bound = innerCandidates7_29 := by
  unfold candidates
  have row : edge 3863 = SearchData.row3863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_29 : scan edge (search edge 3) 4 [29] 29 innerTail7_29 best = best := by
  rw [innerTail7_29, scan, if_neg (by decide), innerCandidates7_29_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_30
    (if 3 ≤ innerCandidates7_29.length then search edge 3 [29, 3863] innerCandidates7_29 3892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_30

def innerTail7_28 : List Nat := 3803 :: innerTail7_29
def innerCandidates7_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates7_28_checked : candidates edge 4 29 3803 innerTail7_29 best.bound = innerCandidates7_28 := by
  unfold candidates
  have row : edge 3803 = SearchData.row3803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_28 : scan edge (search edge 3) 4 [29] 29 innerTail7_28 best = best := by
  rw [innerTail7_28, scan, if_neg (by decide), innerCandidates7_28_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_29
    (if 3 ≤ innerCandidates7_28.length then search edge 3 [29, 3803] innerCandidates7_28 3832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_29

def innerTail7_27 : List Nat := 3767 :: innerTail7_28
def innerCandidates7_27 : List Nat := [4523, 6911, 7487, 7793]
set_option maxRecDepth 4096 in
theorem innerCandidates7_27_checked : candidates edge 4 29 3767 innerTail7_28 best.bound = innerCandidates7_27 := by
  unfold candidates
  have row : edge 3767 = SearchData.row3767.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild7_27 : search edge 3 [29, 3767] innerCandidates7_27 3796 best = best := by decide

theorem innerScan7_27 : scan edge (search edge 3) 4 [29] 29 innerTail7_27 best = best := by
  rw [innerTail7_27, scan, if_neg (by decide), innerCandidates7_27_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_28
    (if 3 ≤ innerCandidates7_27.length then search edge 3 [29, 3767] innerCandidates7_27 3796 best else best) = best
  rw [if_pos (by decide), innerChild7_27]
  exact innerScan7_28

def innerTail7_26 : List Nat := 3659 :: innerTail7_27
def innerCandidates7_26 : List Nat := [3803, 4157, 6047]
set_option maxRecDepth 4096 in
theorem innerCandidates7_26_checked : candidates edge 4 29 3659 innerTail7_27 best.bound = innerCandidates7_26 := by
  unfold candidates
  have row : edge 3659 = SearchData.row3659.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild7_26 : search edge 3 [29, 3659] innerCandidates7_26 3688 best = best := by decide

theorem innerScan7_26 : scan edge (search edge 3) 4 [29] 29 innerTail7_26 best = best := by
  rw [innerTail7_26, scan, if_neg (by decide), innerCandidates7_26_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_27
    (if 3 ≤ innerCandidates7_26.length then search edge 3 [29, 3659] innerCandidates7_26 3688 best else best) = best
  rw [if_pos (by decide), innerChild7_26]
  exact innerScan7_27

def innerTail7_25 : List Nat := 3257 :: innerTail7_26
def innerCandidates7_25 : List Nat := [5441, 6221]
set_option maxRecDepth 4096 in
theorem innerCandidates7_25_checked : candidates edge 4 29 3257 innerTail7_26 best.bound = innerCandidates7_25 := by
  unfold candidates
  have row : edge 3257 = SearchData.row3257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_25 : scan edge (search edge 3) 4 [29] 29 innerTail7_25 best = best := by
  rw [innerTail7_25, scan, if_neg (by decide), innerCandidates7_25_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_26
    (if 3 ≤ innerCandidates7_25.length then search edge 3 [29, 3257] innerCandidates7_25 3286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_26

def innerTail7_24 : List Nat := 2969 :: innerTail7_25
def innerCandidates7_24 : List Nat := [7841]
set_option maxRecDepth 4096 in
theorem innerCandidates7_24_checked : candidates edge 4 29 2969 innerTail7_25 best.bound = innerCandidates7_24 := by
  unfold candidates
  have row : edge 2969 = SearchData.row2969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_24 : scan edge (search edge 3) 4 [29] 29 innerTail7_24 best = best := by
  rw [innerTail7_24, scan, if_neg (by decide), innerCandidates7_24_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_25
    (if 3 ≤ innerCandidates7_24.length then search edge 3 [29, 2969] innerCandidates7_24 2998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_25

def innerTail7_23 : List Nat := 2801 :: innerTail7_24
def innerCandidates7_23 : List Nat := [3803]
set_option maxRecDepth 4096 in
theorem innerCandidates7_23_checked : candidates edge 4 29 2801 innerTail7_24 best.bound = innerCandidates7_23 := by
  unfold candidates
  have row : edge 2801 = SearchData.row2801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_23 : scan edge (search edge 3) 4 [29] 29 innerTail7_23 best = best := by
  rw [innerTail7_23, scan, if_neg (by decide), innerCandidates7_23_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_24
    (if 3 ≤ innerCandidates7_23.length then search edge 3 [29, 2801] innerCandidates7_23 2830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_24

def innerTail7_22 : List Nat := 2711 :: innerTail7_23
def innerCandidates7_22 : List Nat := [5441, 7589]
set_option maxRecDepth 4096 in
theorem innerCandidates7_22_checked : candidates edge 4 29 2711 innerTail7_23 best.bound = innerCandidates7_22 := by
  unfold candidates
  have row : edge 2711 = SearchData.row2711.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_22 : scan edge (search edge 3) 4 [29] 29 innerTail7_22 best = best := by
  rw [innerTail7_22, scan, if_neg (by decide), innerCandidates7_22_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_23
    (if 3 ≤ innerCandidates7_22.length then search edge 3 [29, 2711] innerCandidates7_22 2740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_23

def innerTail7_21 : List Nat := 2549 :: innerTail7_22
def innerCandidates7_21 : List Nat := [4241, 7793]
set_option maxRecDepth 4096 in
theorem innerCandidates7_21_checked : candidates edge 4 29 2549 innerTail7_22 best.bound = innerCandidates7_21 := by
  unfold candidates
  have row : edge 2549 = PrimePairCache.smallRow2549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_21 : scan edge (search edge 3) 4 [29] 29 innerTail7_21 best = best := by
  rw [innerTail7_21, scan, if_neg (by decide), innerCandidates7_21_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_22
    (if 3 ≤ innerCandidates7_21.length then search edge 3 [29, 2549] innerCandidates7_21 2578 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_22

def innerTail7_20 : List Nat := 2477 :: innerTail7_21
def innerCandidates7_20 : List Nat := [2801, 5039, 7853]
set_option maxRecDepth 4096 in
theorem innerCandidates7_20_checked : candidates edge 4 29 2477 innerTail7_21 best.bound = innerCandidates7_20 := by
  unfold candidates
  have row : edge 2477 = PrimePairCache.smallRow2477.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild7_20 : search edge 3 [29, 2477] innerCandidates7_20 2506 best = best := by decide

theorem innerScan7_20 : scan edge (search edge 3) 4 [29] 29 innerTail7_20 best = best := by
  rw [innerTail7_20, scan, if_neg (by decide), innerCandidates7_20_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_21
    (if 3 ≤ innerCandidates7_20.length then search edge 3 [29, 2477] innerCandidates7_20 2506 best else best) = best
  rw [if_pos (by decide), innerChild7_20]
  exact innerScan7_21

def innerTail7_19 : List Nat := 2441 :: innerTail7_20
def innerCandidates7_19 : List Nat := [4157, 4787]
set_option maxRecDepth 4096 in
theorem innerCandidates7_19_checked : candidates edge 4 29 2441 innerTail7_20 best.bound = innerCandidates7_19 := by
  unfold candidates
  have row : edge 2441 = PrimePairCache.smallRow2441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_19 : scan edge (search edge 3) 4 [29] 29 innerTail7_19 best = best := by
  rw [innerTail7_19, scan, if_neg (by decide), innerCandidates7_19_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_20
    (if 3 ≤ innerCandidates7_19.length then search edge 3 [29, 2441] innerCandidates7_19 2470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_20

def innerTail7_18 : List Nat := 2393 :: innerTail7_19
def innerCandidates7_18 : List Nat := [4241]
set_option maxRecDepth 4096 in
theorem innerCandidates7_18_checked : candidates edge 4 29 2393 innerTail7_19 best.bound = innerCandidates7_18 := by
  unfold candidates
  have row : edge 2393 = PrimePairCache.smallRow2393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_18 : scan edge (search edge 3) 4 [29] 29 innerTail7_18 best = best := by
  rw [innerTail7_18, scan, if_neg (by decide), innerCandidates7_18_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_19
    (if 3 ≤ innerCandidates7_18.length then search edge 3 [29, 2393] innerCandidates7_18 2422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_19

def innerTail7_17 : List Nat := 2309 :: innerTail7_18
def innerCandidates7_17 : List Nat := [5081, 8237]
set_option maxRecDepth 4096 in
theorem innerCandidates7_17_checked : candidates edge 4 29 2309 innerTail7_18 best.bound = innerCandidates7_17 := by
  unfold candidates
  have row : edge 2309 = PrimePairCache.smallRow2309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_17 : scan edge (search edge 3) 4 [29] 29 innerTail7_17 best = best := by
  rw [innerTail7_17, scan, if_neg (by decide), innerCandidates7_17_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_18
    (if 3 ≤ innerCandidates7_17.length then search edge 3 [29, 2309] innerCandidates7_17 2338 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_18

def innerTail7_16 : List Nat := 2141 :: innerTail7_17
def innerCandidates7_16 : List Nat := [2309]
set_option maxRecDepth 4096 in
theorem innerCandidates7_16_checked : candidates edge 4 29 2141 innerTail7_17 best.bound = innerCandidates7_16 := by
  unfold candidates
  have row : edge 2141 = PrimePairCache.smallRow2141.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_16 : scan edge (search edge 3) 4 [29] 29 innerTail7_16 best = best := by
  rw [innerTail7_16, scan, if_neg (by decide), innerCandidates7_16_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_17
    (if 3 ≤ innerCandidates7_16.length then search edge 3 [29, 2141] innerCandidates7_16 2170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_17

def innerTail7_15 : List Nat := 2081 :: innerTail7_16
def innerCandidates7_15 : List Nat := [5441, 7793]
set_option maxRecDepth 4096 in
theorem innerCandidates7_15_checked : candidates edge 4 29 2081 innerTail7_16 best.bound = innerCandidates7_15 := by
  unfold candidates
  have row : edge 2081 = PrimePairCache.smallRow2081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_15 : scan edge (search edge 3) 4 [29] 29 innerTail7_15 best = best := by
  rw [innerTail7_15, scan, if_neg (by decide), innerCandidates7_15_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_16
    (if 3 ≤ innerCandidates7_15.length then search edge 3 [29, 2081] innerCandidates7_15 2110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_16

def innerTail7_14 : List Nat := 2027 :: innerTail7_15
def innerCandidates7_14 : List Nat := [6911]
set_option maxRecDepth 4096 in
theorem innerCandidates7_14_checked : candidates edge 4 29 2027 innerTail7_15 best.bound = innerCandidates7_14 := by
  unfold candidates
  have row : edge 2027 = PrimePairCache.smallRow2027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_14 : scan edge (search edge 3) 4 [29] 29 innerTail7_14 best = best := by
  rw [innerTail7_14, scan, if_neg (by decide), innerCandidates7_14_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_15
    (if 3 ≤ innerCandidates7_14.length then search edge 3 [29, 2027] innerCandidates7_14 2056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_15

def innerTail7_13 : List Nat := 1997 :: innerTail7_14
def innerCandidates7_13 : List Nat := [4397]
set_option maxRecDepth 4096 in
theorem innerCandidates7_13_checked : candidates edge 4 29 1997 innerTail7_14 best.bound = innerCandidates7_13 := by
  unfold candidates
  have row : edge 1997 = PrimePairCache.smallRow1997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_13 : scan edge (search edge 3) 4 [29] 29 innerTail7_13 best = best := by
  rw [innerTail7_13, scan, if_neg (by decide), innerCandidates7_13_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_14
    (if 3 ≤ innerCandidates7_13.length then search edge 3 [29, 1997] innerCandidates7_13 2026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_14

def innerTail7_12 : List Nat := 1901 :: innerTail7_13
def innerCandidates7_12 : List Nat := [2801, 2969, 5861, 7589]
set_option maxRecDepth 4096 in
theorem innerCandidates7_12_checked : candidates edge 4 29 1901 innerTail7_13 best.bound = innerCandidates7_12 := by
  unfold candidates
  have row : edge 1901 = PrimePairCache.smallRow1901.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild7_12 : search edge 3 [29, 1901] innerCandidates7_12 1930 best = best := by decide

theorem innerScan7_12 : scan edge (search edge 3) 4 [29] 29 innerTail7_12 best = best := by
  rw [innerTail7_12, scan, if_neg (by decide), innerCandidates7_12_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_13
    (if 3 ≤ innerCandidates7_12.length then search edge 3 [29, 1901] innerCandidates7_12 1930 best else best) = best
  rw [if_pos (by decide), innerChild7_12]
  exact innerScan7_13

def innerTail7_11 : List Nat := 881 :: innerTail7_12
def innerCandidates7_11 : List Nat := [6827, 7607]
set_option maxRecDepth 4096 in
theorem innerCandidates7_11_checked : candidates edge 4 29 881 innerTail7_12 best.bound = innerCandidates7_11 := by
  unfold candidates
  have row : edge 881 = PrimePairCache.smallRow881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_11 : scan edge (search edge 3) 4 [29] 29 innerTail7_11 best = best := by
  rw [innerTail7_11, scan, if_neg (by decide), innerCandidates7_11_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_12
    (if 3 ≤ innerCandidates7_11.length then search edge 3 [29, 881] innerCandidates7_11 910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_12

def innerTail7_10 : List Nat := 761 :: innerTail7_11
def innerCandidates7_10 : List Nat := [6911]
set_option maxRecDepth 4096 in
theorem innerCandidates7_10_checked : candidates edge 4 29 761 innerTail7_11 best.bound = innerCandidates7_10 := by
  unfold candidates
  have row : edge 761 = PrimePairCache.smallRow761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_10 : scan edge (search edge 3) 4 [29] 29 innerTail7_10 best = best := by
  rw [innerTail7_10, scan, if_neg (by decide), innerCandidates7_10_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_11
    (if 3 ≤ innerCandidates7_10.length then search edge 3 [29, 761] innerCandidates7_10 790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_11

def innerTail7_9 : List Nat := 683 :: innerTail7_10
def innerCandidates7_9 : List Nat := [6047]
set_option maxRecDepth 4096 in
theorem innerCandidates7_9_checked : candidates edge 4 29 683 innerTail7_10 best.bound = innerCandidates7_9 := by
  unfold candidates
  have row : edge 683 = PrimePairCache.smallRow683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_9 : scan edge (search edge 3) 4 [29] 29 innerTail7_9 best = best := by
  rw [innerTail7_9, scan, if_neg (by decide), innerCandidates7_9_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_10
    (if 3 ≤ innerCandidates7_9.length then search edge 3 [29, 683] innerCandidates7_9 712 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_10

def innerTail7_8 : List Nat := 599 :: innerTail7_9
def innerCandidates7_8 : List Nat := [5081, 6047, 7841]
set_option maxRecDepth 4096 in
theorem innerCandidates7_8_checked : candidates edge 4 29 599 innerTail7_9 best.bound = innerCandidates7_8 := by
  unfold candidates
  have row : edge 599 = PrimePairCache.smallRow599.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild7_8 : search edge 3 [29, 599] innerCandidates7_8 628 best = best := by decide

theorem innerScan7_8 : scan edge (search edge 3) 4 [29] 29 innerTail7_8 best = best := by
  rw [innerTail7_8, scan, if_neg (by decide), innerCandidates7_8_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_9
    (if 3 ≤ innerCandidates7_8.length then search edge 3 [29, 599] innerCandidates7_8 628 best else best) = best
  rw [if_pos (by decide), innerChild7_8]
  exact innerScan7_9

def innerTail7_7 : List Nat := 569 :: innerTail7_8
def innerCandidates7_7 : List Nat := [4673]
set_option maxRecDepth 4096 in
theorem innerCandidates7_7_checked : candidates edge 4 29 569 innerTail7_8 best.bound = innerCandidates7_7 := by
  unfold candidates
  have row : edge 569 = PrimePairCache.smallRow569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_7 : scan edge (search edge 3) 4 [29] 29 innerTail7_7 best = best := by
  rw [innerTail7_7, scan, if_neg (by decide), innerCandidates7_7_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_8
    (if 3 ≤ innerCandidates7_7.length then search edge 3 [29, 569] innerCandidates7_7 598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_8

def innerTail7_6 : List Nat := 401 :: innerTail7_7
def innerCandidates7_6 : List Nat := [2441, 7589]
set_option maxRecDepth 4096 in
theorem innerCandidates7_6_checked : candidates edge 4 29 401 innerTail7_7 best.bound = innerCandidates7_6 := by
  unfold candidates
  have row : edge 401 = PrimePairCache.smallRow401.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_6 : scan edge (search edge 3) 4 [29] 29 innerTail7_6 best = best := by
  rw [innerTail7_6, scan, if_neg (by decide), innerCandidates7_6_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_7
    (if 3 ≤ innerCandidates7_6.length then search edge 3 [29, 401] innerCandidates7_6 430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_7

def innerTail7_5 : List Nat := 383 :: innerTail7_6
def innerCandidates7_5 : List Nat := [3257, 7853]
set_option maxRecDepth 4096 in
theorem innerCandidates7_5_checked : candidates edge 4 29 383 innerTail7_6 best.bound = innerCandidates7_5 := by
  unfold candidates
  have row : edge 383 = PrimePairCache.smallRow383.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_5 : scan edge (search edge 3) 4 [29] 29 innerTail7_5 best = best := by
  rw [innerTail7_5, scan, if_neg (by decide), innerCandidates7_5_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_6
    (if 3 ≤ innerCandidates7_5.length then search edge 3 [29, 383] innerCandidates7_5 412 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_6

def innerTail7_4 : List Nat := 347 :: innerTail7_5
def innerCandidates7_4 : List Nat := [401, 5861, 7841]
set_option maxRecDepth 4096 in
theorem innerCandidates7_4_checked : candidates edge 4 29 347 innerTail7_5 best.bound = innerCandidates7_4 := by
  unfold candidates
  have row : edge 347 = PrimePairCache.smallRow347.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild7_4 : search edge 3 [29, 347] innerCandidates7_4 376 best = best := by decide

theorem innerScan7_4 : scan edge (search edge 3) 4 [29] 29 innerTail7_4 best = best := by
  rw [innerTail7_4, scan, if_neg (by decide), innerCandidates7_4_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_5
    (if 3 ≤ innerCandidates7_4.length then search edge 3 [29, 347] innerCandidates7_4 376 best else best) = best
  rw [if_pos (by decide), innerChild7_4]
  exact innerScan7_5

def innerTail7_3 : List Nat := 179 :: innerTail7_4
def innerCandidates7_3 : List Nat := [383, 2309, 4787]
set_option maxRecDepth 4096 in
theorem innerCandidates7_3_checked : candidates edge 4 29 179 innerTail7_4 best.bound = innerCandidates7_3 := by
  unfold candidates
  have row : edge 179 = PrimePairCache.smallRow179.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild7_3 : search edge 3 [29, 179] innerCandidates7_3 208 best = best := by decide

theorem innerScan7_3 : scan edge (search edge 3) 4 [29] 29 innerTail7_3 best = best := by
  rw [innerTail7_3, scan, if_neg (by decide), innerCandidates7_3_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_4
    (if 3 ≤ innerCandidates7_3.length then search edge 3 [29, 179] innerCandidates7_3 208 best else best) = best
  rw [if_pos (by decide), innerChild7_3]
  exact innerScan7_4

def innerTail7_2 : List Nat := 167 :: innerTail7_3
def innerCandidates7_2 : List Nat := [2441, 5441, 6827]
set_option maxRecDepth 4096 in
theorem innerCandidates7_2_checked : candidates edge 4 29 167 innerTail7_3 best.bound = innerCandidates7_2 := by
  unfold candidates
  have row : edge 167 = PrimePairCache.smallRow167.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild7_2 : search edge 3 [29, 167] innerCandidates7_2 196 best = best := by decide

theorem innerScan7_2 : scan edge (search edge 3) 4 [29] 29 innerTail7_2 best = best := by
  rw [innerTail7_2, scan, if_neg (by decide), innerCandidates7_2_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_3
    (if 3 ≤ innerCandidates7_2.length then search edge 3 [29, 167] innerCandidates7_2 196 best else best) = best
  rw [if_pos (by decide), innerChild7_2]
  exact innerScan7_3

def innerTail7_1 : List Nat := 137 :: innerTail7_2
def innerCandidates7_1 : List Nat := [5237, 7757]
set_option maxRecDepth 4096 in
theorem innerCandidates7_1_checked : candidates edge 4 29 137 innerTail7_2 best.bound = innerCandidates7_1 := by
  unfold candidates
  have row : edge 137 = PrimePairCache.smallRow137.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan7_1 : scan edge (search edge 3) 4 [29] 29 innerTail7_1 best = best := by
  rw [innerTail7_1, scan, if_neg (by decide), innerCandidates7_1_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_2
    (if 3 ≤ innerCandidates7_1.length then search edge 3 [29, 137] innerCandidates7_1 166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan7_2

def innerTail7_0 : List Nat := 71 :: innerTail7_1
def innerCandidates7_0 : List Nat := [2477, 3863, 4673, 5237, 5441, 6563, 7589, 7841]
set_option maxRecDepth 4096 in
theorem innerCandidates7_0_checked : candidates edge 4 29 71 innerTail7_1 best.bound = innerCandidates7_0 := by
  unfold candidates
  have row : edge 71 = PrimePairCache.smallRow71.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild7_0 : search edge 3 [29, 71] innerCandidates7_0 100 best = best := by decide

theorem innerScan7_0 : scan edge (search edge 3) 4 [29] 29 innerTail7_0 best = best := by
  rw [innerTail7_0, scan, if_neg (by decide), innerCandidates7_0_checked]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_1
    (if 3 ≤ innerCandidates7_0.length then search edge 3 [29, 71] innerCandidates7_0 100 best else best) = best
  rw [if_pos (by decide), innerChild7_0]
  exact innerScan7_1

theorem rootBranch7 : search edge 4 [29] rootCandidates7 29 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [29] 29 innerTail7_0 best = best
  exact innerScan7_0

end ProjectEulerSolutions.PrimePairUpper.Search
