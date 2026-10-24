import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates5 : List Nat := [31, 79, 97, 163, 181, 433, 571, 577, 709, 727, 937, 991, 997, 1033, 1039, 1237, 1249, 1297, 1447, 1621, 1657, 1669, 1693, 1699, 1831, 1861, 2053, 2113, 2161, 2239, 2251, 2539, 2677, 2791, 3727, 3793, 3847, 3943, 4057, 4507, 4723, 4861, 5077, 5659, 5737, 5791, 5869, 6043, 6073, 6271, 6661, 6709, 6907, 6961, 6991, 7159, 7297, 7507, 7573, 7699, 7741, 7759, 8179]
set_option maxRecDepth 4096 in
theorem rootCandidates5_checked : candidates edge 5 0 19 tail6 best.bound = rootCandidates5 := by
  unfold candidates
  have row : edge 19 = PrimePairCache.smallRow19.contains := by funext q; rfl
  rw [row]
  decide

def innerTail5_63 : List Nat := []
theorem innerScan5_63 : scan edge (search edge 3) 4 [19] 19 innerTail5_63 best = best := rfl

def innerTail5_62 : List Nat := 8179 :: innerTail5_63
def innerCandidates5_62 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates5_62_checked : candidates edge 4 19 8179 innerTail5_63 best.bound = innerCandidates5_62 := by
  unfold candidates
  have row : edge 8179 = SearchData.row8179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_62 : scan edge (search edge 3) 4 [19] 19 innerTail5_62 best = best := by
  rw [innerTail5_62, scan, if_neg (by decide), innerCandidates5_62_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_63
    (if 3 ≤ innerCandidates5_62.length then search edge 3 [19, 8179] innerCandidates5_62 8198 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_63

def innerTail5_61 : List Nat := 7759 :: innerTail5_62
def innerCandidates5_61 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates5_61_checked : candidates edge 4 19 7759 innerTail5_62 best.bound = innerCandidates5_61 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_61 : scan edge (search edge 3) 4 [19] 19 innerTail5_61 best = best := by
  rw [innerTail5_61, scan, if_neg (by decide), innerCandidates5_61_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_62
    (if 3 ≤ innerCandidates5_61.length then search edge 3 [19, 7759] innerCandidates5_61 7778 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_62

def innerTail5_60 : List Nat := 7741 :: innerTail5_61
def innerCandidates5_60 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates5_60_checked : candidates edge 4 19 7741 innerTail5_61 best.bound = innerCandidates5_60 := by
  unfold candidates
  have row : edge 7741 = SearchData.row7741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_60 : scan edge (search edge 3) 4 [19] 19 innerTail5_60 best = best := by
  rw [innerTail5_60, scan, if_neg (by decide), innerCandidates5_60_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_61
    (if 3 ≤ innerCandidates5_60.length then search edge 3 [19, 7741] innerCandidates5_60 7760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_61

def innerTail5_59 : List Nat := 7699 :: innerTail5_60
def innerCandidates5_59 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates5_59_checked : candidates edge 4 19 7699 innerTail5_60 best.bound = innerCandidates5_59 := by
  unfold candidates
  have row : edge 7699 = SearchData.row7699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_59 : scan edge (search edge 3) 4 [19] 19 innerTail5_59 best = best := by
  rw [innerTail5_59, scan, if_neg (by decide), innerCandidates5_59_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_60
    (if 3 ≤ innerCandidates5_59.length then search edge 3 [19, 7699] innerCandidates5_59 7718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_60

def innerTail5_58 : List Nat := 7573 :: innerTail5_59
def innerCandidates5_58 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates5_58_checked : candidates edge 4 19 7573 innerTail5_59 best.bound = innerCandidates5_58 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_58 : scan edge (search edge 3) 4 [19] 19 innerTail5_58 best = best := by
  rw [innerTail5_58, scan, if_neg (by decide), innerCandidates5_58_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_59
    (if 3 ≤ innerCandidates5_58.length then search edge 3 [19, 7573] innerCandidates5_58 7592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_59

def innerTail5_57 : List Nat := 7507 :: innerTail5_58
def innerCandidates5_57 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates5_57_checked : candidates edge 4 19 7507 innerTail5_58 best.bound = innerCandidates5_57 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_57 : scan edge (search edge 3) 4 [19] 19 innerTail5_57 best = best := by
  rw [innerTail5_57, scan, if_neg (by decide), innerCandidates5_57_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_58
    (if 3 ≤ innerCandidates5_57.length then search edge 3 [19, 7507] innerCandidates5_57 7526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_58

def innerTail5_56 : List Nat := 7297 :: innerTail5_57
def innerCandidates5_56 : List Nat := [7507, 8179]
set_option maxRecDepth 4096 in
theorem innerCandidates5_56_checked : candidates edge 4 19 7297 innerTail5_57 best.bound = innerCandidates5_56 := by
  unfold candidates
  have row : edge 7297 = SearchData.row7297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_56 : scan edge (search edge 3) 4 [19] 19 innerTail5_56 best = best := by
  rw [innerTail5_56, scan, if_neg (by decide), innerCandidates5_56_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_57
    (if 3 ≤ innerCandidates5_56.length then search edge 3 [19, 7297] innerCandidates5_56 7316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_57

def innerTail5_55 : List Nat := 7159 :: innerTail5_56
def innerCandidates5_55 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates5_55_checked : candidates edge 4 19 7159 innerTail5_56 best.bound = innerCandidates5_55 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_55 : scan edge (search edge 3) 4 [19] 19 innerTail5_55 best = best := by
  rw [innerTail5_55, scan, if_neg (by decide), innerCandidates5_55_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_56
    (if 3 ≤ innerCandidates5_55.length then search edge 3 [19, 7159] innerCandidates5_55 7178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_56

def innerTail5_54 : List Nat := 6991 :: innerTail5_55
def innerCandidates5_54 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates5_54_checked : candidates edge 4 19 6991 innerTail5_55 best.bound = innerCandidates5_54 := by
  unfold candidates
  have row : edge 6991 = SearchData.row6991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_54 : scan edge (search edge 3) 4 [19] 19 innerTail5_54 best = best := by
  rw [innerTail5_54, scan, if_neg (by decide), innerCandidates5_54_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_55
    (if 3 ≤ innerCandidates5_54.length then search edge 3 [19, 6991] innerCandidates5_54 7010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_55

def innerTail5_53 : List Nat := 6961 :: innerTail5_54
def innerCandidates5_53 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates5_53_checked : candidates edge 4 19 6961 innerTail5_54 best.bound = innerCandidates5_53 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_53 : scan edge (search edge 3) 4 [19] 19 innerTail5_53 best = best := by
  rw [innerTail5_53, scan, if_neg (by decide), innerCandidates5_53_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_54
    (if 3 ≤ innerCandidates5_53.length then search edge 3 [19, 6961] innerCandidates5_53 6980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_54

def innerTail5_52 : List Nat := 6907 :: innerTail5_53
def innerCandidates5_52 : List Nat := [7759]
set_option maxRecDepth 4096 in
theorem innerCandidates5_52_checked : candidates edge 4 19 6907 innerTail5_53 best.bound = innerCandidates5_52 := by
  unfold candidates
  have row : edge 6907 = SearchData.row6907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_52 : scan edge (search edge 3) 4 [19] 19 innerTail5_52 best = best := by
  rw [innerTail5_52, scan, if_neg (by decide), innerCandidates5_52_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_53
    (if 3 ≤ innerCandidates5_52.length then search edge 3 [19, 6907] innerCandidates5_52 6926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_53

def innerTail5_51 : List Nat := 6709 :: innerTail5_52
def innerCandidates5_51 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates5_51_checked : candidates edge 4 19 6709 innerTail5_52 best.bound = innerCandidates5_51 := by
  unfold candidates
  have row : edge 6709 = SearchData.row6709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_51 : scan edge (search edge 3) 4 [19] 19 innerTail5_51 best = best := by
  rw [innerTail5_51, scan, if_neg (by decide), innerCandidates5_51_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_52
    (if 3 ≤ innerCandidates5_51.length then search edge 3 [19, 6709] innerCandidates5_51 6728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_52

def innerTail5_50 : List Nat := 6661 :: innerTail5_51
def innerCandidates5_50 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates5_50_checked : candidates edge 4 19 6661 innerTail5_51 best.bound = innerCandidates5_50 := by
  unfold candidates
  have row : edge 6661 = SearchData.row6661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_50 : scan edge (search edge 3) 4 [19] 19 innerTail5_50 best = best := by
  rw [innerTail5_50, scan, if_neg (by decide), innerCandidates5_50_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_51
    (if 3 ≤ innerCandidates5_50.length then search edge 3 [19, 6661] innerCandidates5_50 6680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_51

def innerTail5_49 : List Nat := 6271 :: innerTail5_50
def innerCandidates5_49 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates5_49_checked : candidates edge 4 19 6271 innerTail5_50 best.bound = innerCandidates5_49 := by
  unfold candidates
  have row : edge 6271 = SearchData.row6271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_49 : scan edge (search edge 3) 4 [19] 19 innerTail5_49 best = best := by
  rw [innerTail5_49, scan, if_neg (by decide), innerCandidates5_49_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_50
    (if 3 ≤ innerCandidates5_49.length then search edge 3 [19, 6271] innerCandidates5_49 6290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_50

def innerTail5_48 : List Nat := 6073 :: innerTail5_49
def innerCandidates5_48 : List Nat := [7759]
set_option maxRecDepth 4096 in
theorem innerCandidates5_48_checked : candidates edge 4 19 6073 innerTail5_49 best.bound = innerCandidates5_48 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_48 : scan edge (search edge 3) 4 [19] 19 innerTail5_48 best = best := by
  rw [innerTail5_48, scan, if_neg (by decide), innerCandidates5_48_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_49
    (if 3 ≤ innerCandidates5_48.length then search edge 3 [19, 6073] innerCandidates5_48 6092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_49

def innerTail5_47 : List Nat := 6043 :: innerTail5_48
def innerCandidates5_47 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates5_47_checked : candidates edge 4 19 6043 innerTail5_48 best.bound = innerCandidates5_47 := by
  unfold candidates
  have row : edge 6043 = SearchData.row6043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_47 : scan edge (search edge 3) 4 [19] 19 innerTail5_47 best = best := by
  rw [innerTail5_47, scan, if_neg (by decide), innerCandidates5_47_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_48
    (if 3 ≤ innerCandidates5_47.length then search edge 3 [19, 6043] innerCandidates5_47 6062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_48

def innerTail5_46 : List Nat := 5869 :: innerTail5_47
def innerCandidates5_46 : List Nat := [6709, 6961]
set_option maxRecDepth 4096 in
theorem innerCandidates5_46_checked : candidates edge 4 19 5869 innerTail5_47 best.bound = innerCandidates5_46 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_46 : scan edge (search edge 3) 4 [19] 19 innerTail5_46 best = best := by
  rw [innerTail5_46, scan, if_neg (by decide), innerCandidates5_46_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_47
    (if 3 ≤ innerCandidates5_46.length then search edge 3 [19, 5869] innerCandidates5_46 5888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_47

def innerTail5_45 : List Nat := 5791 :: innerTail5_46
def innerCandidates5_45 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates5_45_checked : candidates edge 4 19 5791 innerTail5_46 best.bound = innerCandidates5_45 := by
  unfold candidates
  have row : edge 5791 = SearchData.row5791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_45 : scan edge (search edge 3) 4 [19] 19 innerTail5_45 best = best := by
  rw [innerTail5_45, scan, if_neg (by decide), innerCandidates5_45_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_46
    (if 3 ≤ innerCandidates5_45.length then search edge 3 [19, 5791] innerCandidates5_45 5810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_46

def innerTail5_44 : List Nat := 5737 :: innerTail5_45
def innerCandidates5_44 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates5_44_checked : candidates edge 4 19 5737 innerTail5_45 best.bound = innerCandidates5_44 := by
  unfold candidates
  have row : edge 5737 = SearchData.row5737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_44 : scan edge (search edge 3) 4 [19] 19 innerTail5_44 best = best := by
  rw [innerTail5_44, scan, if_neg (by decide), innerCandidates5_44_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_45
    (if 3 ≤ innerCandidates5_44.length then search edge 3 [19, 5737] innerCandidates5_44 5756 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_45

def innerTail5_43 : List Nat := 5659 :: innerTail5_44
def innerCandidates5_43 : List Nat := [6271, 7759]
set_option maxRecDepth 4096 in
theorem innerCandidates5_43_checked : candidates edge 4 19 5659 innerTail5_44 best.bound = innerCandidates5_43 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_43 : scan edge (search edge 3) 4 [19] 19 innerTail5_43 best = best := by
  rw [innerTail5_43, scan, if_neg (by decide), innerCandidates5_43_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_44
    (if 3 ≤ innerCandidates5_43.length then search edge 3 [19, 5659] innerCandidates5_43 5678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_44

def innerTail5_42 : List Nat := 5077 :: innerTail5_43
def innerCandidates5_42 : List Nat := [6043]
set_option maxRecDepth 4096 in
theorem innerCandidates5_42_checked : candidates edge 4 19 5077 innerTail5_43 best.bound = innerCandidates5_42 := by
  unfold candidates
  have row : edge 5077 = SearchData.row5077.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_42 : scan edge (search edge 3) 4 [19] 19 innerTail5_42 best = best := by
  rw [innerTail5_42, scan, if_neg (by decide), innerCandidates5_42_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_43
    (if 3 ≤ innerCandidates5_42.length then search edge 3 [19, 5077] innerCandidates5_42 5096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_43

def innerTail5_41 : List Nat := 4861 :: innerTail5_42
def innerCandidates5_41 : List Nat := [5659, 7573]
set_option maxRecDepth 4096 in
theorem innerCandidates5_41_checked : candidates edge 4 19 4861 innerTail5_42 best.bound = innerCandidates5_41 := by
  unfold candidates
  have row : edge 4861 = SearchData.row4861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_41 : scan edge (search edge 3) 4 [19] 19 innerTail5_41 best = best := by
  rw [innerTail5_41, scan, if_neg (by decide), innerCandidates5_41_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_42
    (if 3 ≤ innerCandidates5_41.length then search edge 3 [19, 4861] innerCandidates5_41 4880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_42

def innerTail5_40 : List Nat := 4723 :: innerTail5_41
def innerCandidates5_40 : List Nat := [5869]
set_option maxRecDepth 4096 in
theorem innerCandidates5_40_checked : candidates edge 4 19 4723 innerTail5_41 best.bound = innerCandidates5_40 := by
  unfold candidates
  have row : edge 4723 = SearchData.row4723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_40 : scan edge (search edge 3) 4 [19] 19 innerTail5_40 best = best := by
  rw [innerTail5_40, scan, if_neg (by decide), innerCandidates5_40_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_41
    (if 3 ≤ innerCandidates5_40.length then search edge 3 [19, 4723] innerCandidates5_40 4742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_41

def innerTail5_39 : List Nat := 4507 :: innerTail5_40
def innerCandidates5_39 : List Nat := [6709, 7759]
set_option maxRecDepth 4096 in
theorem innerCandidates5_39_checked : candidates edge 4 19 4507 innerTail5_40 best.bound = innerCandidates5_39 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_39 : scan edge (search edge 3) 4 [19] 19 innerTail5_39 best = best := by
  rw [innerTail5_39, scan, if_neg (by decide), innerCandidates5_39_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_40
    (if 3 ≤ innerCandidates5_39.length then search edge 3 [19, 4507] innerCandidates5_39 4526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_40

def innerTail5_38 : List Nat := 4057 :: innerTail5_39
def innerCandidates5_38 : List Nat := [5077]
set_option maxRecDepth 4096 in
theorem innerCandidates5_38_checked : candidates edge 4 19 4057 innerTail5_39 best.bound = innerCandidates5_38 := by
  unfold candidates
  have row : edge 4057 = SearchData.row4057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_38 : scan edge (search edge 3) 4 [19] 19 innerTail5_38 best = best := by
  rw [innerTail5_38, scan, if_neg (by decide), innerCandidates5_38_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_39
    (if 3 ≤ innerCandidates5_38.length then search edge 3 [19, 4057] innerCandidates5_38 4076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_39

def innerTail5_37 : List Nat := 3943 :: innerTail5_38
def innerCandidates5_37 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates5_37_checked : candidates edge 4 19 3943 innerTail5_38 best.bound = innerCandidates5_37 := by
  unfold candidates
  have row : edge 3943 = SearchData.row3943.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_37 : scan edge (search edge 3) 4 [19] 19 innerTail5_37 best = best := by
  rw [innerTail5_37, scan, if_neg (by decide), innerCandidates5_37_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_38
    (if 3 ≤ innerCandidates5_37.length then search edge 3 [19, 3943] innerCandidates5_37 3962 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_38

def innerTail5_36 : List Nat := 3847 :: innerTail5_37
def innerCandidates5_36 : List Nat := [5077, 5659, 5737]
set_option maxRecDepth 4096 in
theorem innerCandidates5_36_checked : candidates edge 4 19 3847 innerTail5_37 best.bound = innerCandidates5_36 := by
  unfold candidates
  have row : edge 3847 = SearchData.row3847.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild5_36 : search edge 3 [19, 3847] innerCandidates5_36 3866 best = best := by decide

theorem innerScan5_36 : scan edge (search edge 3) 4 [19] 19 innerTail5_36 best = best := by
  rw [innerTail5_36, scan, if_neg (by decide), innerCandidates5_36_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_37
    (if 3 ≤ innerCandidates5_36.length then search edge 3 [19, 3847] innerCandidates5_36 3866 best else best) = best
  rw [if_pos (by decide), innerChild5_36]
  exact innerScan5_37

def innerTail5_35 : List Nat := 3793 :: innerTail5_36
def innerCandidates5_35 : List Nat := [5659, 6907, 8179]
set_option maxRecDepth 4096 in
theorem innerCandidates5_35_checked : candidates edge 4 19 3793 innerTail5_36 best.bound = innerCandidates5_35 := by
  unfold candidates
  have row : edge 3793 = SearchData.row3793.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild5_35 : search edge 3 [19, 3793] innerCandidates5_35 3812 best = best := by decide

theorem innerScan5_35 : scan edge (search edge 3) 4 [19] 19 innerTail5_35 best = best := by
  rw [innerTail5_35, scan, if_neg (by decide), innerCandidates5_35_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_36
    (if 3 ≤ innerCandidates5_35.length then search edge 3 [19, 3793] innerCandidates5_35 3812 best else best) = best
  rw [if_pos (by decide), innerChild5_35]
  exact innerScan5_36

def innerTail5_34 : List Nat := 3727 :: innerTail5_35
def innerCandidates5_34 : List Nat := [5659, 6961, 7699]
set_option maxRecDepth 4096 in
theorem innerCandidates5_34_checked : candidates edge 4 19 3727 innerTail5_35 best.bound = innerCandidates5_34 := by
  unfold candidates
  have row : edge 3727 = SearchData.row3727.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild5_34 : search edge 3 [19, 3727] innerCandidates5_34 3746 best = best := by decide

theorem innerScan5_34 : scan edge (search edge 3) 4 [19] 19 innerTail5_34 best = best := by
  rw [innerTail5_34, scan, if_neg (by decide), innerCandidates5_34_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_35
    (if 3 ≤ innerCandidates5_34.length then search edge 3 [19, 3727] innerCandidates5_34 3746 best else best) = best
  rw [if_pos (by decide), innerChild5_34]
  exact innerScan5_35

def innerTail5_33 : List Nat := 2791 :: innerTail5_34
def innerCandidates5_33 : List Nat := [6961, 8179]
set_option maxRecDepth 4096 in
theorem innerCandidates5_33_checked : candidates edge 4 19 2791 innerTail5_34 best.bound = innerCandidates5_33 := by
  unfold candidates
  have row : edge 2791 = SearchData.row2791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_33 : scan edge (search edge 3) 4 [19] 19 innerTail5_33 best = best := by
  rw [innerTail5_33, scan, if_neg (by decide), innerCandidates5_33_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_34
    (if 3 ≤ innerCandidates5_33.length then search edge 3 [19, 2791] innerCandidates5_33 2810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_34

def innerTail5_32 : List Nat := 2677 :: innerTail5_33
def innerCandidates5_32 : List Nat := [5737, 6961, 7741]
set_option maxRecDepth 4096 in
theorem innerCandidates5_32_checked : candidates edge 4 19 2677 innerTail5_33 best.bound = innerCandidates5_32 := by
  unfold candidates
  have row : edge 2677 = SearchData.row2677.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild5_32 : search edge 3 [19, 2677] innerCandidates5_32 2696 best = best := by decide

theorem innerScan5_32 : scan edge (search edge 3) 4 [19] 19 innerTail5_32 best = best := by
  rw [innerTail5_32, scan, if_neg (by decide), innerCandidates5_32_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_33
    (if 3 ≤ innerCandidates5_32.length then search edge 3 [19, 2677] innerCandidates5_32 2696 best else best) = best
  rw [if_pos (by decide), innerChild5_32]
  exact innerScan5_33

def innerTail5_31 : List Nat := 2539 :: innerTail5_32
def innerCandidates5_31 : List Nat := [4723, 7159]
set_option maxRecDepth 4096 in
theorem innerCandidates5_31_checked : candidates edge 4 19 2539 innerTail5_32 best.bound = innerCandidates5_31 := by
  unfold candidates
  have row : edge 2539 = PrimePairCache.smallRow2539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_31 : scan edge (search edge 3) 4 [19] 19 innerTail5_31 best = best := by
  rw [innerTail5_31, scan, if_neg (by decide), innerCandidates5_31_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_32
    (if 3 ≤ innerCandidates5_31.length then search edge 3 [19, 2539] innerCandidates5_31 2558 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_32

def innerTail5_30 : List Nat := 2251 :: innerTail5_31
def innerCandidates5_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates5_30_checked : candidates edge 4 19 2251 innerTail5_31 best.bound = innerCandidates5_30 := by
  unfold candidates
  have row : edge 2251 = PrimePairCache.smallRow2251.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_30 : scan edge (search edge 3) 4 [19] 19 innerTail5_30 best = best := by
  rw [innerTail5_30, scan, if_neg (by decide), innerCandidates5_30_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_31
    (if 3 ≤ innerCandidates5_30.length then search edge 3 [19, 2251] innerCandidates5_30 2270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_31

def innerTail5_29 : List Nat := 2239 :: innerTail5_30
def innerCandidates5_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates5_29_checked : candidates edge 4 19 2239 innerTail5_30 best.bound = innerCandidates5_29 := by
  unfold candidates
  have row : edge 2239 = PrimePairCache.smallRow2239.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_29 : scan edge (search edge 3) 4 [19] 19 innerTail5_29 best = best := by
  rw [innerTail5_29, scan, if_neg (by decide), innerCandidates5_29_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_30
    (if 3 ≤ innerCandidates5_29.length then search edge 3 [19, 2239] innerCandidates5_29 2258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_30

def innerTail5_28 : List Nat := 2161 :: innerTail5_29
def innerCandidates5_28 : List Nat := [7759]
set_option maxRecDepth 4096 in
theorem innerCandidates5_28_checked : candidates edge 4 19 2161 innerTail5_29 best.bound = innerCandidates5_28 := by
  unfold candidates
  have row : edge 2161 = PrimePairCache.smallRow2161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_28 : scan edge (search edge 3) 4 [19] 19 innerTail5_28 best = best := by
  rw [innerTail5_28, scan, if_neg (by decide), innerCandidates5_28_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_29
    (if 3 ≤ innerCandidates5_28.length then search edge 3 [19, 2161] innerCandidates5_28 2180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_29

def innerTail5_27 : List Nat := 2113 :: innerTail5_28
def innerCandidates5_27 : List Nat := [3727, 4507]
set_option maxRecDepth 4096 in
theorem innerCandidates5_27_checked : candidates edge 4 19 2113 innerTail5_28 best.bound = innerCandidates5_27 := by
  unfold candidates
  have row : edge 2113 = PrimePairCache.smallRow2113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_27 : scan edge (search edge 3) 4 [19] 19 innerTail5_27 best = best := by
  rw [innerTail5_27, scan, if_neg (by decide), innerCandidates5_27_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_28
    (if 3 ≤ innerCandidates5_27.length then search edge 3 [19, 2113] innerCandidates5_27 2132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_28

def innerTail5_26 : List Nat := 2053 :: innerTail5_27
def innerCandidates5_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates5_26_checked : candidates edge 4 19 2053 innerTail5_27 best.bound = innerCandidates5_26 := by
  unfold candidates
  have row : edge 2053 = PrimePairCache.smallRow2053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_26 : scan edge (search edge 3) 4 [19] 19 innerTail5_26 best = best := by
  rw [innerTail5_26, scan, if_neg (by decide), innerCandidates5_26_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_27
    (if 3 ≤ innerCandidates5_26.length then search edge 3 [19, 2053] innerCandidates5_26 2072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_27

def innerTail5_25 : List Nat := 1861 :: innerTail5_26
def innerCandidates5_25 : List Nat := [2791, 6907]
set_option maxRecDepth 4096 in
theorem innerCandidates5_25_checked : candidates edge 4 19 1861 innerTail5_26 best.bound = innerCandidates5_25 := by
  unfold candidates
  have row : edge 1861 = PrimePairCache.smallRow1861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_25 : scan edge (search edge 3) 4 [19] 19 innerTail5_25 best = best := by
  rw [innerTail5_25, scan, if_neg (by decide), innerCandidates5_25_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_26
    (if 3 ≤ innerCandidates5_25.length then search edge 3 [19, 1861] innerCandidates5_25 1880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_26

def innerTail5_24 : List Nat := 1831 :: innerTail5_25
def innerCandidates5_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates5_24_checked : candidates edge 4 19 1831 innerTail5_25 best.bound = innerCandidates5_24 := by
  unfold candidates
  have row : edge 1831 = PrimePairCache.smallRow1831.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_24 : scan edge (search edge 3) 4 [19] 19 innerTail5_24 best = best := by
  rw [innerTail5_24, scan, if_neg (by decide), innerCandidates5_24_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_25
    (if 3 ≤ innerCandidates5_24.length then search edge 3 [19, 1831] innerCandidates5_24 1850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_25

def innerTail5_23 : List Nat := 1699 :: innerTail5_24
def innerCandidates5_23 : List Nat := [1861, 5659, 5791, 7507]
set_option maxRecDepth 4096 in
theorem innerCandidates5_23_checked : candidates edge 4 19 1699 innerTail5_24 best.bound = innerCandidates5_23 := by
  unfold candidates
  have row : edge 1699 = PrimePairCache.smallRow1699.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild5_23 : search edge 3 [19, 1699] innerCandidates5_23 1718 best = best := by decide

theorem innerScan5_23 : scan edge (search edge 3) 4 [19] 19 innerTail5_23 best = best := by
  rw [innerTail5_23, scan, if_neg (by decide), innerCandidates5_23_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_24
    (if 3 ≤ innerCandidates5_23.length then search edge 3 [19, 1699] innerCandidates5_23 1718 best else best) = best
  rw [if_pos (by decide), innerChild5_23]
  exact innerScan5_24

def innerTail5_22 : List Nat := 1693 :: innerTail5_23
def innerCandidates5_22 : List Nat := [6991]
set_option maxRecDepth 4096 in
theorem innerCandidates5_22_checked : candidates edge 4 19 1693 innerTail5_23 best.bound = innerCandidates5_22 := by
  unfold candidates
  have row : edge 1693 = PrimePairCache.smallRow1693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_22 : scan edge (search edge 3) 4 [19] 19 innerTail5_22 best = best := by
  rw [innerTail5_22, scan, if_neg (by decide), innerCandidates5_22_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_23
    (if 3 ≤ innerCandidates5_22.length then search edge 3 [19, 1693] innerCandidates5_22 1712 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_23

def innerTail5_21 : List Nat := 1669 :: innerTail5_22
def innerCandidates5_21 : List Nat := [2791, 5659, 6661, 6709]
set_option maxRecDepth 4096 in
theorem innerCandidates5_21_checked : candidates edge 4 19 1669 innerTail5_22 best.bound = innerCandidates5_21 := by
  unfold candidates
  have row : edge 1669 = PrimePairCache.smallRow1669.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild5_21 : search edge 3 [19, 1669] innerCandidates5_21 1688 best = best := by decide

theorem innerScan5_21 : scan edge (search edge 3) 4 [19] 19 innerTail5_21 best = best := by
  rw [innerTail5_21, scan, if_neg (by decide), innerCandidates5_21_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_22
    (if 3 ≤ innerCandidates5_21.length then search edge 3 [19, 1669] innerCandidates5_21 1688 best else best) = best
  rw [if_pos (by decide), innerChild5_21]
  exact innerScan5_22

def innerTail5_20 : List Nat := 1657 :: innerTail5_21
def innerCandidates5_20 : List Nat := [1699, 2791, 6907]
set_option maxRecDepth 4096 in
theorem innerCandidates5_20_checked : candidates edge 4 19 1657 innerTail5_21 best.bound = innerCandidates5_20 := by
  unfold candidates
  have row : edge 1657 = PrimePairCache.smallRow1657.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild5_20 : search edge 3 [19, 1657] innerCandidates5_20 1676 best = best := by decide

theorem innerScan5_20 : scan edge (search edge 3) 4 [19] 19 innerTail5_20 best = best := by
  rw [innerTail5_20, scan, if_neg (by decide), innerCandidates5_20_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_21
    (if 3 ≤ innerCandidates5_20.length then search edge 3 [19, 1657] innerCandidates5_20 1676 best else best) = best
  rw [if_pos (by decide), innerChild5_20]
  exact innerScan5_21

def innerTail5_19 : List Nat := 1621 :: innerTail5_20
def innerCandidates5_19 : List Nat := [3943, 6073]
set_option maxRecDepth 4096 in
theorem innerCandidates5_19_checked : candidates edge 4 19 1621 innerTail5_20 best.bound = innerCandidates5_19 := by
  unfold candidates
  have row : edge 1621 = PrimePairCache.smallRow1621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_19 : scan edge (search edge 3) 4 [19] 19 innerTail5_19 best = best := by
  rw [innerTail5_19, scan, if_neg (by decide), innerCandidates5_19_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_20
    (if 3 ≤ innerCandidates5_19.length then search edge 3 [19, 1621] innerCandidates5_19 1640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_20

def innerTail5_18 : List Nat := 1447 :: innerTail5_19
def innerCandidates5_18 : List Nat := [6991, 7759]
set_option maxRecDepth 4096 in
theorem innerCandidates5_18_checked : candidates edge 4 19 1447 innerTail5_19 best.bound = innerCandidates5_18 := by
  unfold candidates
  have row : edge 1447 = PrimePairCache.smallRow1447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_18 : scan edge (search edge 3) 4 [19] 19 innerTail5_18 best = best := by
  rw [innerTail5_18, scan, if_neg (by decide), innerCandidates5_18_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_19
    (if 3 ≤ innerCandidates5_18.length then search edge 3 [19, 1447] innerCandidates5_18 1466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_19

def innerTail5_17 : List Nat := 1297 :: innerTail5_18
def innerCandidates5_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates5_17_checked : candidates edge 4 19 1297 innerTail5_18 best.bound = innerCandidates5_17 := by
  unfold candidates
  have row : edge 1297 = PrimePairCache.smallRow1297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_17 : scan edge (search edge 3) 4 [19] 19 innerTail5_17 best = best := by
  rw [innerTail5_17, scan, if_neg (by decide), innerCandidates5_17_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_18
    (if 3 ≤ innerCandidates5_17.length then search edge 3 [19, 1297] innerCandidates5_17 1316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_18

def innerTail5_16 : List Nat := 1249 :: innerTail5_17
def innerCandidates5_16 : List Nat := [3727]
set_option maxRecDepth 4096 in
theorem innerCandidates5_16_checked : candidates edge 4 19 1249 innerTail5_17 best.bound = innerCandidates5_16 := by
  unfold candidates
  have row : edge 1249 = PrimePairCache.smallRow1249.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_16 : scan edge (search edge 3) 4 [19] 19 innerTail5_16 best = best := by
  rw [innerTail5_16, scan, if_neg (by decide), innerCandidates5_16_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_17
    (if 3 ≤ innerCandidates5_16.length then search edge 3 [19, 1249] innerCandidates5_16 1268 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_17

def innerTail5_15 : List Nat := 1237 :: innerTail5_16
def innerCandidates5_15 : List Nat := [1657, 6271]
set_option maxRecDepth 4096 in
theorem innerCandidates5_15_checked : candidates edge 4 19 1237 innerTail5_16 best.bound = innerCandidates5_15 := by
  unfold candidates
  have row : edge 1237 = PrimePairCache.smallRow1237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_15 : scan edge (search edge 3) 4 [19] 19 innerTail5_15 best = best := by
  rw [innerTail5_15, scan, if_neg (by decide), innerCandidates5_15_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_16
    (if 3 ≤ innerCandidates5_15.length then search edge 3 [19, 1237] innerCandidates5_15 1256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_16

def innerTail5_14 : List Nat := 1039 :: innerTail5_15
def innerCandidates5_14 : List Nat := [4723, 6073, 6709]
set_option maxRecDepth 4096 in
theorem innerCandidates5_14_checked : candidates edge 4 19 1039 innerTail5_15 best.bound = innerCandidates5_14 := by
  unfold candidates
  have row : edge 1039 = PrimePairCache.smallRow1039.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild5_14 : search edge 3 [19, 1039] innerCandidates5_14 1058 best = best := by decide

theorem innerScan5_14 : scan edge (search edge 3) 4 [19] 19 innerTail5_14 best = best := by
  rw [innerTail5_14, scan, if_neg (by decide), innerCandidates5_14_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_15
    (if 3 ≤ innerCandidates5_14.length then search edge 3 [19, 1039] innerCandidates5_14 1058 best else best) = best
  rw [if_pos (by decide), innerChild5_14]
  exact innerScan5_15

def innerTail5_13 : List Nat := 1033 :: innerTail5_14
def innerCandidates5_13 : List Nat := [1249, 5869, 8179]
set_option maxRecDepth 4096 in
theorem innerCandidates5_13_checked : candidates edge 4 19 1033 innerTail5_14 best.bound = innerCandidates5_13 := by
  unfold candidates
  have row : edge 1033 = PrimePairCache.smallRow1033.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild5_13 : search edge 3 [19, 1033] innerCandidates5_13 1052 best = best := by decide

theorem innerScan5_13 : scan edge (search edge 3) 4 [19] 19 innerTail5_13 best = best := by
  rw [innerTail5_13, scan, if_neg (by decide), innerCandidates5_13_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_14
    (if 3 ≤ innerCandidates5_13.length then search edge 3 [19, 1033] innerCandidates5_13 1052 best else best) = best
  rw [if_pos (by decide), innerChild5_13]
  exact innerScan5_14

def innerTail5_12 : List Nat := 997 :: innerTail5_13
def innerCandidates5_12 : List Nat := [7741]
set_option maxRecDepth 4096 in
theorem innerCandidates5_12_checked : candidates edge 4 19 997 innerTail5_13 best.bound = innerCandidates5_12 := by
  unfold candidates
  have row : edge 997 = PrimePairCache.smallRow997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_12 : scan edge (search edge 3) 4 [19] 19 innerTail5_12 best = best := by
  rw [innerTail5_12, scan, if_neg (by decide), innerCandidates5_12_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_13
    (if 3 ≤ innerCandidates5_12.length then search edge 3 [19, 997] innerCandidates5_12 1016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_13

def innerTail5_11 : List Nat := 991 :: innerTail5_12
def innerCandidates5_11 : List Nat := [1861, 3727, 7759]
set_option maxRecDepth 4096 in
theorem innerCandidates5_11_checked : candidates edge 4 19 991 innerTail5_12 best.bound = innerCandidates5_11 := by
  unfold candidates
  have row : edge 991 = PrimePairCache.smallRow991.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild5_11 : search edge 3 [19, 991] innerCandidates5_11 1010 best = best := by decide

theorem innerScan5_11 : scan edge (search edge 3) 4 [19] 19 innerTail5_11 best = best := by
  rw [innerTail5_11, scan, if_neg (by decide), innerCandidates5_11_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_12
    (if 3 ≤ innerCandidates5_11.length then search edge 3 [19, 991] innerCandidates5_11 1010 best else best) = best
  rw [if_pos (by decide), innerChild5_11]
  exact innerScan5_12

def innerTail5_10 : List Nat := 937 :: innerTail5_11
def innerCandidates5_10 : List Nat := [2053, 2791, 3943, 5791]
set_option maxRecDepth 4096 in
theorem innerCandidates5_10_checked : candidates edge 4 19 937 innerTail5_11 best.bound = innerCandidates5_10 := by
  unfold candidates
  have row : edge 937 = PrimePairCache.smallRow937.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild5_10 : search edge 3 [19, 937] innerCandidates5_10 956 best = best := by decide

theorem innerScan5_10 : scan edge (search edge 3) 4 [19] 19 innerTail5_10 best = best := by
  rw [innerTail5_10, scan, if_neg (by decide), innerCandidates5_10_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_11
    (if 3 ≤ innerCandidates5_10.length then search edge 3 [19, 937] innerCandidates5_10 956 best else best) = best
  rw [if_pos (by decide), innerChild5_10]
  exact innerScan5_11

def innerTail5_9 : List Nat := 727 :: innerTail5_10
def innerCandidates5_9 : List Nat := [997, 1237, 1669]
set_option maxRecDepth 4096 in
theorem innerCandidates5_9_checked : candidates edge 4 19 727 innerTail5_10 best.bound = innerCandidates5_9 := by
  unfold candidates
  have row : edge 727 = PrimePairCache.smallRow727.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild5_9 : search edge 3 [19, 727] innerCandidates5_9 746 best = best := by decide

theorem innerScan5_9 : scan edge (search edge 3) 4 [19] 19 innerTail5_9 best = best := by
  rw [innerTail5_9, scan, if_neg (by decide), innerCandidates5_9_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_10
    (if 3 ≤ innerCandidates5_9.length then search edge 3 [19, 727] innerCandidates5_9 746 best else best) = best
  rw [if_pos (by decide), innerChild5_9]
  exact innerScan5_10

def innerTail5_8 : List Nat := 709 :: innerTail5_9
def innerCandidates5_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates5_8_checked : candidates edge 4 19 709 innerTail5_9 best.bound = innerCandidates5_8 := by
  unfold candidates
  have row : edge 709 = PrimePairCache.smallRow709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_8 : scan edge (search edge 3) 4 [19] 19 innerTail5_8 best = best := by
  rw [innerTail5_8, scan, if_neg (by decide), innerCandidates5_8_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_9
    (if 3 ≤ innerCandidates5_8.length then search edge 3 [19, 709] innerCandidates5_8 728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_9

def innerTail5_7 : List Nat := 577 :: innerTail5_8
def innerCandidates5_7 : List Nat := [937, 2539]
set_option maxRecDepth 4096 in
theorem innerCandidates5_7_checked : candidates edge 4 19 577 innerTail5_8 best.bound = innerCandidates5_7 := by
  unfold candidates
  have row : edge 577 = PrimePairCache.smallRow577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_7 : scan edge (search edge 3) 4 [19] 19 innerTail5_7 best = best := by
  rw [innerTail5_7, scan, if_neg (by decide), innerCandidates5_7_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_8
    (if 3 ≤ innerCandidates5_7.length then search edge 3 [19, 577] innerCandidates5_7 596 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_8

def innerTail5_6 : List Nat := 571 :: innerTail5_7
def innerCandidates5_6 : List Nat := [7297]
set_option maxRecDepth 4096 in
theorem innerCandidates5_6_checked : candidates edge 4 19 571 innerTail5_7 best.bound = innerCandidates5_6 := by
  unfold candidates
  have row : edge 571 = PrimePairCache.smallRow571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_6 : scan edge (search edge 3) 4 [19] 19 innerTail5_6 best = best := by
  rw [innerTail5_6, scan, if_neg (by decide), innerCandidates5_6_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_7
    (if 3 ≤ innerCandidates5_6.length then search edge 3 [19, 571] innerCandidates5_6 590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_7

def innerTail5_5 : List Nat := 433 :: innerTail5_6
def innerCandidates5_5 : List Nat := [571]
set_option maxRecDepth 4096 in
theorem innerCandidates5_5_checked : candidates edge 4 19 433 innerTail5_6 best.bound = innerCandidates5_5 := by
  unfold candidates
  have row : edge 433 = PrimePairCache.smallRow433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_5 : scan edge (search edge 3) 4 [19] 19 innerTail5_5 best = best := by
  rw [innerTail5_5, scan, if_neg (by decide), innerCandidates5_5_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_6
    (if 3 ≤ innerCandidates5_5.length then search edge 3 [19, 433] innerCandidates5_5 452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_6

def innerTail5_4 : List Nat := 181 :: innerTail5_5
def innerCandidates5_4 : List Nat := [1657]
set_option maxRecDepth 4096 in
theorem innerCandidates5_4_checked : candidates edge 4 19 181 innerTail5_5 best.bound = innerCandidates5_4 := by
  unfold candidates
  have row : edge 181 = PrimePairCache.smallRow181.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_4 : scan edge (search edge 3) 4 [19] 19 innerTail5_4 best = best := by
  rw [innerTail5_4, scan, if_neg (by decide), innerCandidates5_4_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_5
    (if 3 ≤ innerCandidates5_4.length then search edge 3 [19, 181] innerCandidates5_4 200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_5

def innerTail5_3 : List Nat := 163 :: innerTail5_4
def innerCandidates5_3 : List Nat := [997]
set_option maxRecDepth 4096 in
theorem innerCandidates5_3_checked : candidates edge 4 19 163 innerTail5_4 best.bound = innerCandidates5_3 := by
  unfold candidates
  have row : edge 163 = PrimePairCache.smallRow163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan5_3 : scan edge (search edge 3) 4 [19] 19 innerTail5_3 best = best := by
  rw [innerTail5_3, scan, if_neg (by decide), innerCandidates5_3_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_4
    (if 3 ≤ innerCandidates5_3.length then search edge 3 [19, 163] innerCandidates5_3 182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan5_4

def innerTail5_2 : List Nat := 97 :: innerTail5_3
def innerCandidates5_2 : List Nat := [1039, 3727, 4507, 6271, 6991, 7507]
set_option maxRecDepth 4096 in
theorem innerCandidates5_2_checked : candidates edge 4 19 97 innerTail5_3 best.bound = innerCandidates5_2 := by
  unfold candidates
  have row : edge 97 = PrimePairCache.smallRow97.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild5_2 : search edge 3 [19, 97] innerCandidates5_2 116 best = best := by decide

theorem innerScan5_2 : scan edge (search edge 3) 4 [19] 19 innerTail5_2 best = best := by
  rw [innerTail5_2, scan, if_neg (by decide), innerCandidates5_2_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_3
    (if 3 ≤ innerCandidates5_2.length then search edge 3 [19, 97] innerCandidates5_2 116 best else best) = best
  rw [if_pos (by decide), innerChild5_2]
  exact innerScan5_3

def innerTail5_1 : List Nat := 79 :: innerTail5_2
def innerCandidates5_1 : List Nat := [1447, 1657, 5077, 5659, 5791, 8179]
set_option maxRecDepth 4096 in
theorem innerCandidates5_1_checked : candidates edge 4 19 79 innerTail5_2 best.bound = innerCandidates5_1 := by
  unfold candidates
  have row : edge 79 = PrimePairCache.smallRow79.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild5_1 : search edge 3 [19, 79] innerCandidates5_1 98 best = best := by decide

theorem innerScan5_1 : scan edge (search edge 3) 4 [19] 19 innerTail5_1 best = best := by
  rw [innerTail5_1, scan, if_neg (by decide), innerCandidates5_1_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_2
    (if 3 ≤ innerCandidates5_1.length then search edge 3 [19, 79] innerCandidates5_1 98 best else best) = best
  rw [if_pos (by decide), innerChild5_1]
  exact innerScan5_2

def innerTail5_0 : List Nat := 31 :: innerTail5_1
def innerCandidates5_0 : List Nat := [181, 991, 1237, 1447, 4723, 6073, 6271, 7741]
set_option maxRecDepth 4096 in
theorem innerCandidates5_0_checked : candidates edge 4 19 31 innerTail5_1 best.bound = innerCandidates5_0 := by
  unfold candidates
  have row : edge 31 = PrimePairCache.smallRow31.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild5_0 : search edge 3 [19, 31] innerCandidates5_0 50 best = best := by decide

theorem innerScan5_0 : scan edge (search edge 3) 4 [19] 19 innerTail5_0 best = best := by
  rw [innerTail5_0, scan, if_neg (by decide), innerCandidates5_0_checked]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_1
    (if 3 ≤ innerCandidates5_0.length then search edge 3 [19, 31] innerCandidates5_0 50 best else best) = best
  rw [if_pos (by decide), innerChild5_0]
  exact innerScan5_1

theorem rootBranch5 : search edge 4 [19] rootCandidates5 19 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [19] 19 innerTail5_0 best = best
  exact innerScan5_0

end ProjectEulerSolutions.PrimePairUpper.Search
