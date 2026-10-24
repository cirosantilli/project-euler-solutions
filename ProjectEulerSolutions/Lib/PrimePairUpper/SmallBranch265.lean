import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates2 : List Nat := [23, 113, 239, 251, 353, 503, 587, 701, 743, 839, 863, 941, 953, 1049, 1091, 1103, 1217, 1427, 1487, 1847, 1871, 1949, 1973, 2069, 2213, 2237, 2297, 2339, 2459, 2621, 2657, 2843, 2927, 3083, 3209, 3467, 3539, 3779, 4013, 4073, 4217, 4229, 4259, 4493, 4547, 4643, 4967, 5099, 5153, 5309, 5783, 5807, 5849, 5861, 5987, 6047, 6329, 6719, 6791, 6911, 7109, 7127, 7331, 7517, 7529, 7541, 7577, 7703, 7757, 7877, 7937, 8081, 8171, 8219, 8273, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates2_checked : candidates edge 5 0 11 tail3 best.bound = rootCandidates2 := by
  unfold candidates
  have row : edge 11 = PrimePairCache.smallRow11.contains := by funext q; rfl
  rw [row]
  decide

def innerTail2_76 : List Nat := []
theorem innerScan2_76 : scan edge (search edge 3) 4 [11] 11 innerTail2_76 best = best := rfl

def innerTail2_75 : List Nat := 8387 :: innerTail2_76
def innerCandidates2_75 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_75_checked : candidates edge 4 11 8387 innerTail2_76 best.bound = innerCandidates2_75 := by
  unfold candidates
  have row : edge 8387 = SearchData.row8387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_75 : scan edge (search edge 3) 4 [11] 11 innerTail2_75 best = best := by
  rw [innerTail2_75, scan, if_neg (by decide), innerCandidates2_75_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_76
    (if 3 ≤ innerCandidates2_75.length then search edge 3 [11, 8387] innerCandidates2_75 8398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_76

def innerTail2_74 : List Nat := 8273 :: innerTail2_75
def innerCandidates2_74 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_74_checked : candidates edge 4 11 8273 innerTail2_75 best.bound = innerCandidates2_74 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_74 : scan edge (search edge 3) 4 [11] 11 innerTail2_74 best = best := by
  rw [innerTail2_74, scan, if_neg (by decide), innerCandidates2_74_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_75
    (if 3 ≤ innerCandidates2_74.length then search edge 3 [11, 8273] innerCandidates2_74 8284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_75

def innerTail2_73 : List Nat := 8219 :: innerTail2_74
def innerCandidates2_73 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_73_checked : candidates edge 4 11 8219 innerTail2_74 best.bound = innerCandidates2_73 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_73 : scan edge (search edge 3) 4 [11] 11 innerTail2_73 best = best := by
  rw [innerTail2_73, scan, if_neg (by decide), innerCandidates2_73_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_74
    (if 3 ≤ innerCandidates2_73.length then search edge 3 [11, 8219] innerCandidates2_73 8230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_74

def innerTail2_72 : List Nat := 8171 :: innerTail2_73
def innerCandidates2_72 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_72_checked : candidates edge 4 11 8171 innerTail2_73 best.bound = innerCandidates2_72 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_72 : scan edge (search edge 3) 4 [11] 11 innerTail2_72 best = best := by
  rw [innerTail2_72, scan, if_neg (by decide), innerCandidates2_72_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_73
    (if 3 ≤ innerCandidates2_72.length then search edge 3 [11, 8171] innerCandidates2_72 8182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_73

def innerTail2_71 : List Nat := 8081 :: innerTail2_72
def innerCandidates2_71 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_71_checked : candidates edge 4 11 8081 innerTail2_72 best.bound = innerCandidates2_71 := by
  unfold candidates
  have row : edge 8081 = SearchData.row8081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_71 : scan edge (search edge 3) 4 [11] 11 innerTail2_71 best = best := by
  rw [innerTail2_71, scan, if_neg (by decide), innerCandidates2_71_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_72
    (if 3 ≤ innerCandidates2_71.length then search edge 3 [11, 8081] innerCandidates2_71 8092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_72

def innerTail2_70 : List Nat := 7937 :: innerTail2_71
def innerCandidates2_70 : List Nat := [8081]
set_option maxRecDepth 4096 in
theorem innerCandidates2_70_checked : candidates edge 4 11 7937 innerTail2_71 best.bound = innerCandidates2_70 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_70 : scan edge (search edge 3) 4 [11] 11 innerTail2_70 best = best := by
  rw [innerTail2_70, scan, if_neg (by decide), innerCandidates2_70_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_71
    (if 3 ≤ innerCandidates2_70.length then search edge 3 [11, 7937] innerCandidates2_70 7948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_71

def innerTail2_69 : List Nat := 7877 :: innerTail2_70
def innerCandidates2_69 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_69_checked : candidates edge 4 11 7877 innerTail2_70 best.bound = innerCandidates2_69 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_69 : scan edge (search edge 3) 4 [11] 11 innerTail2_69 best = best := by
  rw [innerTail2_69, scan, if_neg (by decide), innerCandidates2_69_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_70
    (if 3 ≤ innerCandidates2_69.length then search edge 3 [11, 7877] innerCandidates2_69 7888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_70

def innerTail2_68 : List Nat := 7757 :: innerTail2_69
def innerCandidates2_68 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_68_checked : candidates edge 4 11 7757 innerTail2_69 best.bound = innerCandidates2_68 := by
  unfold candidates
  have row : edge 7757 = SearchData.row7757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_68 : scan edge (search edge 3) 4 [11] 11 innerTail2_68 best = best := by
  rw [innerTail2_68, scan, if_neg (by decide), innerCandidates2_68_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_69
    (if 3 ≤ innerCandidates2_68.length then search edge 3 [11, 7757] innerCandidates2_68 7768 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_69

def innerTail2_67 : List Nat := 7703 :: innerTail2_68
def innerCandidates2_67 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_67_checked : candidates edge 4 11 7703 innerTail2_68 best.bound = innerCandidates2_67 := by
  unfold candidates
  have row : edge 7703 = SearchData.row7703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_67 : scan edge (search edge 3) 4 [11] 11 innerTail2_67 best = best := by
  rw [innerTail2_67, scan, if_neg (by decide), innerCandidates2_67_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_68
    (if 3 ≤ innerCandidates2_67.length then search edge 3 [11, 7703] innerCandidates2_67 7714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_68

def innerTail2_66 : List Nat := 7577 :: innerTail2_67
def innerCandidates2_66 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_66_checked : candidates edge 4 11 7577 innerTail2_67 best.bound = innerCandidates2_66 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_66 : scan edge (search edge 3) 4 [11] 11 innerTail2_66 best = best := by
  rw [innerTail2_66, scan, if_neg (by decide), innerCandidates2_66_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_67
    (if 3 ≤ innerCandidates2_66.length then search edge 3 [11, 7577] innerCandidates2_66 7588 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_67

def innerTail2_65 : List Nat := 7541 :: innerTail2_66
def innerCandidates2_65 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_65_checked : candidates edge 4 11 7541 innerTail2_66 best.bound = innerCandidates2_65 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_65 : scan edge (search edge 3) 4 [11] 11 innerTail2_65 best = best := by
  rw [innerTail2_65, scan, if_neg (by decide), innerCandidates2_65_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_66
    (if 3 ≤ innerCandidates2_65.length then search edge 3 [11, 7541] innerCandidates2_65 7552 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_66

def innerTail2_64 : List Nat := 7529 :: innerTail2_65
def innerCandidates2_64 : List Nat := [7877]
set_option maxRecDepth 4096 in
theorem innerCandidates2_64_checked : candidates edge 4 11 7529 innerTail2_65 best.bound = innerCandidates2_64 := by
  unfold candidates
  have row : edge 7529 = SearchData.row7529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_64 : scan edge (search edge 3) 4 [11] 11 innerTail2_64 best = best := by
  rw [innerTail2_64, scan, if_neg (by decide), innerCandidates2_64_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_65
    (if 3 ≤ innerCandidates2_64.length then search edge 3 [11, 7529] innerCandidates2_64 7540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_65

def innerTail2_63 : List Nat := 7517 :: innerTail2_64
def innerCandidates2_63 : List Nat := [8387]
set_option maxRecDepth 4096 in
theorem innerCandidates2_63_checked : candidates edge 4 11 7517 innerTail2_64 best.bound = innerCandidates2_63 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_63 : scan edge (search edge 3) 4 [11] 11 innerTail2_63 best = best := by
  rw [innerTail2_63, scan, if_neg (by decide), innerCandidates2_63_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_64
    (if 3 ≤ innerCandidates2_63.length then search edge 3 [11, 7517] innerCandidates2_63 7528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_64

def innerTail2_62 : List Nat := 7331 :: innerTail2_63
def innerCandidates2_62 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_62_checked : candidates edge 4 11 7331 innerTail2_63 best.bound = innerCandidates2_62 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_62 : scan edge (search edge 3) 4 [11] 11 innerTail2_62 best = best := by
  rw [innerTail2_62, scan, if_neg (by decide), innerCandidates2_62_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_63
    (if 3 ≤ innerCandidates2_62.length then search edge 3 [11, 7331] innerCandidates2_62 7342 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_63

def innerTail2_61 : List Nat := 7127 :: innerTail2_62
def innerCandidates2_61 : List Nat := [7757]
set_option maxRecDepth 4096 in
theorem innerCandidates2_61_checked : candidates edge 4 11 7127 innerTail2_62 best.bound = innerCandidates2_61 := by
  unfold candidates
  have row : edge 7127 = SearchData.row7127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_61 : scan edge (search edge 3) 4 [11] 11 innerTail2_61 best = best := by
  rw [innerTail2_61, scan, if_neg (by decide), innerCandidates2_61_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_62
    (if 3 ≤ innerCandidates2_61.length then search edge 3 [11, 7127] innerCandidates2_61 7138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_62

def innerTail2_60 : List Nat := 7109 :: innerTail2_61
def innerCandidates2_60 : List Nat := [7127]
set_option maxRecDepth 4096 in
theorem innerCandidates2_60_checked : candidates edge 4 11 7109 innerTail2_61 best.bound = innerCandidates2_60 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_60 : scan edge (search edge 3) 4 [11] 11 innerTail2_60 best = best := by
  rw [innerTail2_60, scan, if_neg (by decide), innerCandidates2_60_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_61
    (if 3 ≤ innerCandidates2_60.length then search edge 3 [11, 7109] innerCandidates2_60 7120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_61

def innerTail2_59 : List Nat := 6911 :: innerTail2_60
def innerCandidates2_59 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_59_checked : candidates edge 4 11 6911 innerTail2_60 best.bound = innerCandidates2_59 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_59 : scan edge (search edge 3) 4 [11] 11 innerTail2_59 best = best := by
  rw [innerTail2_59, scan, if_neg (by decide), innerCandidates2_59_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_60
    (if 3 ≤ innerCandidates2_59.length then search edge 3 [11, 6911] innerCandidates2_59 6922 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_60

def innerTail2_58 : List Nat := 6791 :: innerTail2_59
def innerCandidates2_58 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_58_checked : candidates edge 4 11 6791 innerTail2_59 best.bound = innerCandidates2_58 := by
  unfold candidates
  have row : edge 6791 = SearchData.row6791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_58 : scan edge (search edge 3) 4 [11] 11 innerTail2_58 best = best := by
  rw [innerTail2_58, scan, if_neg (by decide), innerCandidates2_58_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_59
    (if 3 ≤ innerCandidates2_58.length then search edge 3 [11, 6791] innerCandidates2_58 6802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_59

def innerTail2_57 : List Nat := 6719 :: innerTail2_58
def innerCandidates2_57 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_57_checked : candidates edge 4 11 6719 innerTail2_58 best.bound = innerCandidates2_57 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_57 : scan edge (search edge 3) 4 [11] 11 innerTail2_57 best = best := by
  rw [innerTail2_57, scan, if_neg (by decide), innerCandidates2_57_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_58
    (if 3 ≤ innerCandidates2_57.length then search edge 3 [11, 6719] innerCandidates2_57 6730 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_58

def innerTail2_56 : List Nat := 6329 :: innerTail2_57
def innerCandidates2_56 : List Nat := [7331, 7937]
set_option maxRecDepth 4096 in
theorem innerCandidates2_56_checked : candidates edge 4 11 6329 innerTail2_57 best.bound = innerCandidates2_56 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_56 : scan edge (search edge 3) 4 [11] 11 innerTail2_56 best = best := by
  rw [innerTail2_56, scan, if_neg (by decide), innerCandidates2_56_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_57
    (if 3 ≤ innerCandidates2_56.length then search edge 3 [11, 6329] innerCandidates2_56 6340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_57

def innerTail2_55 : List Nat := 6047 :: innerTail2_56
def innerCandidates2_55 : List Nat := [6719]
set_option maxRecDepth 4096 in
theorem innerCandidates2_55_checked : candidates edge 4 11 6047 innerTail2_56 best.bound = innerCandidates2_55 := by
  unfold candidates
  have row : edge 6047 = SearchData.row6047.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_55 : scan edge (search edge 3) 4 [11] 11 innerTail2_55 best = best := by
  rw [innerTail2_55, scan, if_neg (by decide), innerCandidates2_55_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_56
    (if 3 ≤ innerCandidates2_55.length then search edge 3 [11, 6047] innerCandidates2_55 6058 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_56

def innerTail2_54 : List Nat := 5987 :: innerTail2_55
def innerCandidates2_54 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_54_checked : candidates edge 4 11 5987 innerTail2_55 best.bound = innerCandidates2_54 := by
  unfold candidates
  have row : edge 5987 = SearchData.row5987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_54 : scan edge (search edge 3) 4 [11] 11 innerTail2_54 best = best := by
  rw [innerTail2_54, scan, if_neg (by decide), innerCandidates2_54_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_55
    (if 3 ≤ innerCandidates2_54.length then search edge 3 [11, 5987] innerCandidates2_54 5998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_55

def innerTail2_53 : List Nat := 5861 :: innerTail2_54
def innerCandidates2_53 : List Nat := [5987, 8171]
set_option maxRecDepth 4096 in
theorem innerCandidates2_53_checked : candidates edge 4 11 5861 innerTail2_54 best.bound = innerCandidates2_53 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_53 : scan edge (search edge 3) 4 [11] 11 innerTail2_53 best = best := by
  rw [innerTail2_53, scan, if_neg (by decide), innerCandidates2_53_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_54
    (if 3 ≤ innerCandidates2_53.length then search edge 3 [11, 5861] innerCandidates2_53 5872 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_54

def innerTail2_52 : List Nat := 5849 :: innerTail2_53
def innerCandidates2_52 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_52_checked : candidates edge 4 11 5849 innerTail2_53 best.bound = innerCandidates2_52 := by
  unfold candidates
  have row : edge 5849 = SearchData.row5849.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_52 : scan edge (search edge 3) 4 [11] 11 innerTail2_52 best = best := by
  rw [innerTail2_52, scan, if_neg (by decide), innerCandidates2_52_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_53
    (if 3 ≤ innerCandidates2_52.length then search edge 3 [11, 5849] innerCandidates2_52 5860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_53

def innerTail2_51 : List Nat := 5807 :: innerTail2_52
def innerCandidates2_51 : List Nat := [6329, 7577]
set_option maxRecDepth 4096 in
theorem innerCandidates2_51_checked : candidates edge 4 11 5807 innerTail2_52 best.bound = innerCandidates2_51 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_51 : scan edge (search edge 3) 4 [11] 11 innerTail2_51 best = best := by
  rw [innerTail2_51, scan, if_neg (by decide), innerCandidates2_51_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_52
    (if 3 ≤ innerCandidates2_51.length then search edge 3 [11, 5807] innerCandidates2_51 5818 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_52

def innerTail2_50 : List Nat := 5783 :: innerTail2_51
def innerCandidates2_50 : List Nat := [6719, 7331]
set_option maxRecDepth 4096 in
theorem innerCandidates2_50_checked : candidates edge 4 11 5783 innerTail2_51 best.bound = innerCandidates2_50 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_50 : scan edge (search edge 3) 4 [11] 11 innerTail2_50 best = best := by
  rw [innerTail2_50, scan, if_neg (by decide), innerCandidates2_50_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_51
    (if 3 ≤ innerCandidates2_50.length then search edge 3 [11, 5783] innerCandidates2_50 5794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_51

def innerTail2_49 : List Nat := 5309 :: innerTail2_50
def innerCandidates2_49 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_49_checked : candidates edge 4 11 5309 innerTail2_50 best.bound = innerCandidates2_49 := by
  unfold candidates
  have row : edge 5309 = SearchData.row5309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_49 : scan edge (search edge 3) 4 [11] 11 innerTail2_49 best = best := by
  rw [innerTail2_49, scan, if_neg (by decide), innerCandidates2_49_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_50
    (if 3 ≤ innerCandidates2_49.length then search edge 3 [11, 5309] innerCandidates2_49 5320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_50

def innerTail2_48 : List Nat := 5153 :: innerTail2_49
def innerCandidates2_48 : List Nat := [5849, 6047, 7529]
set_option maxRecDepth 4096 in
theorem innerCandidates2_48_checked : candidates edge 4 11 5153 innerTail2_49 best.bound = innerCandidates2_48 := by
  unfold candidates
  have row : edge 5153 = SearchData.row5153.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild2_48 : search edge 3 [11, 5153] innerCandidates2_48 5164 best = best := by decide

theorem innerScan2_48 : scan edge (search edge 3) 4 [11] 11 innerTail2_48 best = best := by
  rw [innerTail2_48, scan, if_neg (by decide), innerCandidates2_48_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_49
    (if 3 ≤ innerCandidates2_48.length then search edge 3 [11, 5153] innerCandidates2_48 5164 best else best) = best
  rw [if_pos (by decide), innerChild2_48]
  exact innerScan2_49

def innerTail2_47 : List Nat := 5099 :: innerTail2_48
def innerCandidates2_47 : List Nat := [8273]
set_option maxRecDepth 4096 in
theorem innerCandidates2_47_checked : candidates edge 4 11 5099 innerTail2_48 best.bound = innerCandidates2_47 := by
  unfold candidates
  have row : edge 5099 = SearchData.row5099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_47 : scan edge (search edge 3) 4 [11] 11 innerTail2_47 best = best := by
  rw [innerTail2_47, scan, if_neg (by decide), innerCandidates2_47_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_48
    (if 3 ≤ innerCandidates2_47.length then search edge 3 [11, 5099] innerCandidates2_47 5110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_48

def innerTail2_46 : List Nat := 4967 :: innerTail2_47
def innerCandidates2_46 : List Nat := [5849, 7127, 7877, 8219]
set_option maxRecDepth 4096 in
theorem innerCandidates2_46_checked : candidates edge 4 11 4967 innerTail2_47 best.bound = innerCandidates2_46 := by
  unfold candidates
  have row : edge 4967 = SearchData.row4967.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild2_46 : search edge 3 [11, 4967] innerCandidates2_46 4978 best = best := by decide

theorem innerScan2_46 : scan edge (search edge 3) 4 [11] 11 innerTail2_46 best = best := by
  rw [innerTail2_46, scan, if_neg (by decide), innerCandidates2_46_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_47
    (if 3 ≤ innerCandidates2_46.length then search edge 3 [11, 4967] innerCandidates2_46 4978 best else best) = best
  rw [if_pos (by decide), innerChild2_46]
  exact innerScan2_47

def innerTail2_45 : List Nat := 4643 :: innerTail2_46
def innerCandidates2_45 : List Nat := [5153, 5849, 7757]
set_option maxRecDepth 4096 in
theorem innerCandidates2_45_checked : candidates edge 4 11 4643 innerTail2_46 best.bound = innerCandidates2_45 := by
  unfold candidates
  have row : edge 4643 = SearchData.row4643.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild2_45 : search edge 3 [11, 4643] innerCandidates2_45 4654 best = best := by decide

theorem innerScan2_45 : scan edge (search edge 3) 4 [11] 11 innerTail2_45 best = best := by
  rw [innerTail2_45, scan, if_neg (by decide), innerCandidates2_45_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_46
    (if 3 ≤ innerCandidates2_45.length then search edge 3 [11, 4643] innerCandidates2_45 4654 best else best) = best
  rw [if_pos (by decide), innerChild2_45]
  exact innerScan2_46

def innerTail2_44 : List Nat := 4547 :: innerTail2_45
def innerCandidates2_44 : List Nat := [6791]
set_option maxRecDepth 4096 in
theorem innerCandidates2_44_checked : candidates edge 4 11 4547 innerTail2_45 best.bound = innerCandidates2_44 := by
  unfold candidates
  have row : edge 4547 = SearchData.row4547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_44 : scan edge (search edge 3) 4 [11] 11 innerTail2_44 best = best := by
  rw [innerTail2_44, scan, if_neg (by decide), innerCandidates2_44_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_45
    (if 3 ≤ innerCandidates2_44.length then search edge 3 [11, 4547] innerCandidates2_44 4558 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_45

def innerTail2_43 : List Nat := 4493 :: innerTail2_44
def innerCandidates2_43 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_43_checked : candidates edge 4 11 4493 innerTail2_44 best.bound = innerCandidates2_43 := by
  unfold candidates
  have row : edge 4493 = SearchData.row4493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_43 : scan edge (search edge 3) 4 [11] 11 innerTail2_43 best = best := by
  rw [innerTail2_43, scan, if_neg (by decide), innerCandidates2_43_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_44
    (if 3 ≤ innerCandidates2_43.length then search edge 3 [11, 4493] innerCandidates2_43 4504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_44

def innerTail2_42 : List Nat := 4259 :: innerTail2_43
def innerCandidates2_42 : List Nat := [5807]
set_option maxRecDepth 4096 in
theorem innerCandidates2_42_checked : candidates edge 4 11 4259 innerTail2_43 best.bound = innerCandidates2_42 := by
  unfold candidates
  have row : edge 4259 = SearchData.row4259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_42 : scan edge (search edge 3) 4 [11] 11 innerTail2_42 best = best := by
  rw [innerTail2_42, scan, if_neg (by decide), innerCandidates2_42_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_43
    (if 3 ≤ innerCandidates2_42.length then search edge 3 [11, 4259] innerCandidates2_42 4270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_43

def innerTail2_41 : List Nat := 4229 :: innerTail2_42
def innerCandidates2_41 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_41_checked : candidates edge 4 11 4229 innerTail2_42 best.bound = innerCandidates2_41 := by
  unfold candidates
  have row : edge 4229 = SearchData.row4229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_41 : scan edge (search edge 3) 4 [11] 11 innerTail2_41 best = best := by
  rw [innerTail2_41, scan, if_neg (by decide), innerCandidates2_41_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_42
    (if 3 ≤ innerCandidates2_41.length then search edge 3 [11, 4229] innerCandidates2_41 4240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_42

def innerTail2_40 : List Nat := 4217 :: innerTail2_41
def innerCandidates2_40 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_40_checked : candidates edge 4 11 4217 innerTail2_41 best.bound = innerCandidates2_40 := by
  unfold candidates
  have row : edge 4217 = SearchData.row4217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_40 : scan edge (search edge 3) 4 [11] 11 innerTail2_40 best = best := by
  rw [innerTail2_40, scan, if_neg (by decide), innerCandidates2_40_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_41
    (if 3 ≤ innerCandidates2_40.length then search edge 3 [11, 4217] innerCandidates2_40 4228 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_41

def innerTail2_39 : List Nat := 4073 :: innerTail2_40
def innerCandidates2_39 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_39_checked : candidates edge 4 11 4073 innerTail2_40 best.bound = innerCandidates2_39 := by
  unfold candidates
  have row : edge 4073 = SearchData.row4073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_39 : scan edge (search edge 3) 4 [11] 11 innerTail2_39 best = best := by
  rw [innerTail2_39, scan, if_neg (by decide), innerCandidates2_39_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_40
    (if 3 ≤ innerCandidates2_39.length then search edge 3 [11, 4073] innerCandidates2_39 4084 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_40

def innerTail2_38 : List Nat := 4013 :: innerTail2_39
def innerCandidates2_38 : List Nat := [4643, 5153, 7541, 8387]
set_option maxRecDepth 4096 in
theorem innerCandidates2_38_checked : candidates edge 4 11 4013 innerTail2_39 best.bound = innerCandidates2_38 := by
  unfold candidates
  have row : edge 4013 = SearchData.row4013.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild2_38 : search edge 3 [11, 4013] innerCandidates2_38 4024 best = best := by decide

theorem innerScan2_38 : scan edge (search edge 3) 4 [11] 11 innerTail2_38 best = best := by
  rw [innerTail2_38, scan, if_neg (by decide), innerCandidates2_38_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_39
    (if 3 ≤ innerCandidates2_38.length then search edge 3 [11, 4013] innerCandidates2_38 4024 best else best) = best
  rw [if_pos (by decide), innerChild2_38]
  exact innerScan2_39

def innerTail2_37 : List Nat := 3779 :: innerTail2_38
def innerCandidates2_37 : List Nat := [5099, 6329, 7577]
set_option maxRecDepth 4096 in
theorem innerCandidates2_37_checked : candidates edge 4 11 3779 innerTail2_38 best.bound = innerCandidates2_37 := by
  unfold candidates
  have row : edge 3779 = SearchData.row3779.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild2_37 : search edge 3 [11, 3779] innerCandidates2_37 3790 best = best := by decide

theorem innerScan2_37 : scan edge (search edge 3) 4 [11] 11 innerTail2_37 best = best := by
  rw [innerTail2_37, scan, if_neg (by decide), innerCandidates2_37_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_38
    (if 3 ≤ innerCandidates2_37.length then search edge 3 [11, 3779] innerCandidates2_37 3790 best else best) = best
  rw [if_pos (by decide), innerChild2_37]
  exact innerScan2_38

def innerTail2_36 : List Nat := 3539 :: innerTail2_37
def innerCandidates2_36 : List Nat := [4013, 5861]
set_option maxRecDepth 4096 in
theorem innerCandidates2_36_checked : candidates edge 4 11 3539 innerTail2_37 best.bound = innerCandidates2_36 := by
  unfold candidates
  have row : edge 3539 = SearchData.row3539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_36 : scan edge (search edge 3) 4 [11] 11 innerTail2_36 best = best := by
  rw [innerTail2_36, scan, if_neg (by decide), innerCandidates2_36_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_37
    (if 3 ≤ innerCandidates2_36.length then search edge 3 [11, 3539] innerCandidates2_36 3550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_37

def innerTail2_35 : List Nat := 3467 :: innerTail2_36
def innerCandidates2_35 : List Nat := [5807]
set_option maxRecDepth 4096 in
theorem innerCandidates2_35_checked : candidates edge 4 11 3467 innerTail2_36 best.bound = innerCandidates2_35 := by
  unfold candidates
  have row : edge 3467 = SearchData.row3467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_35 : scan edge (search edge 3) 4 [11] 11 innerTail2_35 best = best := by
  rw [innerTail2_35, scan, if_neg (by decide), innerCandidates2_35_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_36
    (if 3 ≤ innerCandidates2_35.length then search edge 3 [11, 3467] innerCandidates2_35 3478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_36

def innerTail2_34 : List Nat := 3209 :: innerTail2_35
def innerCandidates2_34 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_34_checked : candidates edge 4 11 3209 innerTail2_35 best.bound = innerCandidates2_34 := by
  unfold candidates
  have row : edge 3209 = SearchData.row3209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_34 : scan edge (search edge 3) 4 [11] 11 innerTail2_34 best = best := by
  rw [innerTail2_34, scan, if_neg (by decide), innerCandidates2_34_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_35
    (if 3 ≤ innerCandidates2_34.length then search edge 3 [11, 3209] innerCandidates2_34 3220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_35

def innerTail2_33 : List Nat := 3083 :: innerTail2_34
def innerCandidates2_33 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_33_checked : candidates edge 4 11 3083 innerTail2_34 best.bound = innerCandidates2_33 := by
  unfold candidates
  have row : edge 3083 = SearchData.row3083.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_33 : scan edge (search edge 3) 4 [11] 11 innerTail2_33 best = best := by
  rw [innerTail2_33, scan, if_neg (by decide), innerCandidates2_33_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_34
    (if 3 ≤ innerCandidates2_33.length then search edge 3 [11, 3083] innerCandidates2_33 3094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_34

def innerTail2_32 : List Nat := 2927 :: innerTail2_33
def innerCandidates2_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_32_checked : candidates edge 4 11 2927 innerTail2_33 best.bound = innerCandidates2_32 := by
  unfold candidates
  have row : edge 2927 = SearchData.row2927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_32 : scan edge (search edge 3) 4 [11] 11 innerTail2_32 best = best := by
  rw [innerTail2_32, scan, if_neg (by decide), innerCandidates2_32_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_33
    (if 3 ≤ innerCandidates2_32.length then search edge 3 [11, 2927] innerCandidates2_32 2938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_33

def innerTail2_31 : List Nat := 2843 :: innerTail2_32
def innerCandidates2_31 : List Nat := [5783, 5807, 6047, 8171, 8273]
set_option maxRecDepth 4096 in
theorem innerCandidates2_31_checked : candidates edge 4 11 2843 innerTail2_32 best.bound = innerCandidates2_31 := by
  unfold candidates
  have row : edge 2843 = SearchData.row2843.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild2_31 : search edge 3 [11, 2843] innerCandidates2_31 2854 best = best := by decide

theorem innerScan2_31 : scan edge (search edge 3) 4 [11] 11 innerTail2_31 best = best := by
  rw [innerTail2_31, scan, if_neg (by decide), innerCandidates2_31_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_32
    (if 3 ≤ innerCandidates2_31.length then search edge 3 [11, 2843] innerCandidates2_31 2854 best else best) = best
  rw [if_pos (by decide), innerChild2_31]
  exact innerScan2_32

def innerTail2_30 : List Nat := 2657 :: innerTail2_31
def innerCandidates2_30 : List Nat := [5099]
set_option maxRecDepth 4096 in
theorem innerCandidates2_30_checked : candidates edge 4 11 2657 innerTail2_31 best.bound = innerCandidates2_30 := by
  unfold candidates
  have row : edge 2657 = PrimePairCache.smallRow2657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_30 : scan edge (search edge 3) 4 [11] 11 innerTail2_30 best = best := by
  rw [innerTail2_30, scan, if_neg (by decide), innerCandidates2_30_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_31
    (if 3 ≤ innerCandidates2_30.length then search edge 3 [11, 2657] innerCandidates2_30 2668 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_31

def innerTail2_29 : List Nat := 2621 :: innerTail2_30
def innerCandidates2_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_29_checked : candidates edge 4 11 2621 innerTail2_30 best.bound = innerCandidates2_29 := by
  unfold candidates
  have row : edge 2621 = PrimePairCache.smallRow2621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_29 : scan edge (search edge 3) 4 [11] 11 innerTail2_29 best = best := by
  rw [innerTail2_29, scan, if_neg (by decide), innerCandidates2_29_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_30
    (if 3 ≤ innerCandidates2_29.length then search edge 3 [11, 2621] innerCandidates2_29 2632 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_30

def innerTail2_28 : List Nat := 2459 :: innerTail2_29
def innerCandidates2_28 : List Nat := [2657, 2927]
set_option maxRecDepth 4096 in
theorem innerCandidates2_28_checked : candidates edge 4 11 2459 innerTail2_29 best.bound = innerCandidates2_28 := by
  unfold candidates
  have row : edge 2459 = PrimePairCache.smallRow2459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_28 : scan edge (search edge 3) 4 [11] 11 innerTail2_28 best = best := by
  rw [innerTail2_28, scan, if_neg (by decide), innerCandidates2_28_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_29
    (if 3 ≤ innerCandidates2_28.length then search edge 3 [11, 2459] innerCandidates2_28 2470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_29

def innerTail2_27 : List Nat := 2339 :: innerTail2_28
def innerCandidates2_27 : List Nat := [6047]
set_option maxRecDepth 4096 in
theorem innerCandidates2_27_checked : candidates edge 4 11 2339 innerTail2_28 best.bound = innerCandidates2_27 := by
  unfold candidates
  have row : edge 2339 = PrimePairCache.smallRow2339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_27 : scan edge (search edge 3) 4 [11] 11 innerTail2_27 best = best := by
  rw [innerTail2_27, scan, if_neg (by decide), innerCandidates2_27_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_28
    (if 3 ≤ innerCandidates2_27.length then search edge 3 [11, 2339] innerCandidates2_27 2350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_28

def innerTail2_26 : List Nat := 2297 :: innerTail2_27
def innerCandidates2_26 : List Nat := [4073, 4967, 7127, 7937]
set_option maxRecDepth 4096 in
theorem innerCandidates2_26_checked : candidates edge 4 11 2297 innerTail2_27 best.bound = innerCandidates2_26 := by
  unfold candidates
  have row : edge 2297 = PrimePairCache.smallRow2297.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild2_26 : search edge 3 [11, 2297] innerCandidates2_26 2308 best = best := by decide

theorem innerScan2_26 : scan edge (search edge 3) 4 [11] 11 innerTail2_26 best = best := by
  rw [innerTail2_26, scan, if_neg (by decide), innerCandidates2_26_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_27
    (if 3 ≤ innerCandidates2_26.length then search edge 3 [11, 2297] innerCandidates2_26 2308 best else best) = best
  rw [if_pos (by decide), innerChild2_26]
  exact innerScan2_27

def innerTail2_25 : List Nat := 2237 :: innerTail2_26
def innerCandidates2_25 : List Nat := [6791]
set_option maxRecDepth 4096 in
theorem innerCandidates2_25_checked : candidates edge 4 11 2237 innerTail2_26 best.bound = innerCandidates2_25 := by
  unfold candidates
  have row : edge 2237 = PrimePairCache.smallRow2237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_25 : scan edge (search edge 3) 4 [11] 11 innerTail2_25 best = best := by
  rw [innerTail2_25, scan, if_neg (by decide), innerCandidates2_25_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_26
    (if 3 ≤ innerCandidates2_25.length then search edge 3 [11, 2237] innerCandidates2_25 2248 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_26

def innerTail2_24 : List Nat := 2213 :: innerTail2_25
def innerCandidates2_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_24_checked : candidates edge 4 11 2213 innerTail2_25 best.bound = innerCandidates2_24 := by
  unfold candidates
  have row : edge 2213 = PrimePairCache.smallRow2213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_24 : scan edge (search edge 3) 4 [11] 11 innerTail2_24 best = best := by
  rw [innerTail2_24, scan, if_neg (by decide), innerCandidates2_24_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_25
    (if 3 ≤ innerCandidates2_24.length then search edge 3 [11, 2213] innerCandidates2_24 2224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_25

def innerTail2_23 : List Nat := 2069 :: innerTail2_24
def innerCandidates2_23 : List Nat := [2297, 2339, 7757, 7877, 8219]
set_option maxRecDepth 4096 in
theorem innerCandidates2_23_checked : candidates edge 4 11 2069 innerTail2_24 best.bound = innerCandidates2_23 := by
  unfold candidates
  have row : edge 2069 = PrimePairCache.smallRow2069.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild2_23 : search edge 3 [11, 2069] innerCandidates2_23 2080 best = best := by decide

theorem innerScan2_23 : scan edge (search edge 3) 4 [11] 11 innerTail2_23 best = best := by
  rw [innerTail2_23, scan, if_neg (by decide), innerCandidates2_23_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_24
    (if 3 ≤ innerCandidates2_23.length then search edge 3 [11, 2069] innerCandidates2_23 2080 best else best) = best
  rw [if_pos (by decide), innerChild2_23]
  exact innerScan2_24

def innerTail2_22 : List Nat := 1973 :: innerTail2_23
def innerCandidates2_22 : List Nat := [4493]
set_option maxRecDepth 4096 in
theorem innerCandidates2_22_checked : candidates edge 4 11 1973 innerTail2_23 best.bound = innerCandidates2_22 := by
  unfold candidates
  have row : edge 1973 = PrimePairCache.smallRow1973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_22 : scan edge (search edge 3) 4 [11] 11 innerTail2_22 best = best := by
  rw [innerTail2_22, scan, if_neg (by decide), innerCandidates2_22_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_23
    (if 3 ≤ innerCandidates2_22.length then search edge 3 [11, 1973] innerCandidates2_22 1984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_23

def innerTail2_21 : List Nat := 1949 :: innerTail2_22
def innerCandidates2_21 : List Nat := [2621, 2657]
set_option maxRecDepth 4096 in
theorem innerCandidates2_21_checked : candidates edge 4 11 1949 innerTail2_22 best.bound = innerCandidates2_21 := by
  unfold candidates
  have row : edge 1949 = PrimePairCache.smallRow1949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_21 : scan edge (search edge 3) 4 [11] 11 innerTail2_21 best = best := by
  rw [innerTail2_21, scan, if_neg (by decide), innerCandidates2_21_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_22
    (if 3 ≤ innerCandidates2_21.length then search edge 3 [11, 1949] innerCandidates2_21 1960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_22

def innerTail2_20 : List Nat := 1871 :: innerTail2_21
def innerCandidates2_20 : List Nat := [8171]
set_option maxRecDepth 4096 in
theorem innerCandidates2_20_checked : candidates edge 4 11 1871 innerTail2_21 best.bound = innerCandidates2_20 := by
  unfold candidates
  have row : edge 1871 = PrimePairCache.smallRow1871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_20 : scan edge (search edge 3) 4 [11] 11 innerTail2_20 best = best := by
  rw [innerTail2_20, scan, if_neg (by decide), innerCandidates2_20_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_21
    (if 3 ≤ innerCandidates2_20.length then search edge 3 [11, 1871] innerCandidates2_20 1882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_21

def innerTail2_19 : List Nat := 1847 :: innerTail2_20
def innerCandidates2_19 : List Nat := [3779, 7703]
set_option maxRecDepth 4096 in
theorem innerCandidates2_19_checked : candidates edge 4 11 1847 innerTail2_20 best.bound = innerCandidates2_19 := by
  unfold candidates
  have row : edge 1847 = PrimePairCache.smallRow1847.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_19 : scan edge (search edge 3) 4 [11] 11 innerTail2_19 best = best := by
  rw [innerTail2_19, scan, if_neg (by decide), innerCandidates2_19_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_20
    (if 3 ≤ innerCandidates2_19.length then search edge 3 [11, 1847] innerCandidates2_19 1858 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_20

def innerTail2_18 : List Nat := 1487 :: innerTail2_19
def innerCandidates2_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates2_18_checked : candidates edge 4 11 1487 innerTail2_19 best.bound = innerCandidates2_18 := by
  unfold candidates
  have row : edge 1487 = PrimePairCache.smallRow1487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_18 : scan edge (search edge 3) 4 [11] 11 innerTail2_18 best = best := by
  rw [innerTail2_18, scan, if_neg (by decide), innerCandidates2_18_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_19
    (if 3 ≤ innerCandidates2_18.length then search edge 3 [11, 1487] innerCandidates2_18 1498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_19

def innerTail2_17 : List Nat := 1427 :: innerTail2_18
def innerCandidates2_17 : List Nat := [2339, 7937, 8273]
set_option maxRecDepth 4096 in
theorem innerCandidates2_17_checked : candidates edge 4 11 1427 innerTail2_18 best.bound = innerCandidates2_17 := by
  unfold candidates
  have row : edge 1427 = PrimePairCache.smallRow1427.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild2_17 : search edge 3 [11, 1427] innerCandidates2_17 1438 best = best := by decide

theorem innerScan2_17 : scan edge (search edge 3) 4 [11] 11 innerTail2_17 best = best := by
  rw [innerTail2_17, scan, if_neg (by decide), innerCandidates2_17_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_18
    (if 3 ≤ innerCandidates2_17.length then search edge 3 [11, 1427] innerCandidates2_17 1438 best else best) = best
  rw [if_pos (by decide), innerChild2_17]
  exact innerScan2_18

def innerTail2_16 : List Nat := 1217 :: innerTail2_17
def innerCandidates2_16 : List Nat := [1427, 2843, 5153]
set_option maxRecDepth 4096 in
theorem innerCandidates2_16_checked : candidates edge 4 11 1217 innerTail2_17 best.bound = innerCandidates2_16 := by
  unfold candidates
  have row : edge 1217 = PrimePairCache.smallRow1217.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild2_16 : search edge 3 [11, 1217] innerCandidates2_16 1228 best = best := by decide

theorem innerScan2_16 : scan edge (search edge 3) 4 [11] 11 innerTail2_16 best = best := by
  rw [innerTail2_16, scan, if_neg (by decide), innerCandidates2_16_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_17
    (if 3 ≤ innerCandidates2_16.length then search edge 3 [11, 1217] innerCandidates2_16 1228 best else best) = best
  rw [if_pos (by decide), innerChild2_16]
  exact innerScan2_17

def innerTail2_15 : List Nat := 1103 :: innerTail2_16
def innerCandidates2_15 : List Nat := [2927, 5807, 5987, 6329, 7529]
set_option maxRecDepth 4096 in
theorem innerCandidates2_15_checked : candidates edge 4 11 1103 innerTail2_16 best.bound = innerCandidates2_15 := by
  unfold candidates
  have row : edge 1103 = PrimePairCache.smallRow1103.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild2_15 : search edge 3 [11, 1103] innerCandidates2_15 1114 best = best := by decide

theorem innerScan2_15 : scan edge (search edge 3) 4 [11] 11 innerTail2_15 best = best := by
  rw [innerTail2_15, scan, if_neg (by decide), innerCandidates2_15_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_16
    (if 3 ≤ innerCandidates2_15.length then search edge 3 [11, 1103] innerCandidates2_15 1114 best else best) = best
  rw [if_pos (by decide), innerChild2_15]
  exact innerScan2_16

def innerTail2_14 : List Nat := 1091 :: innerTail2_15
def innerCandidates2_14 : List Nat := [1847, 1949, 8081]
set_option maxRecDepth 4096 in
theorem innerCandidates2_14_checked : candidates edge 4 11 1091 innerTail2_15 best.bound = innerCandidates2_14 := by
  unfold candidates
  have row : edge 1091 = PrimePairCache.smallRow1091.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild2_14 : search edge 3 [11, 1091] innerCandidates2_14 1102 best = best := by decide

theorem innerScan2_14 : scan edge (search edge 3) 4 [11] 11 innerTail2_14 best = best := by
  rw [innerTail2_14, scan, if_neg (by decide), innerCandidates2_14_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_15
    (if 3 ≤ innerCandidates2_14.length then search edge 3 [11, 1091] innerCandidates2_14 1102 best else best) = best
  rw [if_pos (by decide), innerChild2_14]
  exact innerScan2_15

def innerTail2_13 : List Nat := 1049 :: innerTail2_14
def innerCandidates2_13 : List Nat := [1847, 2297, 4493, 7577, 8219]
set_option maxRecDepth 4096 in
theorem innerCandidates2_13_checked : candidates edge 4 11 1049 innerTail2_14 best.bound = innerCandidates2_13 := by
  unfold candidates
  have row : edge 1049 = PrimePairCache.smallRow1049.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild2_13 : search edge 3 [11, 1049] innerCandidates2_13 1060 best = best := by decide

theorem innerScan2_13 : scan edge (search edge 3) 4 [11] 11 innerTail2_13 best = best := by
  rw [innerTail2_13, scan, if_neg (by decide), innerCandidates2_13_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_14
    (if 3 ≤ innerCandidates2_13.length then search edge 3 [11, 1049] innerCandidates2_13 1060 best else best) = best
  rw [if_pos (by decide), innerChild2_13]
  exact innerScan2_14

def innerTail2_12 : List Nat := 953 :: innerTail2_13
def innerCandidates2_12 : List Nat := [7109]
set_option maxRecDepth 4096 in
theorem innerCandidates2_12_checked : candidates edge 4 11 953 innerTail2_13 best.bound = innerCandidates2_12 := by
  unfold candidates
  have row : edge 953 = PrimePairCache.smallRow953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_12 : scan edge (search edge 3) 4 [11] 11 innerTail2_12 best = best := by
  rw [innerTail2_12, scan, if_neg (by decide), innerCandidates2_12_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_13
    (if 3 ≤ innerCandidates2_12.length then search edge 3 [11, 953] innerCandidates2_12 964 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_13

def innerTail2_11 : List Nat := 941 :: innerTail2_12
def innerCandidates2_11 : List Nat := [1049, 6047]
set_option maxRecDepth 4096 in
theorem innerCandidates2_11_checked : candidates edge 4 11 941 innerTail2_12 best.bound = innerCandidates2_11 := by
  unfold candidates
  have row : edge 941 = PrimePairCache.smallRow941.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_11 : scan edge (search edge 3) 4 [11] 11 innerTail2_11 best = best := by
  rw [innerTail2_11, scan, if_neg (by decide), innerCandidates2_11_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_12
    (if 3 ≤ innerCandidates2_11.length then search edge 3 [11, 941] innerCandidates2_11 952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_12

def innerTail2_10 : List Nat := 863 :: innerTail2_11
def innerCandidates2_10 : List Nat := [2297, 2621, 2927, 4259, 5849]
set_option maxRecDepth 4096 in
theorem innerCandidates2_10_checked : candidates edge 4 11 863 innerTail2_11 best.bound = innerCandidates2_10 := by
  unfold candidates
  have row : edge 863 = PrimePairCache.smallRow863.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild2_10 : search edge 3 [11, 863] innerCandidates2_10 874 best = best := by decide

theorem innerScan2_10 : scan edge (search edge 3) 4 [11] 11 innerTail2_10 best = best := by
  rw [innerTail2_10, scan, if_neg (by decide), innerCandidates2_10_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_11
    (if 3 ≤ innerCandidates2_10.length then search edge 3 [11, 863] innerCandidates2_10 874 best else best) = best
  rw [if_pos (by decide), innerChild2_10]
  exact innerScan2_11

def innerTail2_9 : List Nat := 839 :: innerTail2_10
def innerCandidates2_9 : List Nat := [1949, 7331, 7877, 8387]
set_option maxRecDepth 4096 in
theorem innerCandidates2_9_checked : candidates edge 4 11 839 innerTail2_10 best.bound = innerCandidates2_9 := by
  unfold candidates
  have row : edge 839 = PrimePairCache.smallRow839.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild2_9 : search edge 3 [11, 839] innerCandidates2_9 850 best = best := by decide

theorem innerScan2_9 : scan edge (search edge 3) 4 [11] 11 innerTail2_9 best = best := by
  rw [innerTail2_9, scan, if_neg (by decide), innerCandidates2_9_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_10
    (if 3 ≤ innerCandidates2_9.length then search edge 3 [11, 839] innerCandidates2_9 850 best else best) = best
  rw [if_pos (by decide), innerChild2_9]
  exact innerScan2_10

def innerTail2_8 : List Nat := 743 :: innerTail2_9
def innerCandidates2_8 : List Nat := [1871]
set_option maxRecDepth 4096 in
theorem innerCandidates2_8_checked : candidates edge 4 11 743 innerTail2_9 best.bound = innerCandidates2_8 := by
  unfold candidates
  have row : edge 743 = PrimePairCache.smallRow743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_8 : scan edge (search edge 3) 4 [11] 11 innerTail2_8 best = best := by
  rw [innerTail2_8, scan, if_neg (by decide), innerCandidates2_8_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_9
    (if 3 ≤ innerCandidates2_8.length then search edge 3 [11, 743] innerCandidates2_8 754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_9

def innerTail2_7 : List Nat := 701 :: innerTail2_8
def innerCandidates2_7 : List Nat := [7541, 7937, 8171]
set_option maxRecDepth 4096 in
theorem innerCandidates2_7_checked : candidates edge 4 11 701 innerTail2_8 best.bound = innerCandidates2_7 := by
  unfold candidates
  have row : edge 701 = PrimePairCache.smallRow701.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild2_7 : search edge 3 [11, 701] innerCandidates2_7 712 best = best := by decide

theorem innerScan2_7 : scan edge (search edge 3) 4 [11] 11 innerTail2_7 best = best := by
  rw [innerTail2_7, scan, if_neg (by decide), innerCandidates2_7_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_8
    (if 3 ≤ innerCandidates2_7.length then search edge 3 [11, 701] innerCandidates2_7 712 best else best) = best
  rw [if_pos (by decide), innerChild2_7]
  exact innerScan2_8

def innerTail2_6 : List Nat := 587 :: innerTail2_7
def innerCandidates2_6 : List Nat := [4229, 6911, 7127]
set_option maxRecDepth 4096 in
theorem innerCandidates2_6_checked : candidates edge 4 11 587 innerTail2_7 best.bound = innerCandidates2_6 := by
  unfold candidates
  have row : edge 587 = PrimePairCache.smallRow587.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild2_6 : search edge 3 [11, 587] innerCandidates2_6 598 best = best := by decide

theorem innerScan2_6 : scan edge (search edge 3) 4 [11] 11 innerTail2_6 best = best := by
  rw [innerTail2_6, scan, if_neg (by decide), innerCandidates2_6_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_7
    (if 3 ≤ innerCandidates2_6.length then search edge 3 [11, 587] innerCandidates2_6 598 best else best) = best
  rw [if_pos (by decide), innerChild2_6]
  exact innerScan2_7

def innerTail2_5 : List Nat := 503 :: innerTail2_6
def innerCandidates2_5 : List Nat := [4229, 5309]
set_option maxRecDepth 4096 in
theorem innerCandidates2_5_checked : candidates edge 4 11 503 innerTail2_6 best.bound = innerCandidates2_5 := by
  unfold candidates
  have row : edge 503 = PrimePairCache.smallRow503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan2_5 : scan edge (search edge 3) 4 [11] 11 innerTail2_5 best = best := by
  rw [innerTail2_5, scan, if_neg (by decide), innerCandidates2_5_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_6
    (if 3 ≤ innerCandidates2_5.length then search edge 3 [11, 503] innerCandidates2_5 514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan2_6

def innerTail2_4 : List Nat := 353 :: innerTail2_5
def innerCandidates2_4 : List Nat := [2213, 3779, 4493, 4967, 5849, 6791, 7517, 8219]
set_option maxRecDepth 4096 in
theorem innerCandidates2_4_checked : candidates edge 4 11 353 innerTail2_5 best.bound = innerCandidates2_4 := by
  unfold candidates
  have row : edge 353 = PrimePairCache.smallRow353.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild2_4 : search edge 3 [11, 353] innerCandidates2_4 364 best = best := by decide

theorem innerScan2_4 : scan edge (search edge 3) 4 [11] 11 innerTail2_4 best = best := by
  rw [innerTail2_4, scan, if_neg (by decide), innerCandidates2_4_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_5
    (if 3 ≤ innerCandidates2_4.length then search edge 3 [11, 353] innerCandidates2_4 364 best else best) = best
  rw [if_pos (by decide), innerChild2_4]
  exact innerScan2_5

def innerTail2_3 : List Nat := 251 :: innerTail2_4
def innerCandidates2_3 : List Nat := [941, 7127, 8387]
set_option maxRecDepth 4096 in
theorem innerCandidates2_3_checked : candidates edge 4 11 251 innerTail2_4 best.bound = innerCandidates2_3 := by
  unfold candidates
  have row : edge 251 = PrimePairCache.smallRow251.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild2_3 : search edge 3 [11, 251] innerCandidates2_3 262 best = best := by decide

theorem innerScan2_3 : scan edge (search edge 3) 4 [11] 11 innerTail2_3 best = best := by
  rw [innerTail2_3, scan, if_neg (by decide), innerCandidates2_3_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_4
    (if 3 ≤ innerCandidates2_3.length then search edge 3 [11, 251] innerCandidates2_3 262 best else best) = best
  rw [if_pos (by decide), innerChild2_3]
  exact innerScan2_4

def innerTail2_2 : List Nat := 239 :: innerTail2_3
def innerCandidates2_2 : List Nat := [1049, 1091, 1847, 2459, 3467, 5807, 7541]
set_option maxRecDepth 4096 in
theorem innerCandidates2_2_checked : candidates edge 4 11 239 innerTail2_3 best.bound = innerCandidates2_2 := by
  unfold candidates
  have row : edge 239 = PrimePairCache.smallRow239.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild2_2 : search edge 3 [11, 239] innerCandidates2_2 250 best = best := by decide

theorem innerScan2_2 : scan edge (search edge 3) 4 [11] 11 innerTail2_2 best = best := by
  rw [innerTail2_2, scan, if_neg (by decide), innerCandidates2_2_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_3
    (if 3 ≤ innerCandidates2_2.length then search edge 3 [11, 239] innerCandidates2_2 250 best else best) = best
  rw [if_pos (by decide), innerChild2_2]
  exact innerScan2_3

def innerTail2_1 : List Nat := 113 :: innerTail2_2
def innerCandidates2_1 : List Nat := [1217, 1973, 6719]
set_option maxRecDepth 4096 in
theorem innerCandidates2_1_checked : candidates edge 4 11 113 innerTail2_2 best.bound = innerCandidates2_1 := by
  unfold candidates
  have row : edge 113 = PrimePairCache.smallRow113.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild2_1 : search edge 3 [11, 113] innerCandidates2_1 124 best = best := by decide

theorem innerScan2_1 : scan edge (search edge 3) 4 [11] 11 innerTail2_1 best = best := by
  rw [innerTail2_1, scan, if_neg (by decide), innerCandidates2_1_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_2
    (if 3 ≤ innerCandidates2_1.length then search edge 3 [11, 113] innerCandidates2_1 124 best else best) = best
  rw [if_pos (by decide), innerChild2_1]
  exact innerScan2_2

def innerTail2_0 : List Nat := 23 :: innerTail2_1
def innerCandidates2_0 : List Nat := [743, 1871, 3083, 4547, 5849, 6329, 7331, 8171]
set_option maxRecDepth 4096 in
theorem innerCandidates2_0_checked : candidates edge 4 11 23 innerTail2_1 best.bound = innerCandidates2_0 := by
  unfold candidates
  have row : edge 23 = PrimePairCache.smallRow23.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild2_0 : search edge 3 [11, 23] innerCandidates2_0 34 best = best := by decide

theorem innerScan2_0 : scan edge (search edge 3) 4 [11] 11 innerTail2_0 best = best := by
  rw [innerTail2_0, scan, if_neg (by decide), innerCandidates2_0_checked]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_1
    (if 3 ≤ innerCandidates2_0.length then search edge 3 [11, 23] innerCandidates2_0 34 best else best) = best
  rw [if_pos (by decide), innerChild2_0]
  exact innerScan2_1

theorem rootBranch2 : search edge 4 [11] rootCandidates2 11 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [11] 11 innerTail2_0 best = best
  exact innerScan2_0

end ProjectEulerSolutions.PrimePairUpper.Search
