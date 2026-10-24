import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates12 : List Nat := [149, 251, 269, 293, 419, 521, 947, 1091, 1187, 1277, 1283, 1301, 1439, 1481, 1487, 1619, 1721, 1901, 1931, 2027, 2063, 2333, 2393, 2399, 2687, 2741, 2963, 3191, 3257, 3617, 3761, 3833, 3923, 4073, 4211, 4241, 4337, 4751, 4787, 4931, 4937, 5147, 5273, 5297, 5381, 5483, 5639, 5669, 5807, 5879, 5927, 6029, 6089, 6599, 6659, 6701, 6719, 6737, 6803, 6869, 6911, 7517, 8069, 8087, 8111, 8171, 8273]
set_option maxRecDepth 4096 in
theorem rootCandidates12_checked : candidates edge 5 0 47 tail13 best.bound = rootCandidates12 := by
  unfold candidates
  have row : edge 47 = PrimePairCache.smallRow47.contains := by funext q; rfl
  rw [row]
  decide

def innerTail12_67 : List Nat := []
theorem innerScan12_67 : scan edge (search edge 3) 4 [47] 47 innerTail12_67 best = best := rfl

def innerTail12_66 : List Nat := 8273 :: innerTail12_67
def innerCandidates12_66 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates12_66_checked : candidates edge 4 47 8273 innerTail12_67 best.bound = innerCandidates12_66 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_66 : scan edge (search edge 3) 4 [47] 47 innerTail12_66 best = best := by
  rw [innerTail12_66, scan, if_neg (by decide), innerCandidates12_66_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_67
    (if 3 ≤ innerCandidates12_66.length then search edge 3 [47, 8273] innerCandidates12_66 8320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_67

def innerTail12_65 : List Nat := 8171 :: innerTail12_66
def innerCandidates12_65 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates12_65_checked : candidates edge 4 47 8171 innerTail12_66 best.bound = innerCandidates12_65 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_65 : scan edge (search edge 3) 4 [47] 47 innerTail12_65 best = best := by
  rw [innerTail12_65, scan, if_neg (by decide), innerCandidates12_65_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_66
    (if 3 ≤ innerCandidates12_65.length then search edge 3 [47, 8171] innerCandidates12_65 8218 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_66

def innerTail12_64 : List Nat := 8111 :: innerTail12_65
def innerCandidates12_64 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates12_64_checked : candidates edge 4 47 8111 innerTail12_65 best.bound = innerCandidates12_64 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_64 : scan edge (search edge 3) 4 [47] 47 innerTail12_64 best = best := by
  rw [innerTail12_64, scan, if_neg (by decide), innerCandidates12_64_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_65
    (if 3 ≤ innerCandidates12_64.length then search edge 3 [47, 8111] innerCandidates12_64 8158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_65

def innerTail12_63 : List Nat := 8087 :: innerTail12_64
def innerCandidates12_63 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates12_63_checked : candidates edge 4 47 8087 innerTail12_64 best.bound = innerCandidates12_63 := by
  unfold candidates
  have row : edge 8087 = SearchData.row8087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_63 : scan edge (search edge 3) 4 [47] 47 innerTail12_63 best = best := by
  rw [innerTail12_63, scan, if_neg (by decide), innerCandidates12_63_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_64
    (if 3 ≤ innerCandidates12_63.length then search edge 3 [47, 8087] innerCandidates12_63 8134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_64

def innerTail12_62 : List Nat := 8069 :: innerTail12_63
def innerCandidates12_62 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates12_62_checked : candidates edge 4 47 8069 innerTail12_63 best.bound = innerCandidates12_62 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_62 : scan edge (search edge 3) 4 [47] 47 innerTail12_62 best = best := by
  rw [innerTail12_62, scan, if_neg (by decide), innerCandidates12_62_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_63
    (if 3 ≤ innerCandidates12_62.length then search edge 3 [47, 8069] innerCandidates12_62 8116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_63

def innerTail12_61 : List Nat := 7517 :: innerTail12_62
def innerCandidates12_61 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates12_61_checked : candidates edge 4 47 7517 innerTail12_62 best.bound = innerCandidates12_61 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_61 : scan edge (search edge 3) 4 [47] 47 innerTail12_61 best = best := by
  rw [innerTail12_61, scan, if_neg (by decide), innerCandidates12_61_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_62
    (if 3 ≤ innerCandidates12_61.length then search edge 3 [47, 7517] innerCandidates12_61 7564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_62

def innerTail12_60 : List Nat := 6911 :: innerTail12_61
def innerCandidates12_60 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates12_60_checked : candidates edge 4 47 6911 innerTail12_61 best.bound = innerCandidates12_60 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_60 : scan edge (search edge 3) 4 [47] 47 innerTail12_60 best = best := by
  rw [innerTail12_60, scan, if_neg (by decide), innerCandidates12_60_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_61
    (if 3 ≤ innerCandidates12_60.length then search edge 3 [47, 6911] innerCandidates12_60 6958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_61

def innerTail12_59 : List Nat := 6869 :: innerTail12_60
def innerCandidates12_59 : List Nat := [8087, 8111]
set_option maxRecDepth 4096 in
theorem innerCandidates12_59_checked : candidates edge 4 47 6869 innerTail12_60 best.bound = innerCandidates12_59 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_59 : scan edge (search edge 3) 4 [47] 47 innerTail12_59 best = best := by
  rw [innerTail12_59, scan, if_neg (by decide), innerCandidates12_59_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_60
    (if 3 ≤ innerCandidates12_59.length then search edge 3 [47, 6869] innerCandidates12_59 6916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_60

def innerTail12_58 : List Nat := 6803 :: innerTail12_59
def innerCandidates12_58 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates12_58_checked : candidates edge 4 47 6803 innerTail12_59 best.bound = innerCandidates12_58 := by
  unfold candidates
  have row : edge 6803 = SearchData.row6803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_58 : scan edge (search edge 3) 4 [47] 47 innerTail12_58 best = best := by
  rw [innerTail12_58, scan, if_neg (by decide), innerCandidates12_58_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_59
    (if 3 ≤ innerCandidates12_58.length then search edge 3 [47, 6803] innerCandidates12_58 6850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_59

def innerTail12_57 : List Nat := 6737 :: innerTail12_58
def innerCandidates12_57 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates12_57_checked : candidates edge 4 47 6737 innerTail12_58 best.bound = innerCandidates12_57 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_57 : scan edge (search edge 3) 4 [47] 47 innerTail12_57 best = best := by
  rw [innerTail12_57, scan, if_neg (by decide), innerCandidates12_57_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_58
    (if 3 ≤ innerCandidates12_57.length then search edge 3 [47, 6737] innerCandidates12_57 6784 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_58

def innerTail12_56 : List Nat := 6719 :: innerTail12_57
def innerCandidates12_56 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates12_56_checked : candidates edge 4 47 6719 innerTail12_57 best.bound = innerCandidates12_56 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_56 : scan edge (search edge 3) 4 [47] 47 innerTail12_56 best = best := by
  rw [innerTail12_56, scan, if_neg (by decide), innerCandidates12_56_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_57
    (if 3 ≤ innerCandidates12_56.length then search edge 3 [47, 6719] innerCandidates12_56 6766 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_57

def innerTail12_55 : List Nat := 6701 :: innerTail12_56
def innerCandidates12_55 : List Nat := [8069]
set_option maxRecDepth 4096 in
theorem innerCandidates12_55_checked : candidates edge 4 47 6701 innerTail12_56 best.bound = innerCandidates12_55 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_55 : scan edge (search edge 3) 4 [47] 47 innerTail12_55 best = best := by
  rw [innerTail12_55, scan, if_neg (by decide), innerCandidates12_55_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_56
    (if 3 ≤ innerCandidates12_55.length then search edge 3 [47, 6701] innerCandidates12_55 6748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_56

def innerTail12_54 : List Nat := 6659 :: innerTail12_55
def innerCandidates12_54 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates12_54_checked : candidates edge 4 47 6659 innerTail12_55 best.bound = innerCandidates12_54 := by
  unfold candidates
  have row : edge 6659 = SearchData.row6659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_54 : scan edge (search edge 3) 4 [47] 47 innerTail12_54 best = best := by
  rw [innerTail12_54, scan, if_neg (by decide), innerCandidates12_54_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_55
    (if 3 ≤ innerCandidates12_54.length then search edge 3 [47, 6659] innerCandidates12_54 6706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_55

def innerTail12_53 : List Nat := 6599 :: innerTail12_54
def innerCandidates12_53 : List Nat := [8111]
set_option maxRecDepth 4096 in
theorem innerCandidates12_53_checked : candidates edge 4 47 6599 innerTail12_54 best.bound = innerCandidates12_53 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_53 : scan edge (search edge 3) 4 [47] 47 innerTail12_53 best = best := by
  rw [innerTail12_53, scan, if_neg (by decide), innerCandidates12_53_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_54
    (if 3 ≤ innerCandidates12_53.length then search edge 3 [47, 6599] innerCandidates12_53 6646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_54

def innerTail12_52 : List Nat := 6089 :: innerTail12_53
def innerCandidates12_52 : List Nat := [8111]
set_option maxRecDepth 4096 in
theorem innerCandidates12_52_checked : candidates edge 4 47 6089 innerTail12_53 best.bound = innerCandidates12_52 := by
  unfold candidates
  have row : edge 6089 = SearchData.row6089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_52 : scan edge (search edge 3) 4 [47] 47 innerTail12_52 best = best := by
  rw [innerTail12_52, scan, if_neg (by decide), innerCandidates12_52_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_53
    (if 3 ≤ innerCandidates12_52.length then search edge 3 [47, 6089] innerCandidates12_52 6136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_53

def innerTail12_51 : List Nat := 6029 :: innerTail12_52
def innerCandidates12_51 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates12_51_checked : candidates edge 4 47 6029 innerTail12_52 best.bound = innerCandidates12_51 := by
  unfold candidates
  have row : edge 6029 = SearchData.row6029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_51 : scan edge (search edge 3) 4 [47] 47 innerTail12_51 best = best := by
  rw [innerTail12_51, scan, if_neg (by decide), innerCandidates12_51_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_52
    (if 3 ≤ innerCandidates12_51.length then search edge 3 [47, 6029] innerCandidates12_51 6076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_52

def innerTail12_50 : List Nat := 5927 :: innerTail12_51
def innerCandidates12_50 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates12_50_checked : candidates edge 4 47 5927 innerTail12_51 best.bound = innerCandidates12_50 := by
  unfold candidates
  have row : edge 5927 = SearchData.row5927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_50 : scan edge (search edge 3) 4 [47] 47 innerTail12_50 best = best := by
  rw [innerTail12_50, scan, if_neg (by decide), innerCandidates12_50_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_51
    (if 3 ≤ innerCandidates12_50.length then search edge 3 [47, 5927] innerCandidates12_50 5974 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_51

def innerTail12_49 : List Nat := 5879 :: innerTail12_50
def innerCandidates12_49 : List Nat := [8273]
set_option maxRecDepth 4096 in
theorem innerCandidates12_49_checked : candidates edge 4 47 5879 innerTail12_50 best.bound = innerCandidates12_49 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_49 : scan edge (search edge 3) 4 [47] 47 innerTail12_49 best = best := by
  rw [innerTail12_49, scan, if_neg (by decide), innerCandidates12_49_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_50
    (if 3 ≤ innerCandidates12_49.length then search edge 3 [47, 5879] innerCandidates12_49 5926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_50

def innerTail12_48 : List Nat := 5807 :: innerTail12_49
def innerCandidates12_48 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates12_48_checked : candidates edge 4 47 5807 innerTail12_49 best.bound = innerCandidates12_48 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_48 : scan edge (search edge 3) 4 [47] 47 innerTail12_48 best = best := by
  rw [innerTail12_48, scan, if_neg (by decide), innerCandidates12_48_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_49
    (if 3 ≤ innerCandidates12_48.length then search edge 3 [47, 5807] innerCandidates12_48 5854 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_49

def innerTail12_47 : List Nat := 5669 :: innerTail12_48
def innerCandidates12_47 : List Nat := [5879, 7517]
set_option maxRecDepth 4096 in
theorem innerCandidates12_47_checked : candidates edge 4 47 5669 innerTail12_48 best.bound = innerCandidates12_47 := by
  unfold candidates
  have row : edge 5669 = SearchData.row5669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_47 : scan edge (search edge 3) 4 [47] 47 innerTail12_47 best = best := by
  rw [innerTail12_47, scan, if_neg (by decide), innerCandidates12_47_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_48
    (if 3 ≤ innerCandidates12_47.length then search edge 3 [47, 5669] innerCandidates12_47 5716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_48

def innerTail12_46 : List Nat := 5639 :: innerTail12_47
def innerCandidates12_46 : List Nat := [6737]
set_option maxRecDepth 4096 in
theorem innerCandidates12_46_checked : candidates edge 4 47 5639 innerTail12_47 best.bound = innerCandidates12_46 := by
  unfold candidates
  have row : edge 5639 = SearchData.row5639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_46 : scan edge (search edge 3) 4 [47] 47 innerTail12_46 best = best := by
  rw [innerTail12_46, scan, if_neg (by decide), innerCandidates12_46_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_47
    (if 3 ≤ innerCandidates12_46.length then search edge 3 [47, 5639] innerCandidates12_46 5686 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_47

def innerTail12_45 : List Nat := 5483 :: innerTail12_46
def innerCandidates12_45 : List Nat := [6029]
set_option maxRecDepth 4096 in
theorem innerCandidates12_45_checked : candidates edge 4 47 5483 innerTail12_46 best.bound = innerCandidates12_45 := by
  unfold candidates
  have row : edge 5483 = SearchData.row5483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_45 : scan edge (search edge 3) 4 [47] 47 innerTail12_45 best = best := by
  rw [innerTail12_45, scan, if_neg (by decide), innerCandidates12_45_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_46
    (if 3 ≤ innerCandidates12_45.length then search edge 3 [47, 5483] innerCandidates12_45 5530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_46

def innerTail12_44 : List Nat := 5381 :: innerTail12_45
def innerCandidates12_44 : List Nat := [6599]
set_option maxRecDepth 4096 in
theorem innerCandidates12_44_checked : candidates edge 4 47 5381 innerTail12_45 best.bound = innerCandidates12_44 := by
  unfold candidates
  have row : edge 5381 = SearchData.row5381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_44 : scan edge (search edge 3) 4 [47] 47 innerTail12_44 best = best := by
  rw [innerTail12_44, scan, if_neg (by decide), innerCandidates12_44_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_45
    (if 3 ≤ innerCandidates12_44.length then search edge 3 [47, 5381] innerCandidates12_44 5428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_45

def innerTail12_43 : List Nat := 5297 :: innerTail12_44
def innerCandidates12_43 : List Nat := [6701]
set_option maxRecDepth 4096 in
theorem innerCandidates12_43_checked : candidates edge 4 47 5297 innerTail12_44 best.bound = innerCandidates12_43 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_43 : scan edge (search edge 3) 4 [47] 47 innerTail12_43 best = best := by
  rw [innerTail12_43, scan, if_neg (by decide), innerCandidates12_43_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_44
    (if 3 ≤ innerCandidates12_43.length then search edge 3 [47, 5297] innerCandidates12_43 5344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_44

def innerTail12_42 : List Nat := 5273 :: innerTail12_43
def innerCandidates12_42 : List Nat := [5381]
set_option maxRecDepth 4096 in
theorem innerCandidates12_42_checked : candidates edge 4 47 5273 innerTail12_43 best.bound = innerCandidates12_42 := by
  unfold candidates
  have row : edge 5273 = SearchData.row5273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_42 : scan edge (search edge 3) 4 [47] 47 innerTail12_42 best = best := by
  rw [innerTail12_42, scan, if_neg (by decide), innerCandidates12_42_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_43
    (if 3 ≤ innerCandidates12_42.length then search edge 3 [47, 5273] innerCandidates12_42 5320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_43

def innerTail12_41 : List Nat := 5147 :: innerTail12_42
def innerCandidates12_41 : List Nat := [5273, 5297]
set_option maxRecDepth 4096 in
theorem innerCandidates12_41_checked : candidates edge 4 47 5147 innerTail12_42 best.bound = innerCandidates12_41 := by
  unfold candidates
  have row : edge 5147 = SearchData.row5147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_41 : scan edge (search edge 3) 4 [47] 47 innerTail12_41 best = best := by
  rw [innerTail12_41, scan, if_neg (by decide), innerCandidates12_41_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_42
    (if 3 ≤ innerCandidates12_41.length then search edge 3 [47, 5147] innerCandidates12_41 5194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_42

def innerTail12_40 : List Nat := 4937 :: innerTail12_41
def innerCandidates12_40 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates12_40_checked : candidates edge 4 47 4937 innerTail12_41 best.bound = innerCandidates12_40 := by
  unfold candidates
  have row : edge 4937 = SearchData.row4937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_40 : scan edge (search edge 3) 4 [47] 47 innerTail12_40 best = best := by
  rw [innerTail12_40, scan, if_neg (by decide), innerCandidates12_40_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_41
    (if 3 ≤ innerCandidates12_40.length then search edge 3 [47, 4937] innerCandidates12_40 4984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_41

def innerTail12_39 : List Nat := 4931 :: innerTail12_40
def innerCandidates12_39 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates12_39_checked : candidates edge 4 47 4931 innerTail12_40 best.bound = innerCandidates12_39 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_39 : scan edge (search edge 3) 4 [47] 47 innerTail12_39 best = best := by
  rw [innerTail12_39, scan, if_neg (by decide), innerCandidates12_39_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_40
    (if 3 ≤ innerCandidates12_39.length then search edge 3 [47, 4931] innerCandidates12_39 4978 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_40

def innerTail12_38 : List Nat := 4787 :: innerTail12_39
def innerCandidates12_38 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates12_38_checked : candidates edge 4 47 4787 innerTail12_39 best.bound = innerCandidates12_38 := by
  unfold candidates
  have row : edge 4787 = SearchData.row4787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_38 : scan edge (search edge 3) 4 [47] 47 innerTail12_38 best = best := by
  rw [innerTail12_38, scan, if_neg (by decide), innerCandidates12_38_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_39
    (if 3 ≤ innerCandidates12_38.length then search edge 3 [47, 4787] innerCandidates12_38 4834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_39

def innerTail12_37 : List Nat := 4751 :: innerTail12_38
def innerCandidates12_37 : List Nat := [5297, 5879, 6659]
set_option maxRecDepth 4096 in
theorem innerCandidates12_37_checked : candidates edge 4 47 4751 innerTail12_38 best.bound = innerCandidates12_37 := by
  unfold candidates
  have row : edge 4751 = SearchData.row4751.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild12_37 : search edge 3 [47, 4751] innerCandidates12_37 4798 best = best := by decide

theorem innerScan12_37 : scan edge (search edge 3) 4 [47] 47 innerTail12_37 best = best := by
  rw [innerTail12_37, scan, if_neg (by decide), innerCandidates12_37_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_38
    (if 3 ≤ innerCandidates12_37.length then search edge 3 [47, 4751] innerCandidates12_37 4798 best else best) = best
  rw [if_pos (by decide), innerChild12_37]
  exact innerScan12_38

def innerTail12_36 : List Nat := 4337 :: innerTail12_37
def innerCandidates12_36 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates12_36_checked : candidates edge 4 47 4337 innerTail12_37 best.bound = innerCandidates12_36 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_36 : scan edge (search edge 3) 4 [47] 47 innerTail12_36 best = best := by
  rw [innerTail12_36, scan, if_neg (by decide), innerCandidates12_36_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_37
    (if 3 ≤ innerCandidates12_36.length then search edge 3 [47, 4337] innerCandidates12_36 4384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_37

def innerTail12_35 : List Nat := 4241 :: innerTail12_36
def innerCandidates12_35 : List Nat := [4751]
set_option maxRecDepth 4096 in
theorem innerCandidates12_35_checked : candidates edge 4 47 4241 innerTail12_36 best.bound = innerCandidates12_35 := by
  unfold candidates
  have row : edge 4241 = SearchData.row4241.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_35 : scan edge (search edge 3) 4 [47] 47 innerTail12_35 best = best := by
  rw [innerTail12_35, scan, if_neg (by decide), innerCandidates12_35_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_36
    (if 3 ≤ innerCandidates12_35.length then search edge 3 [47, 4241] innerCandidates12_35 4288 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_36

def innerTail12_34 : List Nat := 4211 :: innerTail12_35
def innerCandidates12_34 : List Nat := [5483, 6911]
set_option maxRecDepth 4096 in
theorem innerCandidates12_34_checked : candidates edge 4 47 4211 innerTail12_35 best.bound = innerCandidates12_34 := by
  unfold candidates
  have row : edge 4211 = SearchData.row4211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_34 : scan edge (search edge 3) 4 [47] 47 innerTail12_34 best = best := by
  rw [innerTail12_34, scan, if_neg (by decide), innerCandidates12_34_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_35
    (if 3 ≤ innerCandidates12_34.length then search edge 3 [47, 4211] innerCandidates12_34 4258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_35

def innerTail12_33 : List Nat := 4073 :: innerTail12_34
def innerCandidates12_33 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates12_33_checked : candidates edge 4 47 4073 innerTail12_34 best.bound = innerCandidates12_33 := by
  unfold candidates
  have row : edge 4073 = SearchData.row4073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_33 : scan edge (search edge 3) 4 [47] 47 innerTail12_33 best = best := by
  rw [innerTail12_33, scan, if_neg (by decide), innerCandidates12_33_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_34
    (if 3 ≤ innerCandidates12_33.length then search edge 3 [47, 4073] innerCandidates12_33 4120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_34

def innerTail12_32 : List Nat := 3923 :: innerTail12_33
def innerCandidates12_32 : List Nat := [4211, 6803]
set_option maxRecDepth 4096 in
theorem innerCandidates12_32_checked : candidates edge 4 47 3923 innerTail12_33 best.bound = innerCandidates12_32 := by
  unfold candidates
  have row : edge 3923 = SearchData.row3923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_32 : scan edge (search edge 3) 4 [47] 47 innerTail12_32 best = best := by
  rw [innerTail12_32, scan, if_neg (by decide), innerCandidates12_32_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_33
    (if 3 ≤ innerCandidates12_32.length then search edge 3 [47, 3923] innerCandidates12_32 3970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_33

def innerTail12_31 : List Nat := 3833 :: innerTail12_32
def innerCandidates12_31 : List Nat := [6911]
set_option maxRecDepth 4096 in
theorem innerCandidates12_31_checked : candidates edge 4 47 3833 innerTail12_32 best.bound = innerCandidates12_31 := by
  unfold candidates
  have row : edge 3833 = SearchData.row3833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_31 : scan edge (search edge 3) 4 [47] 47 innerTail12_31 best = best := by
  rw [innerTail12_31, scan, if_neg (by decide), innerCandidates12_31_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_32
    (if 3 ≤ innerCandidates12_31.length then search edge 3 [47, 3833] innerCandidates12_31 3880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_32

def innerTail12_30 : List Nat := 3761 :: innerTail12_31
def innerCandidates12_30 : List Nat := [5807]
set_option maxRecDepth 4096 in
theorem innerCandidates12_30_checked : candidates edge 4 47 3761 innerTail12_31 best.bound = innerCandidates12_30 := by
  unfold candidates
  have row : edge 3761 = SearchData.row3761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_30 : scan edge (search edge 3) 4 [47] 47 innerTail12_30 best = best := by
  rw [innerTail12_30, scan, if_neg (by decide), innerCandidates12_30_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_31
    (if 3 ≤ innerCandidates12_30.length then search edge 3 [47, 3761] innerCandidates12_30 3808 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_31

def innerTail12_29 : List Nat := 3617 :: innerTail12_30
def innerCandidates12_29 : List Nat := [3923, 5669, 6089]
set_option maxRecDepth 4096 in
theorem innerCandidates12_29_checked : candidates edge 4 47 3617 innerTail12_30 best.bound = innerCandidates12_29 := by
  unfold candidates
  have row : edge 3617 = SearchData.row3617.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild12_29 : search edge 3 [47, 3617] innerCandidates12_29 3664 best = best := by decide

theorem innerScan12_29 : scan edge (search edge 3) 4 [47] 47 innerTail12_29 best = best := by
  rw [innerTail12_29, scan, if_neg (by decide), innerCandidates12_29_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_30
    (if 3 ≤ innerCandidates12_29.length then search edge 3 [47, 3617] innerCandidates12_29 3664 best else best) = best
  rw [if_pos (by decide), innerChild12_29]
  exact innerScan12_30

def innerTail12_28 : List Nat := 3257 :: innerTail12_29
def innerCandidates12_28 : List Nat := [4937]
set_option maxRecDepth 4096 in
theorem innerCandidates12_28_checked : candidates edge 4 47 3257 innerTail12_29 best.bound = innerCandidates12_28 := by
  unfold candidates
  have row : edge 3257 = SearchData.row3257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_28 : scan edge (search edge 3) 4 [47] 47 innerTail12_28 best = best := by
  rw [innerTail12_28, scan, if_neg (by decide), innerCandidates12_28_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_29
    (if 3 ≤ innerCandidates12_28.length then search edge 3 [47, 3257] innerCandidates12_28 3304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_29

def innerTail12_27 : List Nat := 3191 :: innerTail12_28
def innerCandidates12_27 : List Nat := [8069, 8111]
set_option maxRecDepth 4096 in
theorem innerCandidates12_27_checked : candidates edge 4 47 3191 innerTail12_28 best.bound = innerCandidates12_27 := by
  unfold candidates
  have row : edge 3191 = SearchData.row3191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_27 : scan edge (search edge 3) 4 [47] 47 innerTail12_27 best = best := by
  rw [innerTail12_27, scan, if_neg (by decide), innerCandidates12_27_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_28
    (if 3 ≤ innerCandidates12_27.length then search edge 3 [47, 3191] innerCandidates12_27 3238 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_28

def innerTail12_26 : List Nat := 2963 :: innerTail12_27
def innerCandidates12_26 : List Nat := [5381, 8111]
set_option maxRecDepth 4096 in
theorem innerCandidates12_26_checked : candidates edge 4 47 2963 innerTail12_27 best.bound = innerCandidates12_26 := by
  unfold candidates
  have row : edge 2963 = SearchData.row2963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_26 : scan edge (search edge 3) 4 [47] 47 innerTail12_26 best = best := by
  rw [innerTail12_26, scan, if_neg (by decide), innerCandidates12_26_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_27
    (if 3 ≤ innerCandidates12_26.length then search edge 3 [47, 2963] innerCandidates12_26 3010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_27

def innerTail12_25 : List Nat := 2741 :: innerTail12_26
def innerCandidates12_25 : List Nat := [2963, 6737, 6911]
set_option maxRecDepth 4096 in
theorem innerCandidates12_25_checked : candidates edge 4 47 2741 innerTail12_26 best.bound = innerCandidates12_25 := by
  unfold candidates
  have row : edge 2741 = SearchData.row2741.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild12_25 : search edge 3 [47, 2741] innerCandidates12_25 2788 best = best := by decide

theorem innerScan12_25 : scan edge (search edge 3) 4 [47] 47 innerTail12_25 best = best := by
  rw [innerTail12_25, scan, if_neg (by decide), innerCandidates12_25_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_26
    (if 3 ≤ innerCandidates12_25.length then search edge 3 [47, 2741] innerCandidates12_25 2788 best else best) = best
  rw [if_pos (by decide), innerChild12_25]
  exact innerScan12_26

def innerTail12_24 : List Nat := 2687 :: innerTail12_25
def innerCandidates12_24 : List Nat := [6599, 6737, 8069]
set_option maxRecDepth 4096 in
theorem innerCandidates12_24_checked : candidates edge 4 47 2687 innerTail12_25 best.bound = innerCandidates12_24 := by
  unfold candidates
  have row : edge 2687 = SearchData.row2687.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild12_24 : search edge 3 [47, 2687] innerCandidates12_24 2734 best = best := by decide

theorem innerScan12_24 : scan edge (search edge 3) 4 [47] 47 innerTail12_24 best = best := by
  rw [innerTail12_24, scan, if_neg (by decide), innerCandidates12_24_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_25
    (if 3 ≤ innerCandidates12_24.length then search edge 3 [47, 2687] innerCandidates12_24 2734 best else best) = best
  rw [if_pos (by decide), innerChild12_24]
  exact innerScan12_25

def innerTail12_23 : List Nat := 2399 :: innerTail12_24
def innerCandidates12_23 : List Nat := [6701, 6737, 8087]
set_option maxRecDepth 4096 in
theorem innerCandidates12_23_checked : candidates edge 4 47 2399 innerTail12_24 best.bound = innerCandidates12_23 := by
  unfold candidates
  have row : edge 2399 = PrimePairCache.smallRow2399.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild12_23 : search edge 3 [47, 2399] innerCandidates12_23 2446 best = best := by decide

theorem innerScan12_23 : scan edge (search edge 3) 4 [47] 47 innerTail12_23 best = best := by
  rw [innerTail12_23, scan, if_neg (by decide), innerCandidates12_23_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_24
    (if 3 ≤ innerCandidates12_23.length then search edge 3 [47, 2399] innerCandidates12_23 2446 best else best) = best
  rw [if_pos (by decide), innerChild12_23]
  exact innerScan12_24

def innerTail12_22 : List Nat := 2393 :: innerTail12_23
def innerCandidates12_22 : List Nat := [4241, 8069]
set_option maxRecDepth 4096 in
theorem innerCandidates12_22_checked : candidates edge 4 47 2393 innerTail12_23 best.bound = innerCandidates12_22 := by
  unfold candidates
  have row : edge 2393 = PrimePairCache.smallRow2393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_22 : scan edge (search edge 3) 4 [47] 47 innerTail12_22 best = best := by
  rw [innerTail12_22, scan, if_neg (by decide), innerCandidates12_22_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_23
    (if 3 ≤ innerCandidates12_22.length then search edge 3 [47, 2393] innerCandidates12_22 2440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_23

def innerTail12_21 : List Nat := 2333 :: innerTail12_22
def innerCandidates12_21 : List Nat := [2963, 5297, 5927, 6701]
set_option maxRecDepth 4096 in
theorem innerCandidates12_21_checked : candidates edge 4 47 2333 innerTail12_22 best.bound = innerCandidates12_21 := by
  unfold candidates
  have row : edge 2333 = PrimePairCache.smallRow2333.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild12_21 : search edge 3 [47, 2333] innerCandidates12_21 2380 best = best := by decide

theorem innerScan12_21 : scan edge (search edge 3) 4 [47] 47 innerTail12_21 best = best := by
  rw [innerTail12_21, scan, if_neg (by decide), innerCandidates12_21_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_22
    (if 3 ≤ innerCandidates12_21.length then search edge 3 [47, 2333] innerCandidates12_21 2380 best else best) = best
  rw [if_pos (by decide), innerChild12_21]
  exact innerScan12_22

def innerTail12_20 : List Nat := 2063 :: innerTail12_21
def innerCandidates12_20 : List Nat := [2333, 3617, 6599]
set_option maxRecDepth 4096 in
theorem innerCandidates12_20_checked : candidates edge 4 47 2063 innerTail12_21 best.bound = innerCandidates12_20 := by
  unfold candidates
  have row : edge 2063 = PrimePairCache.smallRow2063.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild12_20 : search edge 3 [47, 2063] innerCandidates12_20 2110 best = best := by decide

theorem innerScan12_20 : scan edge (search edge 3) 4 [47] 47 innerTail12_20 best = best := by
  rw [innerTail12_20, scan, if_neg (by decide), innerCandidates12_20_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_21
    (if 3 ≤ innerCandidates12_20.length then search edge 3 [47, 2063] innerCandidates12_20 2110 best else best) = best
  rw [if_pos (by decide), innerChild12_20]
  exact innerScan12_21

def innerTail12_19 : List Nat := 2027 :: innerTail12_20
def innerCandidates12_19 : List Nat := [3761, 5483, 6701, 6911]
set_option maxRecDepth 4096 in
theorem innerCandidates12_19_checked : candidates edge 4 47 2027 innerTail12_20 best.bound = innerCandidates12_19 := by
  unfold candidates
  have row : edge 2027 = PrimePairCache.smallRow2027.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild12_19 : search edge 3 [47, 2027] innerCandidates12_19 2074 best = best := by decide

theorem innerScan12_19 : scan edge (search edge 3) 4 [47] 47 innerTail12_19 best = best := by
  rw [innerTail12_19, scan, if_neg (by decide), innerCandidates12_19_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_20
    (if 3 ≤ innerCandidates12_19.length then search edge 3 [47, 2027] innerCandidates12_19 2074 best else best) = best
  rw [if_pos (by decide), innerChild12_19]
  exact innerScan12_20

def innerTail12_18 : List Nat := 1931 :: innerTail12_19
def innerCandidates12_18 : List Nat := [2399, 5297, 6719]
set_option maxRecDepth 4096 in
theorem innerCandidates12_18_checked : candidates edge 4 47 1931 innerTail12_19 best.bound = innerCandidates12_18 := by
  unfold candidates
  have row : edge 1931 = PrimePairCache.smallRow1931.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild12_18 : search edge 3 [47, 1931] innerCandidates12_18 1978 best = best := by decide

theorem innerScan12_18 : scan edge (search edge 3) 4 [47] 47 innerTail12_18 best = best := by
  rw [innerTail12_18, scan, if_neg (by decide), innerCandidates12_18_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_19
    (if 3 ≤ innerCandidates12_18.length then search edge 3 [47, 1931] innerCandidates12_18 1978 best else best) = best
  rw [if_pos (by decide), innerChild12_18]
  exact innerScan12_19

def innerTail12_17 : List Nat := 1901 :: innerTail12_18
def innerCandidates12_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates12_17_checked : candidates edge 4 47 1901 innerTail12_18 best.bound = innerCandidates12_17 := by
  unfold candidates
  have row : edge 1901 = PrimePairCache.smallRow1901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_17 : scan edge (search edge 3) 4 [47] 47 innerTail12_17 best = best := by
  rw [innerTail12_17, scan, if_neg (by decide), innerCandidates12_17_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_18
    (if 3 ≤ innerCandidates12_17.length then search edge 3 [47, 1901] innerCandidates12_17 1948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_18

def innerTail12_16 : List Nat := 1721 :: innerTail12_17
def innerCandidates12_16 : List Nat := [8273]
set_option maxRecDepth 4096 in
theorem innerCandidates12_16_checked : candidates edge 4 47 1721 innerTail12_17 best.bound = innerCandidates12_16 := by
  unfold candidates
  have row : edge 1721 = PrimePairCache.smallRow1721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_16 : scan edge (search edge 3) 4 [47] 47 innerTail12_16 best = best := by
  rw [innerTail12_16, scan, if_neg (by decide), innerCandidates12_16_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_17
    (if 3 ≤ innerCandidates12_16.length then search edge 3 [47, 1721] innerCandidates12_16 1768 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_17

def innerTail12_15 : List Nat := 1619 :: innerTail12_16
def innerCandidates12_15 : List Nat := [3257, 6701]
set_option maxRecDepth 4096 in
theorem innerCandidates12_15_checked : candidates edge 4 47 1619 innerTail12_16 best.bound = innerCandidates12_15 := by
  unfold candidates
  have row : edge 1619 = PrimePairCache.smallRow1619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_15 : scan edge (search edge 3) 4 [47] 47 innerTail12_15 best = best := by
  rw [innerTail12_15, scan, if_neg (by decide), innerCandidates12_15_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_16
    (if 3 ≤ innerCandidates12_15.length then search edge 3 [47, 1619] innerCandidates12_15 1666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_16

def innerTail12_14 : List Nat := 1487 :: innerTail12_15
def innerCandidates12_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates12_14_checked : candidates edge 4 47 1487 innerTail12_15 best.bound = innerCandidates12_14 := by
  unfold candidates
  have row : edge 1487 = PrimePairCache.smallRow1487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_14 : scan edge (search edge 3) 4 [47] 47 innerTail12_14 best = best := by
  rw [innerTail12_14, scan, if_neg (by decide), innerCandidates12_14_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_15
    (if 3 ≤ innerCandidates12_14.length then search edge 3 [47, 1487] innerCandidates12_14 1534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_15

def innerTail12_13 : List Nat := 1481 :: innerTail12_14
def innerCandidates12_13 : List Nat := [4211, 5639, 6869]
set_option maxRecDepth 4096 in
theorem innerCandidates12_13_checked : candidates edge 4 47 1481 innerTail12_14 best.bound = innerCandidates12_13 := by
  unfold candidates
  have row : edge 1481 = PrimePairCache.smallRow1481.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild12_13 : search edge 3 [47, 1481] innerCandidates12_13 1528 best = best := by decide

theorem innerScan12_13 : scan edge (search edge 3) 4 [47] 47 innerTail12_13 best = best := by
  rw [innerTail12_13, scan, if_neg (by decide), innerCandidates12_13_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_14
    (if 3 ≤ innerCandidates12_13.length then search edge 3 [47, 1481] innerCandidates12_13 1528 best else best) = best
  rw [if_pos (by decide), innerChild12_13]
  exact innerScan12_14

def innerTail12_12 : List Nat := 1439 :: innerTail12_13
def innerCandidates12_12 : List Nat := [3833, 4337, 4931, 5273, 6659, 6869]
set_option maxRecDepth 4096 in
theorem innerCandidates12_12_checked : candidates edge 4 47 1439 innerTail12_13 best.bound = innerCandidates12_12 := by
  unfold candidates
  have row : edge 1439 = PrimePairCache.smallRow1439.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild12_12 : search edge 3 [47, 1439] innerCandidates12_12 1486 best = best := by decide

theorem innerScan12_12 : scan edge (search edge 3) 4 [47] 47 innerTail12_12 best = best := by
  rw [innerTail12_12, scan, if_neg (by decide), innerCandidates12_12_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_13
    (if 3 ≤ innerCandidates12_12.length then search edge 3 [47, 1439] innerCandidates12_12 1486 best else best) = best
  rw [if_pos (by decide), innerChild12_12]
  exact innerScan12_13

def innerTail12_11 : List Nat := 1301 :: innerTail12_12
def innerCandidates12_11 : List Nat := [2393, 5669]
set_option maxRecDepth 4096 in
theorem innerCandidates12_11_checked : candidates edge 4 47 1301 innerTail12_12 best.bound = innerCandidates12_11 := by
  unfold candidates
  have row : edge 1301 = PrimePairCache.smallRow1301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_11 : scan edge (search edge 3) 4 [47] 47 innerTail12_11 best = best := by
  rw [innerTail12_11, scan, if_neg (by decide), innerCandidates12_11_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_12
    (if 3 ≤ innerCandidates12_11.length then search edge 3 [47, 1301] innerCandidates12_11 1348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_12

def innerTail12_10 : List Nat := 1283 :: innerTail12_11
def innerCandidates12_10 : List Nat := [1619, 2741, 3617, 5807, 6869]
set_option maxRecDepth 4096 in
theorem innerCandidates12_10_checked : candidates edge 4 47 1283 innerTail12_11 best.bound = innerCandidates12_10 := by
  unfold candidates
  have row : edge 1283 = PrimePairCache.smallRow1283.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild12_10 : search edge 3 [47, 1283] innerCandidates12_10 1330 best = best := by decide

theorem innerScan12_10 : scan edge (search edge 3) 4 [47] 47 innerTail12_10 best = best := by
  rw [innerTail12_10, scan, if_neg (by decide), innerCandidates12_10_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_11
    (if 3 ≤ innerCandidates12_10.length then search edge 3 [47, 1283] innerCandidates12_10 1330 best else best) = best
  rw [if_pos (by decide), innerChild12_10]
  exact innerScan12_11

def innerTail12_9 : List Nat := 1277 :: innerTail12_10
def innerCandidates12_9 : List Nat := [4787]
set_option maxRecDepth 4096 in
theorem innerCandidates12_9_checked : candidates edge 4 47 1277 innerTail12_10 best.bound = innerCandidates12_9 := by
  unfold candidates
  have row : edge 1277 = PrimePairCache.smallRow1277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_9 : scan edge (search edge 3) 4 [47] 47 innerTail12_9 best = best := by
  rw [innerTail12_9, scan, if_neg (by decide), innerCandidates12_9_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_10
    (if 3 ≤ innerCandidates12_9.length then search edge 3 [47, 1277] innerCandidates12_9 1324 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_10

def innerTail12_8 : List Nat := 1187 :: innerTail12_9
def innerCandidates12_8 : List Nat := [4937, 6701]
set_option maxRecDepth 4096 in
theorem innerCandidates12_8_checked : candidates edge 4 47 1187 innerTail12_9 best.bound = innerCandidates12_8 := by
  unfold candidates
  have row : edge 1187 = PrimePairCache.smallRow1187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_8 : scan edge (search edge 3) 4 [47] 47 innerTail12_8 best = best := by
  rw [innerTail12_8, scan, if_neg (by decide), innerCandidates12_8_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_9
    (if 3 ≤ innerCandidates12_8.length then search edge 3 [47, 1187] innerCandidates12_8 1234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_9

def innerTail12_7 : List Nat := 1091 :: innerTail12_8
def innerCandidates12_7 : List Nat := [4931, 8111]
set_option maxRecDepth 4096 in
theorem innerCandidates12_7_checked : candidates edge 4 47 1091 innerTail12_8 best.bound = innerCandidates12_7 := by
  unfold candidates
  have row : edge 1091 = PrimePairCache.smallRow1091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_7 : scan edge (search edge 3) 4 [47] 47 innerTail12_7 best = best := by
  rw [innerTail12_7, scan, if_neg (by decide), innerCandidates12_7_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_8
    (if 3 ≤ innerCandidates12_7.length then search edge 3 [47, 1091] innerCandidates12_7 1138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_8

def innerTail12_6 : List Nat := 947 :: innerTail12_7
def innerCandidates12_6 : List Nat := [1439, 1901, 4931, 8111]
set_option maxRecDepth 4096 in
theorem innerCandidates12_6_checked : candidates edge 4 47 947 innerTail12_7 best.bound = innerCandidates12_6 := by
  unfold candidates
  have row : edge 947 = PrimePairCache.smallRow947.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild12_6 : search edge 3 [47, 947] innerCandidates12_6 994 best = best := by decide

theorem innerScan12_6 : scan edge (search edge 3) 4 [47] 47 innerTail12_6 best = best := by
  rw [innerTail12_6, scan, if_neg (by decide), innerCandidates12_6_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_7
    (if 3 ≤ innerCandidates12_6.length then search edge 3 [47, 947] innerCandidates12_6 994 best else best) = best
  rw [if_pos (by decide), innerChild12_6]
  exact innerScan12_7

def innerTail12_5 : List Nat := 521 :: innerTail12_6
def innerCandidates12_5 : List Nat := [1091, 1721, 4337, 5381]
set_option maxRecDepth 4096 in
theorem innerCandidates12_5_checked : candidates edge 4 47 521 innerTail12_6 best.bound = innerCandidates12_5 := by
  unfold candidates
  have row : edge 521 = PrimePairCache.smallRow521.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild12_5 : search edge 3 [47, 521] innerCandidates12_5 568 best = best := by decide

theorem innerScan12_5 : scan edge (search edge 3) 4 [47] 47 innerTail12_5 best = best := by
  rw [innerTail12_5, scan, if_neg (by decide), innerCandidates12_5_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_6
    (if 3 ≤ innerCandidates12_5.length then search edge 3 [47, 521] innerCandidates12_5 568 best else best) = best
  rw [if_pos (by decide), innerChild12_5]
  exact innerScan12_6

def innerTail12_4 : List Nat := 419 :: innerTail12_5
def innerCandidates12_4 : List Nat := [5273, 5297, 8111]
set_option maxRecDepth 4096 in
theorem innerCandidates12_4_checked : candidates edge 4 47 419 innerTail12_5 best.bound = innerCandidates12_4 := by
  unfold candidates
  have row : edge 419 = PrimePairCache.smallRow419.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild12_4 : search edge 3 [47, 419] innerCandidates12_4 466 best = best := by decide

theorem innerScan12_4 : scan edge (search edge 3) 4 [47] 47 innerTail12_4 best = best := by
  rw [innerTail12_4, scan, if_neg (by decide), innerCandidates12_4_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_5
    (if 3 ≤ innerCandidates12_4.length then search edge 3 [47, 419] innerCandidates12_4 466 best else best) = best
  rw [if_pos (by decide), innerChild12_4]
  exact innerScan12_5

def innerTail12_3 : List Nat := 293 :: innerTail12_4
def innerCandidates12_3 : List Nat := [4751, 5147]
set_option maxRecDepth 4096 in
theorem innerCandidates12_3_checked : candidates edge 4 47 293 innerTail12_4 best.bound = innerCandidates12_3 := by
  unfold candidates
  have row : edge 293 = PrimePairCache.smallRow293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_3 : scan edge (search edge 3) 4 [47] 47 innerTail12_3 best = best := by
  rw [innerTail12_3, scan, if_neg (by decide), innerCandidates12_3_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_4
    (if 3 ≤ innerCandidates12_3.length then search edge 3 [47, 293] innerCandidates12_3 340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_4

def innerTail12_2 : List Nat := 269 :: innerTail12_3
def innerCandidates12_2 : List Nat := [2741]
set_option maxRecDepth 4096 in
theorem innerCandidates12_2_checked : candidates edge 4 47 269 innerTail12_3 best.bound = innerCandidates12_2 := by
  unfold candidates
  have row : edge 269 = PrimePairCache.smallRow269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_2 : scan edge (search edge 3) 4 [47] 47 innerTail12_2 best = best := by
  rw [innerTail12_2, scan, if_neg (by decide), innerCandidates12_2_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_3
    (if 3 ≤ innerCandidates12_2.length then search edge 3 [47, 269] innerCandidates12_2 316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_3

def innerTail12_1 : List Nat := 251 :: innerTail12_2
def innerCandidates12_1 : List Nat := [2063, 5147, 5273]
set_option maxRecDepth 4096 in
theorem innerCandidates12_1_checked : candidates edge 4 47 251 innerTail12_2 best.bound = innerCandidates12_1 := by
  unfold candidates
  have row : edge 251 = PrimePairCache.smallRow251.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild12_1 : search edge 3 [47, 251] innerCandidates12_1 298 best = best := by decide

theorem innerScan12_1 : scan edge (search edge 3) 4 [47] 47 innerTail12_1 best = best := by
  rw [innerTail12_1, scan, if_neg (by decide), innerCandidates12_1_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_2
    (if 3 ≤ innerCandidates12_1.length then search edge 3 [47, 251] innerCandidates12_1 298 best else best) = best
  rw [if_pos (by decide), innerChild12_1]
  exact innerScan12_2

def innerTail12_0 : List Nat := 149 :: innerTail12_1
def innerCandidates12_0 : List Nat := [251, 3617]
set_option maxRecDepth 4096 in
theorem innerCandidates12_0_checked : candidates edge 4 47 149 innerTail12_1 best.bound = innerCandidates12_0 := by
  unfold candidates
  have row : edge 149 = PrimePairCache.smallRow149.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan12_0 : scan edge (search edge 3) 4 [47] 47 innerTail12_0 best = best := by
  rw [innerTail12_0, scan, if_neg (by decide), innerCandidates12_0_checked]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_1
    (if 3 ≤ innerCandidates12_0.length then search edge 3 [47, 149] innerCandidates12_0 196 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan12_1

theorem rootBranch12 : search edge 4 [47] rootCandidates12 47 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [47] 47 innerTail12_0 best = best
  exact innerScan12_0

end ProjectEulerSolutions.PrimePairUpper.Search
