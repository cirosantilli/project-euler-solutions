import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates3 : List Nat := [19, 61, 103, 127, 241, 331, 337, 367, 523, 577, 709, 829, 997, 1009, 1129, 1213, 1231, 1321, 1381, 1489, 1543, 1627, 1861, 2113, 2137, 2287, 2347, 2383, 2689, 2851, 2953, 2971, 3187, 3499, 3559, 3583, 3673, 3967, 4219, 4243, 4327, 4363, 4513, 4591, 4789, 4909, 4951, 5077, 5197, 5209, 5431, 5581, 5623, 5647, 5701, 5851, 6043, 6163, 6343, 6397, 6421, 6691, 6733, 6841, 6991, 7477, 7507, 7639, 7723, 7927, 7993, 8053, 8059, 8179]
set_option maxRecDepth 4096 in
theorem rootCandidates3_checked : candidates edge 5 0 13 tail4 best.bound = rootCandidates3 := by
  unfold candidates
  have row : edge 13 = PrimePairCache.smallRow13.contains := by funext q; rfl
  rw [row]
  decide

def innerTail3_74 : List Nat := []
theorem innerScan3_74 : scan edge (search edge 3) 4 [13] 13 innerTail3_74 best = best := rfl

def innerTail3_73 : List Nat := 8179 :: innerTail3_74
def innerCandidates3_73 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_73_checked : candidates edge 4 13 8179 innerTail3_74 best.bound = innerCandidates3_73 := by
  unfold candidates
  have row : edge 8179 = SearchData.row8179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_73 : scan edge (search edge 3) 4 [13] 13 innerTail3_73 best = best := by
  rw [innerTail3_73, scan, if_neg (by decide), innerCandidates3_73_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_74
    (if 3 ≤ innerCandidates3_73.length then search edge 3 [13, 8179] innerCandidates3_73 8192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_74

def innerTail3_72 : List Nat := 8059 :: innerTail3_73
def innerCandidates3_72 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_72_checked : candidates edge 4 13 8059 innerTail3_73 best.bound = innerCandidates3_72 := by
  unfold candidates
  have row : edge 8059 = SearchData.row8059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_72 : scan edge (search edge 3) 4 [13] 13 innerTail3_72 best = best := by
  rw [innerTail3_72, scan, if_neg (by decide), innerCandidates3_72_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_73
    (if 3 ≤ innerCandidates3_72.length then search edge 3 [13, 8059] innerCandidates3_72 8072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_73

def innerTail3_71 : List Nat := 8053 :: innerTail3_72
def innerCandidates3_71 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_71_checked : candidates edge 4 13 8053 innerTail3_72 best.bound = innerCandidates3_71 := by
  unfold candidates
  have row : edge 8053 = SearchData.row8053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_71 : scan edge (search edge 3) 4 [13] 13 innerTail3_71 best = best := by
  rw [innerTail3_71, scan, if_neg (by decide), innerCandidates3_71_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_72
    (if 3 ≤ innerCandidates3_71.length then search edge 3 [13, 8053] innerCandidates3_71 8066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_72

def innerTail3_70 : List Nat := 7993 :: innerTail3_71
def innerCandidates3_70 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_70_checked : candidates edge 4 13 7993 innerTail3_71 best.bound = innerCandidates3_70 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_70 : scan edge (search edge 3) 4 [13] 13 innerTail3_70 best = best := by
  rw [innerTail3_70, scan, if_neg (by decide), innerCandidates3_70_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_71
    (if 3 ≤ innerCandidates3_70.length then search edge 3 [13, 7993] innerCandidates3_70 8006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_71

def innerTail3_69 : List Nat := 7927 :: innerTail3_70
def innerCandidates3_69 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_69_checked : candidates edge 4 13 7927 innerTail3_70 best.bound = innerCandidates3_69 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_69 : scan edge (search edge 3) 4 [13] 13 innerTail3_69 best = best := by
  rw [innerTail3_69, scan, if_neg (by decide), innerCandidates3_69_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_70
    (if 3 ≤ innerCandidates3_69.length then search edge 3 [13, 7927] innerCandidates3_69 7940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_70

def innerTail3_68 : List Nat := 7723 :: innerTail3_69
def innerCandidates3_68 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_68_checked : candidates edge 4 13 7723 innerTail3_69 best.bound = innerCandidates3_68 := by
  unfold candidates
  have row : edge 7723 = SearchData.row7723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_68 : scan edge (search edge 3) 4 [13] 13 innerTail3_68 best = best := by
  rw [innerTail3_68, scan, if_neg (by decide), innerCandidates3_68_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_69
    (if 3 ≤ innerCandidates3_68.length then search edge 3 [13, 7723] innerCandidates3_68 7736 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_69

def innerTail3_67 : List Nat := 7639 :: innerTail3_68
def innerCandidates3_67 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_67_checked : candidates edge 4 13 7639 innerTail3_68 best.bound = innerCandidates3_67 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_67 : scan edge (search edge 3) 4 [13] 13 innerTail3_67 best = best := by
  rw [innerTail3_67, scan, if_neg (by decide), innerCandidates3_67_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_68
    (if 3 ≤ innerCandidates3_67.length then search edge 3 [13, 7639] innerCandidates3_67 7652 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_68

def innerTail3_66 : List Nat := 7507 :: innerTail3_67
def innerCandidates3_66 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_66_checked : candidates edge 4 13 7507 innerTail3_67 best.bound = innerCandidates3_66 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_66 : scan edge (search edge 3) 4 [13] 13 innerTail3_66 best = best := by
  rw [innerTail3_66, scan, if_neg (by decide), innerCandidates3_66_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_67
    (if 3 ≤ innerCandidates3_66.length then search edge 3 [13, 7507] innerCandidates3_66 7520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_67

def innerTail3_65 : List Nat := 7477 :: innerTail3_66
def innerCandidates3_65 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_65_checked : candidates edge 4 13 7477 innerTail3_66 best.bound = innerCandidates3_65 := by
  unfold candidates
  have row : edge 7477 = SearchData.row7477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_65 : scan edge (search edge 3) 4 [13] 13 innerTail3_65 best = best := by
  rw [innerTail3_65, scan, if_neg (by decide), innerCandidates3_65_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_66
    (if 3 ≤ innerCandidates3_65.length then search edge 3 [13, 7477] innerCandidates3_65 7490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_66

def innerTail3_64 : List Nat := 6991 :: innerTail3_65
def innerCandidates3_64 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_64_checked : candidates edge 4 13 6991 innerTail3_65 best.bound = innerCandidates3_64 := by
  unfold candidates
  have row : edge 6991 = SearchData.row6991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_64 : scan edge (search edge 3) 4 [13] 13 innerTail3_64 best = best := by
  rw [innerTail3_64, scan, if_neg (by decide), innerCandidates3_64_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_65
    (if 3 ≤ innerCandidates3_64.length then search edge 3 [13, 6991] innerCandidates3_64 7004 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_65

def innerTail3_63 : List Nat := 6841 :: innerTail3_64
def innerCandidates3_63 : List Nat := [6991]
set_option maxRecDepth 4096 in
theorem innerCandidates3_63_checked : candidates edge 4 13 6841 innerTail3_64 best.bound = innerCandidates3_63 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_63 : scan edge (search edge 3) 4 [13] 13 innerTail3_63 best = best := by
  rw [innerTail3_63, scan, if_neg (by decide), innerCandidates3_63_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_64
    (if 3 ≤ innerCandidates3_63.length then search edge 3 [13, 6841] innerCandidates3_63 6854 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_64

def innerTail3_62 : List Nat := 6733 :: innerTail3_63
def innerCandidates3_62 : List Nat := [7993]
set_option maxRecDepth 4096 in
theorem innerCandidates3_62_checked : candidates edge 4 13 6733 innerTail3_63 best.bound = innerCandidates3_62 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_62 : scan edge (search edge 3) 4 [13] 13 innerTail3_62 best = best := by
  rw [innerTail3_62, scan, if_neg (by decide), innerCandidates3_62_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_63
    (if 3 ≤ innerCandidates3_62.length then search edge 3 [13, 6733] innerCandidates3_62 6746 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_63

def innerTail3_61 : List Nat := 6691 :: innerTail3_62
def innerCandidates3_61 : List Nat := [7507]
set_option maxRecDepth 4096 in
theorem innerCandidates3_61_checked : candidates edge 4 13 6691 innerTail3_62 best.bound = innerCandidates3_61 := by
  unfold candidates
  have row : edge 6691 = SearchData.row6691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_61 : scan edge (search edge 3) 4 [13] 13 innerTail3_61 best = best := by
  rw [innerTail3_61, scan, if_neg (by decide), innerCandidates3_61_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_62
    (if 3 ≤ innerCandidates3_61.length then search edge 3 [13, 6691] innerCandidates3_61 6704 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_62

def innerTail3_60 : List Nat := 6421 :: innerTail3_61
def innerCandidates3_60 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_60_checked : candidates edge 4 13 6421 innerTail3_61 best.bound = innerCandidates3_60 := by
  unfold candidates
  have row : edge 6421 = SearchData.row6421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_60 : scan edge (search edge 3) 4 [13] 13 innerTail3_60 best = best := by
  rw [innerTail3_60, scan, if_neg (by decide), innerCandidates3_60_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_61
    (if 3 ≤ innerCandidates3_60.length then search edge 3 [13, 6421] innerCandidates3_60 6434 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_61

def innerTail3_59 : List Nat := 6397 :: innerTail3_60
def innerCandidates3_59 : List Nat := [8179]
set_option maxRecDepth 4096 in
theorem innerCandidates3_59_checked : candidates edge 4 13 6397 innerTail3_60 best.bound = innerCandidates3_59 := by
  unfold candidates
  have row : edge 6397 = SearchData.row6397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_59 : scan edge (search edge 3) 4 [13] 13 innerTail3_59 best = best := by
  rw [innerTail3_59, scan, if_neg (by decide), innerCandidates3_59_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_60
    (if 3 ≤ innerCandidates3_59.length then search edge 3 [13, 6397] innerCandidates3_59 6410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_60

def innerTail3_58 : List Nat := 6343 :: innerTail3_59
def innerCandidates3_58 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_58_checked : candidates edge 4 13 6343 innerTail3_59 best.bound = innerCandidates3_58 := by
  unfold candidates
  have row : edge 6343 = SearchData.row6343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_58 : scan edge (search edge 3) 4 [13] 13 innerTail3_58 best = best := by
  rw [innerTail3_58, scan, if_neg (by decide), innerCandidates3_58_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_59
    (if 3 ≤ innerCandidates3_58.length then search edge 3 [13, 6343] innerCandidates3_58 6356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_59

def innerTail3_57 : List Nat := 6163 :: innerTail3_58
def innerCandidates3_57 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_57_checked : candidates edge 4 13 6163 innerTail3_58 best.bound = innerCandidates3_57 := by
  unfold candidates
  have row : edge 6163 = SearchData.row6163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_57 : scan edge (search edge 3) 4 [13] 13 innerTail3_57 best = best := by
  rw [innerTail3_57, scan, if_neg (by decide), innerCandidates3_57_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_58
    (if 3 ≤ innerCandidates3_57.length then search edge 3 [13, 6163] innerCandidates3_57 6176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_58

def innerTail3_56 : List Nat := 6043 :: innerTail3_57
def innerCandidates3_56 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_56_checked : candidates edge 4 13 6043 innerTail3_57 best.bound = innerCandidates3_56 := by
  unfold candidates
  have row : edge 6043 = SearchData.row6043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_56 : scan edge (search edge 3) 4 [13] 13 innerTail3_56 best = best := by
  rw [innerTail3_56, scan, if_neg (by decide), innerCandidates3_56_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_57
    (if 3 ≤ innerCandidates3_56.length then search edge 3 [13, 6043] innerCandidates3_56 6056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_57

def innerTail3_55 : List Nat := 5851 :: innerTail3_56
def innerCandidates3_55 : List Nat := [7507]
set_option maxRecDepth 4096 in
theorem innerCandidates3_55_checked : candidates edge 4 13 5851 innerTail3_56 best.bound = innerCandidates3_55 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_55 : scan edge (search edge 3) 4 [13] 13 innerTail3_55 best = best := by
  rw [innerTail3_55, scan, if_neg (by decide), innerCandidates3_55_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_56
    (if 3 ≤ innerCandidates3_55.length then search edge 3 [13, 5851] innerCandidates3_55 5864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_56

def innerTail3_54 : List Nat := 5701 :: innerTail3_55
def innerCandidates3_54 : List Nat := [6733]
set_option maxRecDepth 4096 in
theorem innerCandidates3_54_checked : candidates edge 4 13 5701 innerTail3_55 best.bound = innerCandidates3_54 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_54 : scan edge (search edge 3) 4 [13] 13 innerTail3_54 best = best := by
  rw [innerTail3_54, scan, if_neg (by decide), innerCandidates3_54_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_55
    (if 3 ≤ innerCandidates3_54.length then search edge 3 [13, 5701] innerCandidates3_54 5714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_55

def innerTail3_53 : List Nat := 5647 :: innerTail3_54
def innerCandidates3_53 : List Nat := [6733]
set_option maxRecDepth 4096 in
theorem innerCandidates3_53_checked : candidates edge 4 13 5647 innerTail3_54 best.bound = innerCandidates3_53 := by
  unfold candidates
  have row : edge 5647 = SearchData.row5647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_53 : scan edge (search edge 3) 4 [13] 13 innerTail3_53 best = best := by
  rw [innerTail3_53, scan, if_neg (by decide), innerCandidates3_53_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_54
    (if 3 ≤ innerCandidates3_53.length then search edge 3 [13, 5647] innerCandidates3_53 5660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_54

def innerTail3_52 : List Nat := 5623 :: innerTail3_53
def innerCandidates3_52 : List Nat := [6421, 7639]
set_option maxRecDepth 4096 in
theorem innerCandidates3_52_checked : candidates edge 4 13 5623 innerTail3_53 best.bound = innerCandidates3_52 := by
  unfold candidates
  have row : edge 5623 = SearchData.row5623.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_52 : scan edge (search edge 3) 4 [13] 13 innerTail3_52 best = best := by
  rw [innerTail3_52, scan, if_neg (by decide), innerCandidates3_52_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_53
    (if 3 ≤ innerCandidates3_52.length then search edge 3 [13, 5623] innerCandidates3_52 5636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_53

def innerTail3_51 : List Nat := 5581 :: innerTail3_52
def innerCandidates3_51 : List Nat := [6991]
set_option maxRecDepth 4096 in
theorem innerCandidates3_51_checked : candidates edge 4 13 5581 innerTail3_52 best.bound = innerCandidates3_51 := by
  unfold candidates
  have row : edge 5581 = SearchData.row5581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_51 : scan edge (search edge 3) 4 [13] 13 innerTail3_51 best = best := by
  rw [innerTail3_51, scan, if_neg (by decide), innerCandidates3_51_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_52
    (if 3 ≤ innerCandidates3_51.length then search edge 3 [13, 5581] innerCandidates3_51 5594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_52

def innerTail3_50 : List Nat := 5431 :: innerTail3_51
def innerCandidates3_50 : List Nat := [7507]
set_option maxRecDepth 4096 in
theorem innerCandidates3_50_checked : candidates edge 4 13 5431 innerTail3_51 best.bound = innerCandidates3_50 := by
  unfold candidates
  have row : edge 5431 = SearchData.row5431.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_50 : scan edge (search edge 3) 4 [13] 13 innerTail3_50 best = best := by
  rw [innerTail3_50, scan, if_neg (by decide), innerCandidates3_50_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_51
    (if 3 ≤ innerCandidates3_50.length then search edge 3 [13, 5431] innerCandidates3_50 5444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_51

def innerTail3_49 : List Nat := 5209 :: innerTail3_50
def innerCandidates3_49 : List Nat := [6397]
set_option maxRecDepth 4096 in
theorem innerCandidates3_49_checked : candidates edge 4 13 5209 innerTail3_50 best.bound = innerCandidates3_49 := by
  unfold candidates
  have row : edge 5209 = SearchData.row5209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_49 : scan edge (search edge 3) 4 [13] 13 innerTail3_49 best = best := by
  rw [innerTail3_49, scan, if_neg (by decide), innerCandidates3_49_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_50
    (if 3 ≤ innerCandidates3_49.length then search edge 3 [13, 5209] innerCandidates3_49 5222 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_50

def innerTail3_48 : List Nat := 5197 :: innerTail3_49
def innerCandidates3_48 : List Nat := [5701, 6733]
set_option maxRecDepth 4096 in
theorem innerCandidates3_48_checked : candidates edge 4 13 5197 innerTail3_49 best.bound = innerCandidates3_48 := by
  unfold candidates
  have row : edge 5197 = SearchData.row5197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_48 : scan edge (search edge 3) 4 [13] 13 innerTail3_48 best = best := by
  rw [innerTail3_48, scan, if_neg (by decide), innerCandidates3_48_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_49
    (if 3 ≤ innerCandidates3_48.length then search edge 3 [13, 5197] innerCandidates3_48 5210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_49

def innerTail3_47 : List Nat := 5077 :: innerTail3_48
def innerCandidates3_47 : List Nat := [6043]
set_option maxRecDepth 4096 in
theorem innerCandidates3_47_checked : candidates edge 4 13 5077 innerTail3_48 best.bound = innerCandidates3_47 := by
  unfold candidates
  have row : edge 5077 = SearchData.row5077.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_47 : scan edge (search edge 3) 4 [13] 13 innerTail3_47 best = best := by
  rw [innerTail3_47, scan, if_neg (by decide), innerCandidates3_47_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_48
    (if 3 ≤ innerCandidates3_47.length then search edge 3 [13, 5077] innerCandidates3_47 5090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_48

def innerTail3_46 : List Nat := 4951 :: innerTail3_47
def innerCandidates3_46 : List Nat := [6343]
set_option maxRecDepth 4096 in
theorem innerCandidates3_46_checked : candidates edge 4 13 4951 innerTail3_47 best.bound = innerCandidates3_46 := by
  unfold candidates
  have row : edge 4951 = SearchData.row4951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_46 : scan edge (search edge 3) 4 [13] 13 innerTail3_46 best = best := by
  rw [innerTail3_46, scan, if_neg (by decide), innerCandidates3_46_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_47
    (if 3 ≤ innerCandidates3_46.length then search edge 3 [13, 4951] innerCandidates3_46 4964 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_47

def innerTail3_45 : List Nat := 4909 :: innerTail3_46
def innerCandidates3_45 : List Nat := [5647, 7723]
set_option maxRecDepth 4096 in
theorem innerCandidates3_45_checked : candidates edge 4 13 4909 innerTail3_46 best.bound = innerCandidates3_45 := by
  unfold candidates
  have row : edge 4909 = SearchData.row4909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_45 : scan edge (search edge 3) 4 [13] 13 innerTail3_45 best = best := by
  rw [innerTail3_45, scan, if_neg (by decide), innerCandidates3_45_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_46
    (if 3 ≤ innerCandidates3_45.length then search edge 3 [13, 4909] innerCandidates3_45 4922 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_46

def innerTail3_44 : List Nat := 4789 :: innerTail3_45
def innerCandidates3_44 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_44_checked : candidates edge 4 13 4789 innerTail3_45 best.bound = innerCandidates3_44 := by
  unfold candidates
  have row : edge 4789 = SearchData.row4789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_44 : scan edge (search edge 3) 4 [13] 13 innerTail3_44 best = best := by
  rw [innerTail3_44, scan, if_neg (by decide), innerCandidates3_44_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_45
    (if 3 ≤ innerCandidates3_44.length then search edge 3 [13, 4789] innerCandidates3_44 4802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_45

def innerTail3_43 : List Nat := 4591 :: innerTail3_44
def innerCandidates3_43 : List Nat := [5197]
set_option maxRecDepth 4096 in
theorem innerCandidates3_43_checked : candidates edge 4 13 4591 innerTail3_44 best.bound = innerCandidates3_43 := by
  unfold candidates
  have row : edge 4591 = SearchData.row4591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_43 : scan edge (search edge 3) 4 [13] 13 innerTail3_43 best = best := by
  rw [innerTail3_43, scan, if_neg (by decide), innerCandidates3_43_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_44
    (if 3 ≤ innerCandidates3_43.length then search edge 3 [13, 4591] innerCandidates3_43 4604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_44

def innerTail3_42 : List Nat := 4513 :: innerTail3_43
def innerCandidates3_42 : List Nat := [4591, 5581, 6163, 6397, 6841, 7723]
set_option maxRecDepth 4096 in
theorem innerCandidates3_42_checked : candidates edge 4 13 4513 innerTail3_43 best.bound = innerCandidates3_42 := by
  unfold candidates
  have row : edge 4513 = SearchData.row4513.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild3_42 : search edge 3 [13, 4513] innerCandidates3_42 4526 best = best := by decide

theorem innerScan3_42 : scan edge (search edge 3) 4 [13] 13 innerTail3_42 best = best := by
  rw [innerTail3_42, scan, if_neg (by decide), innerCandidates3_42_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_43
    (if 3 ≤ innerCandidates3_42.length then search edge 3 [13, 4513] innerCandidates3_42 4526 best else best) = best
  rw [if_pos (by decide), innerChild3_42]
  exact innerScan3_43

def innerTail3_41 : List Nat := 4363 :: innerTail3_42
def innerCandidates3_41 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_41_checked : candidates edge 4 13 4363 innerTail3_42 best.bound = innerCandidates3_41 := by
  unfold candidates
  have row : edge 4363 = SearchData.row4363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_41 : scan edge (search edge 3) 4 [13] 13 innerTail3_41 best = best := by
  rw [innerTail3_41, scan, if_neg (by decide), innerCandidates3_41_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_42
    (if 3 ≤ innerCandidates3_41.length then search edge 3 [13, 4363] innerCandidates3_41 4376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_42

def innerTail3_40 : List Nat := 4327 :: innerTail3_41
def innerCandidates3_40 : List Nat := [4591, 7639]
set_option maxRecDepth 4096 in
theorem innerCandidates3_40_checked : candidates edge 4 13 4327 innerTail3_41 best.bound = innerCandidates3_40 := by
  unfold candidates
  have row : edge 4327 = SearchData.row4327.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_40 : scan edge (search edge 3) 4 [13] 13 innerTail3_40 best = best := by
  rw [innerTail3_40, scan, if_neg (by decide), innerCandidates3_40_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_41
    (if 3 ≤ innerCandidates3_40.length then search edge 3 [13, 4327] innerCandidates3_40 4340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_41

def innerTail3_39 : List Nat := 4243 :: innerTail3_40
def innerCandidates3_39 : List Nat := [4591]
set_option maxRecDepth 4096 in
theorem innerCandidates3_39_checked : candidates edge 4 13 4243 innerTail3_40 best.bound = innerCandidates3_39 := by
  unfold candidates
  have row : edge 4243 = SearchData.row4243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_39 : scan edge (search edge 3) 4 [13] 13 innerTail3_39 best = best := by
  rw [innerTail3_39, scan, if_neg (by decide), innerCandidates3_39_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_40
    (if 3 ≤ innerCandidates3_39.length then search edge 3 [13, 4243] innerCandidates3_39 4256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_40

def innerTail3_38 : List Nat := 4219 :: innerTail3_39
def innerCandidates3_38 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_38_checked : candidates edge 4 13 4219 innerTail3_39 best.bound = innerCandidates3_38 := by
  unfold candidates
  have row : edge 4219 = SearchData.row4219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_38 : scan edge (search edge 3) 4 [13] 13 innerTail3_38 best = best := by
  rw [innerTail3_38, scan, if_neg (by decide), innerCandidates3_38_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_39
    (if 3 ≤ innerCandidates3_38.length then search edge 3 [13, 4219] innerCandidates3_38 4232 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_39

def innerTail3_37 : List Nat := 3967 :: innerTail3_38
def innerCandidates3_37 : List Nat := [7927, 8053]
set_option maxRecDepth 4096 in
theorem innerCandidates3_37_checked : candidates edge 4 13 3967 innerTail3_38 best.bound = innerCandidates3_37 := by
  unfold candidates
  have row : edge 3967 = SearchData.row3967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_37 : scan edge (search edge 3) 4 [13] 13 innerTail3_37 best = best := by
  rw [innerTail3_37, scan, if_neg (by decide), innerCandidates3_37_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_38
    (if 3 ≤ innerCandidates3_37.length then search edge 3 [13, 3967] innerCandidates3_37 3980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_38

def innerTail3_36 : List Nat := 3673 :: innerTail3_37
def innerCandidates3_36 : List Nat := [5581, 6691]
set_option maxRecDepth 4096 in
theorem innerCandidates3_36_checked : candidates edge 4 13 3673 innerTail3_37 best.bound = innerCandidates3_36 := by
  unfold candidates
  have row : edge 3673 = SearchData.row3673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_36 : scan edge (search edge 3) 4 [13] 13 innerTail3_36 best = best := by
  rw [innerTail3_36, scan, if_neg (by decide), innerCandidates3_36_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_37
    (if 3 ≤ innerCandidates3_36.length then search edge 3 [13, 3673] innerCandidates3_36 3686 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_37

def innerTail3_35 : List Nat := 3583 :: innerTail3_36
def innerCandidates3_35 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_35_checked : candidates edge 4 13 3583 innerTail3_36 best.bound = innerCandidates3_35 := by
  unfold candidates
  have row : edge 3583 = SearchData.row3583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_35 : scan edge (search edge 3) 4 [13] 13 innerTail3_35 best = best := by
  rw [innerTail3_35, scan, if_neg (by decide), innerCandidates3_35_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_36
    (if 3 ≤ innerCandidates3_35.length then search edge 3 [13, 3583] innerCandidates3_35 3596 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_36

def innerTail3_34 : List Nat := 3559 :: innerTail3_35
def innerCandidates3_34 : List Nat := [3583, 8053]
set_option maxRecDepth 4096 in
theorem innerCandidates3_34_checked : candidates edge 4 13 3559 innerTail3_35 best.bound = innerCandidates3_34 := by
  unfold candidates
  have row : edge 3559 = SearchData.row3559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_34 : scan edge (search edge 3) 4 [13] 13 innerTail3_34 best = best := by
  rw [innerTail3_34, scan, if_neg (by decide), innerCandidates3_34_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_35
    (if 3 ≤ innerCandidates3_34.length then search edge 3 [13, 3559] innerCandidates3_34 3572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_35

def innerTail3_33 : List Nat := 3499 :: innerTail3_34
def innerCandidates3_33 : List Nat := [7993]
set_option maxRecDepth 4096 in
theorem innerCandidates3_33_checked : candidates edge 4 13 3499 innerTail3_34 best.bound = innerCandidates3_33 := by
  unfold candidates
  have row : edge 3499 = SearchData.row3499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_33 : scan edge (search edge 3) 4 [13] 13 innerTail3_33 best = best := by
  rw [innerTail3_33, scan, if_neg (by decide), innerCandidates3_33_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_34
    (if 3 ≤ innerCandidates3_33.length then search edge 3 [13, 3499] innerCandidates3_33 3512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_34

def innerTail3_32 : List Nat := 3187 :: innerTail3_33
def innerCandidates3_32 : List Nat := [5209, 7477]
set_option maxRecDepth 4096 in
theorem innerCandidates3_32_checked : candidates edge 4 13 3187 innerTail3_33 best.bound = innerCandidates3_32 := by
  unfold candidates
  have row : edge 3187 = SearchData.row3187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_32 : scan edge (search edge 3) 4 [13] 13 innerTail3_32 best = best := by
  rw [innerTail3_32, scan, if_neg (by decide), innerCandidates3_32_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_33
    (if 3 ≤ innerCandidates3_32.length then search edge 3 [13, 3187] innerCandidates3_32 3200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_33

def innerTail3_31 : List Nat := 2971 :: innerTail3_32
def innerCandidates3_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_31_checked : candidates edge 4 13 2971 innerTail3_32 best.bound = innerCandidates3_31 := by
  unfold candidates
  have row : edge 2971 = SearchData.row2971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_31 : scan edge (search edge 3) 4 [13] 13 innerTail3_31 best = best := by
  rw [innerTail3_31, scan, if_neg (by decide), innerCandidates3_31_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_32
    (if 3 ≤ innerCandidates3_31.length then search edge 3 [13, 2971] innerCandidates3_31 2984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_32

def innerTail3_30 : List Nat := 2953 :: innerTail3_31
def innerCandidates3_30 : List Nat := [6733]
set_option maxRecDepth 4096 in
theorem innerCandidates3_30_checked : candidates edge 4 13 2953 innerTail3_31 best.bound = innerCandidates3_30 := by
  unfold candidates
  have row : edge 2953 = SearchData.row2953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_30 : scan edge (search edge 3) 4 [13] 13 innerTail3_30 best = best := by
  rw [innerTail3_30, scan, if_neg (by decide), innerCandidates3_30_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_31
    (if 3 ≤ innerCandidates3_30.length then search edge 3 [13, 2953] innerCandidates3_30 2966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_31

def innerTail3_29 : List Nat := 2851 :: innerTail3_30
def innerCandidates3_29 : List Nat := [3187, 4951]
set_option maxRecDepth 4096 in
theorem innerCandidates3_29_checked : candidates edge 4 13 2851 innerTail3_30 best.bound = innerCandidates3_29 := by
  unfold candidates
  have row : edge 2851 = SearchData.row2851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_29 : scan edge (search edge 3) 4 [13] 13 innerTail3_29 best = best := by
  rw [innerTail3_29, scan, if_neg (by decide), innerCandidates3_29_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_30
    (if 3 ≤ innerCandidates3_29.length then search edge 3 [13, 2851] innerCandidates3_29 2864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_30

def innerTail3_28 : List Nat := 2689 :: innerTail3_29
def innerCandidates3_28 : List Nat := [6043, 7477]
set_option maxRecDepth 4096 in
theorem innerCandidates3_28_checked : candidates edge 4 13 2689 innerTail3_29 best.bound = innerCandidates3_28 := by
  unfold candidates
  have row : edge 2689 = SearchData.row2689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_28 : scan edge (search edge 3) 4 [13] 13 innerTail3_28 best = best := by
  rw [innerTail3_28, scan, if_neg (by decide), innerCandidates3_28_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_29
    (if 3 ≤ innerCandidates3_28.length then search edge 3 [13, 2689] innerCandidates3_28 2702 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_29

def innerTail3_27 : List Nat := 2383 :: innerTail3_28
def innerCandidates3_27 : List Nat := [3583, 5431, 6733]
set_option maxRecDepth 4096 in
theorem innerCandidates3_27_checked : candidates edge 4 13 2383 innerTail3_28 best.bound = innerCandidates3_27 := by
  unfold candidates
  have row : edge 2383 = PrimePairCache.smallRow2383.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild3_27 : search edge 3 [13, 2383] innerCandidates3_27 2396 best = best := by decide

theorem innerScan3_27 : scan edge (search edge 3) 4 [13] 13 innerTail3_27 best = best := by
  rw [innerTail3_27, scan, if_neg (by decide), innerCandidates3_27_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_28
    (if 3 ≤ innerCandidates3_27.length then search edge 3 [13, 2383] innerCandidates3_27 2396 best else best) = best
  rw [if_pos (by decide), innerChild3_27]
  exact innerScan3_28

def innerTail3_26 : List Nat := 2347 :: innerTail3_27
def innerCandidates3_26 : List Nat := [2851, 4327]
set_option maxRecDepth 4096 in
theorem innerCandidates3_26_checked : candidates edge 4 13 2347 innerTail3_27 best.bound = innerCandidates3_26 := by
  unfold candidates
  have row : edge 2347 = PrimePairCache.smallRow2347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_26 : scan edge (search edge 3) 4 [13] 13 innerTail3_26 best = best := by
  rw [innerTail3_26, scan, if_neg (by decide), innerCandidates3_26_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_27
    (if 3 ≤ innerCandidates3_26.length then search edge 3 [13, 2347] innerCandidates3_26 2360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_27

def innerTail3_25 : List Nat := 2287 :: innerTail3_26
def innerCandidates3_25 : List Nat := [3187, 5077, 5431]
set_option maxRecDepth 4096 in
theorem innerCandidates3_25_checked : candidates edge 4 13 2287 innerTail3_26 best.bound = innerCandidates3_25 := by
  unfold candidates
  have row : edge 2287 = PrimePairCache.smallRow2287.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild3_25 : search edge 3 [13, 2287] innerCandidates3_25 2300 best = best := by decide

theorem innerScan3_25 : scan edge (search edge 3) 4 [13] 13 innerTail3_25 best = best := by
  rw [innerTail3_25, scan, if_neg (by decide), innerCandidates3_25_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_26
    (if 3 ≤ innerCandidates3_25.length then search edge 3 [13, 2287] innerCandidates3_25 2300 best else best) = best
  rw [if_pos (by decide), innerChild3_25]
  exact innerScan3_26

def innerTail3_24 : List Nat := 2137 :: innerTail3_25
def innerCandidates3_24 : List Nat := [4513, 6991]
set_option maxRecDepth 4096 in
theorem innerCandidates3_24_checked : candidates edge 4 13 2137 innerTail3_25 best.bound = innerCandidates3_24 := by
  unfold candidates
  have row : edge 2137 = PrimePairCache.smallRow2137.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_24 : scan edge (search edge 3) 4 [13] 13 innerTail3_24 best = best := by
  rw [innerTail3_24, scan, if_neg (by decide), innerCandidates3_24_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_25
    (if 3 ≤ innerCandidates3_24.length then search edge 3 [13, 2137] innerCandidates3_24 2150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_25

def innerTail3_23 : List Nat := 2113 :: innerTail3_24
def innerCandidates3_23 : List Nat := [3583, 5431, 7927]
set_option maxRecDepth 4096 in
theorem innerCandidates3_23_checked : candidates edge 4 13 2113 innerTail3_24 best.bound = innerCandidates3_23 := by
  unfold candidates
  have row : edge 2113 = PrimePairCache.smallRow2113.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild3_23 : search edge 3 [13, 2113] innerCandidates3_23 2126 best = best := by decide

theorem innerScan3_23 : scan edge (search edge 3) 4 [13] 13 innerTail3_23 best = best := by
  rw [innerTail3_23, scan, if_neg (by decide), innerCandidates3_23_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_24
    (if 3 ≤ innerCandidates3_23.length then search edge 3 [13, 2113] innerCandidates3_23 2126 best else best) = best
  rw [if_pos (by decide), innerChild3_23]
  exact innerScan3_24

def innerTail3_22 : List Nat := 1861 :: innerTail3_23
def innerCandidates3_22 : List Nat := [2383, 4789]
set_option maxRecDepth 4096 in
theorem innerCandidates3_22_checked : candidates edge 4 13 1861 innerTail3_23 best.bound = innerCandidates3_22 := by
  unfold candidates
  have row : edge 1861 = PrimePairCache.smallRow1861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_22 : scan edge (search edge 3) 4 [13] 13 innerTail3_22 best = best := by
  rw [innerTail3_22, scan, if_neg (by decide), innerCandidates3_22_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_23
    (if 3 ≤ innerCandidates3_22.length then search edge 3 [13, 1861] innerCandidates3_22 1874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_23

def innerTail3_21 : List Nat := 1627 :: innerTail3_22
def innerCandidates3_21 : List Nat := [4591, 7507]
set_option maxRecDepth 4096 in
theorem innerCandidates3_21_checked : candidates edge 4 13 1627 innerTail3_22 best.bound = innerCandidates3_21 := by
  unfold candidates
  have row : edge 1627 = PrimePairCache.smallRow1627.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_21 : scan edge (search edge 3) 4 [13] 13 innerTail3_21 best = best := by
  rw [innerTail3_21, scan, if_neg (by decide), innerCandidates3_21_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_22
    (if 3 ≤ innerCandidates3_21.length then search edge 3 [13, 1627] innerCandidates3_21 1640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_22

def innerTail3_20 : List Nat := 1543 :: innerTail3_21
def innerCandidates3_20 : List Nat := [3967, 8053]
set_option maxRecDepth 4096 in
theorem innerCandidates3_20_checked : candidates edge 4 13 1543 innerTail3_21 best.bound = innerCandidates3_20 := by
  unfold candidates
  have row : edge 1543 = PrimePairCache.smallRow1543.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_20 : scan edge (search edge 3) 4 [13] 13 innerTail3_20 best = best := by
  rw [innerTail3_20, scan, if_neg (by decide), innerCandidates3_20_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_21
    (if 3 ≤ innerCandidates3_20.length then search edge 3 [13, 1543] innerCandidates3_20 1556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_21

def innerTail3_19 : List Nat := 1489 :: innerTail3_20
def innerCandidates3_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_19_checked : candidates edge 4 13 1489 innerTail3_20 best.bound = innerCandidates3_19 := by
  unfold candidates
  have row : edge 1489 = PrimePairCache.smallRow1489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_19 : scan edge (search edge 3) 4 [13] 13 innerTail3_19 best = best := by
  rw [innerTail3_19, scan, if_neg (by decide), innerCandidates3_19_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_20
    (if 3 ≤ innerCandidates3_19.length then search edge 3 [13, 1489] innerCandidates3_19 1502 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_20

def innerTail3_18 : List Nat := 1381 :: innerTail3_19
def innerCandidates3_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_18_checked : candidates edge 4 13 1381 innerTail3_19 best.bound = innerCandidates3_18 := by
  unfold candidates
  have row : edge 1381 = PrimePairCache.smallRow1381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_18 : scan edge (search edge 3) 4 [13] 13 innerTail3_18 best = best := by
  rw [innerTail3_18, scan, if_neg (by decide), innerCandidates3_18_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_19
    (if 3 ≤ innerCandidates3_18.length then search edge 3 [13, 1381] innerCandidates3_18 1394 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_19

def innerTail3_17 : List Nat := 1321 :: innerTail3_18
def innerCandidates3_17 : List Nat := [1381, 5581]
set_option maxRecDepth 4096 in
theorem innerCandidates3_17_checked : candidates edge 4 13 1321 innerTail3_18 best.bound = innerCandidates3_17 := by
  unfold candidates
  have row : edge 1321 = PrimePairCache.smallRow1321.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_17 : scan edge (search edge 3) 4 [13] 13 innerTail3_17 best = best := by
  rw [innerTail3_17, scan, if_neg (by decide), innerCandidates3_17_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_18
    (if 3 ≤ innerCandidates3_17.length then search edge 3 [13, 1321] innerCandidates3_17 1334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_18

def innerTail3_16 : List Nat := 1231 :: innerTail3_17
def innerCandidates3_16 : List Nat := [2689, 4243, 4327, 4789, 5647]
set_option maxRecDepth 4096 in
theorem innerCandidates3_16_checked : candidates edge 4 13 1231 innerTail3_17 best.bound = innerCandidates3_16 := by
  unfold candidates
  have row : edge 1231 = PrimePairCache.smallRow1231.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild3_16 : search edge 3 [13, 1231] innerCandidates3_16 1244 best = best := by decide

theorem innerScan3_16 : scan edge (search edge 3) 4 [13] 13 innerTail3_16 best = best := by
  rw [innerTail3_16, scan, if_neg (by decide), innerCandidates3_16_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_17
    (if 3 ≤ innerCandidates3_16.length then search edge 3 [13, 1231] innerCandidates3_16 1244 best else best) = best
  rw [if_pos (by decide), innerChild3_16]
  exact innerScan3_17

def innerTail3_15 : List Nat := 1213 :: innerTail3_16
def innerCandidates3_15 : List Nat := [2851, 6043]
set_option maxRecDepth 4096 in
theorem innerCandidates3_15_checked : candidates edge 4 13 1213 innerTail3_16 best.bound = innerCandidates3_15 := by
  unfold candidates
  have row : edge 1213 = PrimePairCache.smallRow1213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_15 : scan edge (search edge 3) 4 [13] 13 innerTail3_15 best = best := by
  rw [innerTail3_15, scan, if_neg (by decide), innerCandidates3_15_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_16
    (if 3 ≤ innerCandidates3_15.length then search edge 3 [13, 1213] innerCandidates3_15 1226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_16

def innerTail3_14 : List Nat := 1129 :: innerTail3_15
def innerCandidates3_14 : List Nat := [6421]
set_option maxRecDepth 4096 in
theorem innerCandidates3_14_checked : candidates edge 4 13 1129 innerTail3_15 best.bound = innerCandidates3_14 := by
  unfold candidates
  have row : edge 1129 = PrimePairCache.smallRow1129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_14 : scan edge (search edge 3) 4 [13] 13 innerTail3_14 best = best := by
  rw [innerTail3_14, scan, if_neg (by decide), innerCandidates3_14_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_15
    (if 3 ≤ innerCandidates3_14.length then search edge 3 [13, 1129] innerCandidates3_14 1142 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_15

def innerTail3_13 : List Nat := 1009 :: innerTail3_14
def innerCandidates3_13 : List Nat := [2851, 6043, 6397]
set_option maxRecDepth 4096 in
theorem innerCandidates3_13_checked : candidates edge 4 13 1009 innerTail3_14 best.bound = innerCandidates3_13 := by
  unfold candidates
  have row : edge 1009 = PrimePairCache.smallRow1009.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild3_13 : search edge 3 [13, 1009] innerCandidates3_13 1022 best = best := by decide

theorem innerScan3_13 : scan edge (search edge 3) 4 [13] 13 innerTail3_13 best = best := by
  rw [innerTail3_13, scan, if_neg (by decide), innerCandidates3_13_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_14
    (if 3 ≤ innerCandidates3_13.length then search edge 3 [13, 1009] innerCandidates3_13 1022 best else best) = best
  rw [if_pos (by decide), innerChild3_13]
  exact innerScan3_14

def innerTail3_12 : List Nat := 997 :: innerTail3_13
def innerCandidates3_12 : List Nat := [4243, 4513]
set_option maxRecDepth 4096 in
theorem innerCandidates3_12_checked : candidates edge 4 13 997 innerTail3_13 best.bound = innerCandidates3_12 := by
  unfold candidates
  have row : edge 997 = PrimePairCache.smallRow997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_12 : scan edge (search edge 3) 4 [13] 13 innerTail3_12 best = best := by
  rw [innerTail3_12, scan, if_neg (by decide), innerCandidates3_12_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_13
    (if 3 ≤ innerCandidates3_12.length then search edge 3 [13, 997] innerCandidates3_12 1010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_13

def innerTail3_11 : List Nat := 829 :: innerTail3_12
def innerCandidates3_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_11_checked : candidates edge 4 13 829 innerTail3_12 best.bound = innerCandidates3_11 := by
  unfold candidates
  have row : edge 829 = PrimePairCache.smallRow829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_11 : scan edge (search edge 3) 4 [13] 13 innerTail3_11 best = best := by
  rw [innerTail3_11, scan, if_neg (by decide), innerCandidates3_11_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_12
    (if 3 ≤ innerCandidates3_11.length then search edge 3 [13, 829] innerCandidates3_11 842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_12

def innerTail3_10 : List Nat := 709 :: innerTail3_11
def innerCandidates3_10 : List Nat := [1543]
set_option maxRecDepth 4096 in
theorem innerCandidates3_10_checked : candidates edge 4 13 709 innerTail3_11 best.bound = innerCandidates3_10 := by
  unfold candidates
  have row : edge 709 = PrimePairCache.smallRow709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_10 : scan edge (search edge 3) 4 [13] 13 innerTail3_10 best = best := by
  rw [innerTail3_10, scan, if_neg (by decide), innerCandidates3_10_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_11
    (if 3 ≤ innerCandidates3_10.length then search edge 3 [13, 709] innerCandidates3_10 722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_11

def innerTail3_9 : List Nat := 577 :: innerTail3_10
def innerCandidates3_9 : List Nat := [5431]
set_option maxRecDepth 4096 in
theorem innerCandidates3_9_checked : candidates edge 4 13 577 innerTail3_10 best.bound = innerCandidates3_9 := by
  unfold candidates
  have row : edge 577 = PrimePairCache.smallRow577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_9 : scan edge (search edge 3) 4 [13] 13 innerTail3_9 best = best := by
  rw [innerTail3_9, scan, if_neg (by decide), innerCandidates3_9_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_10
    (if 3 ≤ innerCandidates3_9.length then search edge 3 [13, 577] innerCandidates3_9 590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_10

def innerTail3_8 : List Nat := 523 :: innerTail3_9
def innerCandidates3_8 : List Nat := [577, 1129, 2113, 3499, 4591, 4909, 5851]
set_option maxRecDepth 4096 in
theorem innerCandidates3_8_checked : candidates edge 4 13 523 innerTail3_9 best.bound = innerCandidates3_8 := by
  unfold candidates
  have row : edge 523 = PrimePairCache.smallRow523.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild3_8 : search edge 3 [13, 523] innerCandidates3_8 536 best = best := by decide

theorem innerScan3_8 : scan edge (search edge 3) 4 [13] 13 innerTail3_8 best = best := by
  rw [innerTail3_8, scan, if_neg (by decide), innerCandidates3_8_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_9
    (if 3 ≤ innerCandidates3_8.length then search edge 3 [13, 523] innerCandidates3_8 536 best else best) = best
  rw [if_pos (by decide), innerChild3_8]
  exact innerScan3_9

def innerTail3_7 : List Nat := 367 :: innerTail3_8
def innerCandidates3_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates3_7_checked : candidates edge 4 13 367 innerTail3_8 best.bound = innerCandidates3_7 := by
  unfold candidates
  have row : edge 367 = PrimePairCache.smallRow367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_7 : scan edge (search edge 3) 4 [13] 13 innerTail3_7 best = best := by
  rw [innerTail3_7, scan, if_neg (by decide), innerCandidates3_7_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_8
    (if 3 ≤ innerCandidates3_7.length then search edge 3 [13, 367] innerCandidates3_7 380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_8

def innerTail3_6 : List Nat := 337 :: innerTail3_7
def innerCandidates3_6 : List Nat := [3499, 5431, 7723]
set_option maxRecDepth 4096 in
theorem innerCandidates3_6_checked : candidates edge 4 13 337 innerTail3_7 best.bound = innerCandidates3_6 := by
  unfold candidates
  have row : edge 337 = PrimePairCache.smallRow337.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild3_6 : search edge 3 [13, 337] innerCandidates3_6 350 best = best := by decide

theorem innerScan3_6 : scan edge (search edge 3) 4 [13] 13 innerTail3_6 best = best := by
  rw [innerTail3_6, scan, if_neg (by decide), innerCandidates3_6_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_7
    (if 3 ≤ innerCandidates3_6.length then search edge 3 [13, 337] innerCandidates3_6 350 best else best) = best
  rw [if_pos (by decide), innerChild3_6]
  exact innerScan3_7

def innerTail3_5 : List Nat := 331 :: innerTail3_6
def innerCandidates3_5 : List Nat := [577, 4513]
set_option maxRecDepth 4096 in
theorem innerCandidates3_5_checked : candidates edge 4 13 331 innerTail3_6 best.bound = innerCandidates3_5 := by
  unfold candidates
  have row : edge 331 = PrimePairCache.smallRow331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_5 : scan edge (search edge 3) 4 [13] 13 innerTail3_5 best = best := by
  rw [innerTail3_5, scan, if_neg (by decide), innerCandidates3_5_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_6
    (if 3 ≤ innerCandidates3_5.length then search edge 3 [13, 331] innerCandidates3_5 344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_6

def innerTail3_4 : List Nat := 241 :: innerTail3_5
def innerCandidates3_4 : List Nat := [5431]
set_option maxRecDepth 4096 in
theorem innerCandidates3_4_checked : candidates edge 4 13 241 innerTail3_5 best.bound = innerCandidates3_4 := by
  unfold candidates
  have row : edge 241 = PrimePairCache.smallRow241.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_4 : scan edge (search edge 3) 4 [13] 13 innerTail3_4 best = best := by
  rw [innerTail3_4, scan, if_neg (by decide), innerCandidates3_4_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_5
    (if 3 ≤ innerCandidates3_4.length then search edge 3 [13, 241] innerCandidates3_4 254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_5

def innerTail3_3 : List Nat := 127 :: innerTail3_4
def innerCandidates3_3 : List Nat := [241, 331, 1129, 3673, 4363, 6733, 7723, 7993]
set_option maxRecDepth 4096 in
theorem innerCandidates3_3_checked : candidates edge 4 13 127 innerTail3_4 best.bound = innerCandidates3_3 := by
  unfold candidates
  have row : edge 127 = PrimePairCache.smallRow127.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild3_3 : search edge 3 [13, 127] innerCandidates3_3 140 best = best := by decide

theorem innerScan3_3 : scan edge (search edge 3) 4 [13] 13 innerTail3_3 best = best := by
  rw [innerTail3_3, scan, if_neg (by decide), innerCandidates3_3_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_4
    (if 3 ≤ innerCandidates3_3.length then search edge 3 [13, 127] innerCandidates3_3 140 best else best) = best
  rw [if_pos (by decide), innerChild3_3]
  exact innerScan3_4

def innerTail3_2 : List Nat := 103 :: innerTail3_3
def innerCandidates3_2 : List Nat := [997]
set_option maxRecDepth 4096 in
theorem innerCandidates3_2_checked : candidates edge 4 13 103 innerTail3_3 best.bound = innerCandidates3_2 := by
  unfold candidates
  have row : edge 103 = PrimePairCache.smallRow103.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan3_2 : scan edge (search edge 3) 4 [13] 13 innerTail3_2 best = best := by
  rw [innerTail3_2, scan, if_neg (by decide), innerCandidates3_2_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_3
    (if 3 ≤ innerCandidates3_2.length then search edge 3 [13, 103] innerCandidates3_2 116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan3_3

def innerTail3_1 : List Nat := 61 :: innerTail3_2
def innerCandidates3_1 : List Nat := [331, 2383, 2971, 5431, 5623]
set_option maxRecDepth 4096 in
theorem innerCandidates3_1_checked : candidates edge 4 13 61 innerTail3_2 best.bound = innerCandidates3_1 := by
  unfold candidates
  have row : edge 61 = PrimePairCache.smallRow61.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild3_1 : search edge 3 [13, 61] innerCandidates3_1 74 best = best := by decide

theorem innerScan3_1 : scan edge (search edge 3) 4 [13] 13 innerTail3_1 best = best := by
  rw [innerTail3_1, scan, if_neg (by decide), innerCandidates3_1_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_2
    (if 3 ≤ innerCandidates3_1.length then search edge 3 [13, 61] innerCandidates3_1 74 best else best) = best
  rw [if_pos (by decide), innerChild3_1]
  exact innerScan3_2

def innerTail3_0 : List Nat := 19 :: innerTail3_1
def innerCandidates3_0 : List Nat := [577, 709, 997, 1861, 2113, 5077, 6043, 6991, 7507, 8179]
set_option maxRecDepth 4096 in
theorem innerCandidates3_0_checked : candidates edge 4 13 19 innerTail3_1 best.bound = innerCandidates3_0 := by
  unfold candidates
  have row : edge 19 = PrimePairCache.smallRow19.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild3_0 : search edge 3 [13, 19] innerCandidates3_0 32 best = best := by decide

theorem innerScan3_0 : scan edge (search edge 3) 4 [13] 13 innerTail3_0 best = best := by
  rw [innerTail3_0, scan, if_neg (by decide), innerCandidates3_0_checked]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_1
    (if 3 ≤ innerCandidates3_0.length then search edge 3 [13, 19] innerCandidates3_0 32 best else best) = best
  rw [if_pos (by decide), innerChild3_0]
  exact innerScan3_1

theorem rootBranch3 : search edge 4 [13] rootCandidates3 13 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [13] 13 innerTail3_0 best = best
  exact innerScan3_0

end ProjectEulerSolutions.PrimePairUpper.Search
