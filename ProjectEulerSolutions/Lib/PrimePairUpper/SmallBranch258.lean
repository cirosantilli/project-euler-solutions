import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates9 : List Nat := [67, 79, 199, 277, 313, 463, 607, 967, 991, 1069, 1153, 1237, 1303, 1321, 1447, 1471, 1549, 1699, 1873, 1951, 2137, 2179, 2269, 2371, 2377, 2473, 2539, 2677, 2689, 2707, 2803, 3049, 3181, 3187, 3463, 3613, 4093, 4159, 4177, 4603, 4729, 4789, 4909, 5101, 5113, 5227, 5281, 5449, 5623, 5779, 5857, 5923, 6133, 6373, 6469, 6529, 6823, 7297, 7369, 7393, 7477, 7537, 7873, 7963, 8011, 8089, 8101]
set_option maxRecDepth 4096 in
theorem rootCandidates9_checked : candidates edge 5 0 37 tail10 best.bound = rootCandidates9 := by
  unfold candidates
  have row : edge 37 = PrimePairCache.smallRow37.contains := by funext q; rfl
  rw [row]
  decide

def innerTail9_67 : List Nat := []
theorem innerScan9_67 : scan edge (search edge 3) 4 [37] 37 innerTail9_67 best = best := rfl

def innerTail9_66 : List Nat := 8101 :: innerTail9_67
def innerCandidates9_66 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_66_checked : candidates edge 4 37 8101 innerTail9_67 best.bound = innerCandidates9_66 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_66 : scan edge (search edge 3) 4 [37] 37 innerTail9_66 best = best := by
  rw [innerTail9_66, scan, if_neg (by decide), innerCandidates9_66_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_67
    (if 3 ≤ innerCandidates9_66.length then search edge 3 [37, 8101] innerCandidates9_66 8138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_67

def innerTail9_65 : List Nat := 8089 :: innerTail9_66
def innerCandidates9_65 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_65_checked : candidates edge 4 37 8089 innerTail9_66 best.bound = innerCandidates9_65 := by
  unfold candidates
  have row : edge 8089 = SearchData.row8089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_65 : scan edge (search edge 3) 4 [37] 37 innerTail9_65 best = best := by
  rw [innerTail9_65, scan, if_neg (by decide), innerCandidates9_65_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_66
    (if 3 ≤ innerCandidates9_65.length then search edge 3 [37, 8089] innerCandidates9_65 8126 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_66

def innerTail9_64 : List Nat := 8011 :: innerTail9_65
def innerCandidates9_64 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_64_checked : candidates edge 4 37 8011 innerTail9_65 best.bound = innerCandidates9_64 := by
  unfold candidates
  have row : edge 8011 = SearchData.row8011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_64 : scan edge (search edge 3) 4 [37] 37 innerTail9_64 best = best := by
  rw [innerTail9_64, scan, if_neg (by decide), innerCandidates9_64_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_65
    (if 3 ≤ innerCandidates9_64.length then search edge 3 [37, 8011] innerCandidates9_64 8048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_65

def innerTail9_63 : List Nat := 7963 :: innerTail9_64
def innerCandidates9_63 : List Nat := [8011]
set_option maxRecDepth 4096 in
theorem innerCandidates9_63_checked : candidates edge 4 37 7963 innerTail9_64 best.bound = innerCandidates9_63 := by
  unfold candidates
  have row : edge 7963 = SearchData.row7963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_63 : scan edge (search edge 3) 4 [37] 37 innerTail9_63 best = best := by
  rw [innerTail9_63, scan, if_neg (by decide), innerCandidates9_63_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_64
    (if 3 ≤ innerCandidates9_63.length then search edge 3 [37, 7963] innerCandidates9_63 8000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_64

def innerTail9_62 : List Nat := 7873 :: innerTail9_63
def innerCandidates9_62 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_62_checked : candidates edge 4 37 7873 innerTail9_63 best.bound = innerCandidates9_62 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_62 : scan edge (search edge 3) 4 [37] 37 innerTail9_62 best = best := by
  rw [innerTail9_62, scan, if_neg (by decide), innerCandidates9_62_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_63
    (if 3 ≤ innerCandidates9_62.length then search edge 3 [37, 7873] innerCandidates9_62 7910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_63

def innerTail9_61 : List Nat := 7537 :: innerTail9_62
def innerCandidates9_61 : List Nat := [8101]
set_option maxRecDepth 4096 in
theorem innerCandidates9_61_checked : candidates edge 4 37 7537 innerTail9_62 best.bound = innerCandidates9_61 := by
  unfold candidates
  have row : edge 7537 = SearchData.row7537.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_61 : scan edge (search edge 3) 4 [37] 37 innerTail9_61 best = best := by
  rw [innerTail9_61, scan, if_neg (by decide), innerCandidates9_61_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_62
    (if 3 ≤ innerCandidates9_61.length then search edge 3 [37, 7537] innerCandidates9_61 7574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_62

def innerTail9_60 : List Nat := 7477 :: innerTail9_61
def innerCandidates9_60 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_60_checked : candidates edge 4 37 7477 innerTail9_61 best.bound = innerCandidates9_60 := by
  unfold candidates
  have row : edge 7477 = SearchData.row7477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_60 : scan edge (search edge 3) 4 [37] 37 innerTail9_60 best = best := by
  rw [innerTail9_60, scan, if_neg (by decide), innerCandidates9_60_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_61
    (if 3 ≤ innerCandidates9_60.length then search edge 3 [37, 7477] innerCandidates9_60 7514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_61

def innerTail9_59 : List Nat := 7393 :: innerTail9_60
def innerCandidates9_59 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_59_checked : candidates edge 4 37 7393 innerTail9_60 best.bound = innerCandidates9_59 := by
  unfold candidates
  have row : edge 7393 = SearchData.row7393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_59 : scan edge (search edge 3) 4 [37] 37 innerTail9_59 best = best := by
  rw [innerTail9_59, scan, if_neg (by decide), innerCandidates9_59_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_60
    (if 3 ≤ innerCandidates9_59.length then search edge 3 [37, 7393] innerCandidates9_59 7430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_60

def innerTail9_58 : List Nat := 7369 :: innerTail9_59
def innerCandidates9_58 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_58_checked : candidates edge 4 37 7369 innerTail9_59 best.bound = innerCandidates9_58 := by
  unfold candidates
  have row : edge 7369 = SearchData.row7369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_58 : scan edge (search edge 3) 4 [37] 37 innerTail9_58 best = best := by
  rw [innerTail9_58, scan, if_neg (by decide), innerCandidates9_58_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_59
    (if 3 ≤ innerCandidates9_58.length then search edge 3 [37, 7369] innerCandidates9_58 7406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_59

def innerTail9_57 : List Nat := 7297 :: innerTail9_58
def innerCandidates9_57 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_57_checked : candidates edge 4 37 7297 innerTail9_58 best.bound = innerCandidates9_57 := by
  unfold candidates
  have row : edge 7297 = SearchData.row7297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_57 : scan edge (search edge 3) 4 [37] 37 innerTail9_57 best = best := by
  rw [innerTail9_57, scan, if_neg (by decide), innerCandidates9_57_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_58
    (if 3 ≤ innerCandidates9_57.length then search edge 3 [37, 7297] innerCandidates9_57 7334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_58

def innerTail9_56 : List Nat := 6823 :: innerTail9_57
def innerCandidates9_56 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_56_checked : candidates edge 4 37 6823 innerTail9_57 best.bound = innerCandidates9_56 := by
  unfold candidates
  have row : edge 6823 = SearchData.row6823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_56 : scan edge (search edge 3) 4 [37] 37 innerTail9_56 best = best := by
  rw [innerTail9_56, scan, if_neg (by decide), innerCandidates9_56_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_57
    (if 3 ≤ innerCandidates9_56.length then search edge 3 [37, 6823] innerCandidates9_56 6860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_57

def innerTail9_55 : List Nat := 6529 :: innerTail9_56
def innerCandidates9_55 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_55_checked : candidates edge 4 37 6529 innerTail9_56 best.bound = innerCandidates9_55 := by
  unfold candidates
  have row : edge 6529 = SearchData.row6529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_55 : scan edge (search edge 3) 4 [37] 37 innerTail9_55 best = best := by
  rw [innerTail9_55, scan, if_neg (by decide), innerCandidates9_55_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_56
    (if 3 ≤ innerCandidates9_55.length then search edge 3 [37, 6529] innerCandidates9_55 6566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_56

def innerTail9_54 : List Nat := 6469 :: innerTail9_55
def innerCandidates9_54 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_54_checked : candidates edge 4 37 6469 innerTail9_55 best.bound = innerCandidates9_54 := by
  unfold candidates
  have row : edge 6469 = SearchData.row6469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_54 : scan edge (search edge 3) 4 [37] 37 innerTail9_54 best = best := by
  rw [innerTail9_54, scan, if_neg (by decide), innerCandidates9_54_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_55
    (if 3 ≤ innerCandidates9_54.length then search edge 3 [37, 6469] innerCandidates9_54 6506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_55

def innerTail9_53 : List Nat := 6373 :: innerTail9_54
def innerCandidates9_53 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_53_checked : candidates edge 4 37 6373 innerTail9_54 best.bound = innerCandidates9_53 := by
  unfold candidates
  have row : edge 6373 = SearchData.row6373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_53 : scan edge (search edge 3) 4 [37] 37 innerTail9_53 best = best := by
  rw [innerTail9_53, scan, if_neg (by decide), innerCandidates9_53_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_54
    (if 3 ≤ innerCandidates9_53.length then search edge 3 [37, 6373] innerCandidates9_53 6410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_54

def innerTail9_52 : List Nat := 6133 :: innerTail9_53
def innerCandidates9_52 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_52_checked : candidates edge 4 37 6133 innerTail9_53 best.bound = innerCandidates9_52 := by
  unfold candidates
  have row : edge 6133 = SearchData.row6133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_52 : scan edge (search edge 3) 4 [37] 37 innerTail9_52 best = best := by
  rw [innerTail9_52, scan, if_neg (by decide), innerCandidates9_52_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_53
    (if 3 ≤ innerCandidates9_52.length then search edge 3 [37, 6133] innerCandidates9_52 6170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_53

def innerTail9_51 : List Nat := 5923 :: innerTail9_52
def innerCandidates9_51 : List Nat := [7963]
set_option maxRecDepth 4096 in
theorem innerCandidates9_51_checked : candidates edge 4 37 5923 innerTail9_52 best.bound = innerCandidates9_51 := by
  unfold candidates
  have row : edge 5923 = SearchData.row5923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_51 : scan edge (search edge 3) 4 [37] 37 innerTail9_51 best = best := by
  rw [innerTail9_51, scan, if_neg (by decide), innerCandidates9_51_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_52
    (if 3 ≤ innerCandidates9_51.length then search edge 3 [37, 5923] innerCandidates9_51 5960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_52

def innerTail9_50 : List Nat := 5857 :: innerTail9_51
def innerCandidates9_50 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_50_checked : candidates edge 4 37 5857 innerTail9_51 best.bound = innerCandidates9_50 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_50 : scan edge (search edge 3) 4 [37] 37 innerTail9_50 best = best := by
  rw [innerTail9_50, scan, if_neg (by decide), innerCandidates9_50_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_51
    (if 3 ≤ innerCandidates9_50.length then search edge 3 [37, 5857] innerCandidates9_50 5894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_51

def innerTail9_49 : List Nat := 5779 :: innerTail9_50
def innerCandidates9_49 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_49_checked : candidates edge 4 37 5779 innerTail9_50 best.bound = innerCandidates9_49 := by
  unfold candidates
  have row : edge 5779 = SearchData.row5779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_49 : scan edge (search edge 3) 4 [37] 37 innerTail9_49 best = best := by
  rw [innerTail9_49, scan, if_neg (by decide), innerCandidates9_49_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_50
    (if 3 ≤ innerCandidates9_49.length then search edge 3 [37, 5779] innerCandidates9_49 5816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_50

def innerTail9_48 : List Nat := 5623 :: innerTail9_49
def innerCandidates9_48 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_48_checked : candidates edge 4 37 5623 innerTail9_49 best.bound = innerCandidates9_48 := by
  unfold candidates
  have row : edge 5623 = SearchData.row5623.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_48 : scan edge (search edge 3) 4 [37] 37 innerTail9_48 best = best := by
  rw [innerTail9_48, scan, if_neg (by decide), innerCandidates9_48_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_49
    (if 3 ≤ innerCandidates9_48.length then search edge 3 [37, 5623] innerCandidates9_48 5660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_49

def innerTail9_47 : List Nat := 5449 :: innerTail9_48
def innerCandidates9_47 : List Nat := [8089]
set_option maxRecDepth 4096 in
theorem innerCandidates9_47_checked : candidates edge 4 37 5449 innerTail9_48 best.bound = innerCandidates9_47 := by
  unfold candidates
  have row : edge 5449 = SearchData.row5449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_47 : scan edge (search edge 3) 4 [37] 37 innerTail9_47 best = best := by
  rw [innerTail9_47, scan, if_neg (by decide), innerCandidates9_47_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_48
    (if 3 ≤ innerCandidates9_47.length then search edge 3 [37, 5449] innerCandidates9_47 5486 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_48

def innerTail9_46 : List Nat := 5281 :: innerTail9_47
def innerCandidates9_46 : List Nat := [7873]
set_option maxRecDepth 4096 in
theorem innerCandidates9_46_checked : candidates edge 4 37 5281 innerTail9_47 best.bound = innerCandidates9_46 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_46 : scan edge (search edge 3) 4 [37] 37 innerTail9_46 best = best := by
  rw [innerTail9_46, scan, if_neg (by decide), innerCandidates9_46_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_47
    (if 3 ≤ innerCandidates9_46.length then search edge 3 [37, 5281] innerCandidates9_46 5318 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_47

def innerTail9_45 : List Nat := 5227 :: innerTail9_46
def innerCandidates9_45 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_45_checked : candidates edge 4 37 5227 innerTail9_46 best.bound = innerCandidates9_45 := by
  unfold candidates
  have row : edge 5227 = SearchData.row5227.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_45 : scan edge (search edge 3) 4 [37] 37 innerTail9_45 best = best := by
  rw [innerTail9_45, scan, if_neg (by decide), innerCandidates9_45_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_46
    (if 3 ≤ innerCandidates9_45.length then search edge 3 [37, 5227] innerCandidates9_45 5264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_46

def innerTail9_44 : List Nat := 5113 :: innerTail9_45
def innerCandidates9_44 : List Nat := [5449, 6823]
set_option maxRecDepth 4096 in
theorem innerCandidates9_44_checked : candidates edge 4 37 5113 innerTail9_45 best.bound = innerCandidates9_44 := by
  unfold candidates
  have row : edge 5113 = SearchData.row5113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_44 : scan edge (search edge 3) 4 [37] 37 innerTail9_44 best = best := by
  rw [innerTail9_44, scan, if_neg (by decide), innerCandidates9_44_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_45
    (if 3 ≤ innerCandidates9_44.length then search edge 3 [37, 5113] innerCandidates9_44 5150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_45

def innerTail9_43 : List Nat := 5101 :: innerTail9_44
def innerCandidates9_43 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_43_checked : candidates edge 4 37 5101 innerTail9_44 best.bound = innerCandidates9_43 := by
  unfold candidates
  have row : edge 5101 = SearchData.row5101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_43 : scan edge (search edge 3) 4 [37] 37 innerTail9_43 best = best := by
  rw [innerTail9_43, scan, if_neg (by decide), innerCandidates9_43_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_44
    (if 3 ≤ innerCandidates9_43.length then search edge 3 [37, 5101] innerCandidates9_43 5138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_44

def innerTail9_42 : List Nat := 4909 :: innerTail9_43
def innerCandidates9_42 : List Nat := [5857]
set_option maxRecDepth 4096 in
theorem innerCandidates9_42_checked : candidates edge 4 37 4909 innerTail9_43 best.bound = innerCandidates9_42 := by
  unfold candidates
  have row : edge 4909 = SearchData.row4909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_42 : scan edge (search edge 3) 4 [37] 37 innerTail9_42 best = best := by
  rw [innerTail9_42, scan, if_neg (by decide), innerCandidates9_42_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_43
    (if 3 ≤ innerCandidates9_42.length then search edge 3 [37, 4909] innerCandidates9_42 4946 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_43

def innerTail9_41 : List Nat := 4789 :: innerTail9_42
def innerCandidates9_41 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_41_checked : candidates edge 4 37 4789 innerTail9_42 best.bound = innerCandidates9_41 := by
  unfold candidates
  have row : edge 4789 = SearchData.row4789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_41 : scan edge (search edge 3) 4 [37] 37 innerTail9_41 best = best := by
  rw [innerTail9_41, scan, if_neg (by decide), innerCandidates9_41_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_42
    (if 3 ≤ innerCandidates9_41.length then search edge 3 [37, 4789] innerCandidates9_41 4826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_42

def innerTail9_40 : List Nat := 4729 :: innerTail9_41
def innerCandidates9_40 : List Nat := [4909, 7963]
set_option maxRecDepth 4096 in
theorem innerCandidates9_40_checked : candidates edge 4 37 4729 innerTail9_41 best.bound = innerCandidates9_40 := by
  unfold candidates
  have row : edge 4729 = SearchData.row4729.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_40 : scan edge (search edge 3) 4 [37] 37 innerTail9_40 best = best := by
  rw [innerTail9_40, scan, if_neg (by decide), innerCandidates9_40_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_41
    (if 3 ≤ innerCandidates9_40.length then search edge 3 [37, 4729] innerCandidates9_40 4766 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_41

def innerTail9_39 : List Nat := 4603 :: innerTail9_40
def innerCandidates9_39 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_39_checked : candidates edge 4 37 4603 innerTail9_40 best.bound = innerCandidates9_39 := by
  unfold candidates
  have row : edge 4603 = SearchData.row4603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_39 : scan edge (search edge 3) 4 [37] 37 innerTail9_39 best = best := by
  rw [innerTail9_39, scan, if_neg (by decide), innerCandidates9_39_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_40
    (if 3 ≤ innerCandidates9_39.length then search edge 3 [37, 4603] innerCandidates9_39 4640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_40

def innerTail9_38 : List Nat := 4177 :: innerTail9_39
def innerCandidates9_38 : List Nat := [7297, 7537, 8101]
set_option maxRecDepth 4096 in
theorem innerCandidates9_38_checked : candidates edge 4 37 4177 innerTail9_39 best.bound = innerCandidates9_38 := by
  unfold candidates
  have row : edge 4177 = SearchData.row4177.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild9_38 : search edge 3 [37, 4177] innerCandidates9_38 4214 best = best := by decide

theorem innerScan9_38 : scan edge (search edge 3) 4 [37] 37 innerTail9_38 best = best := by
  rw [innerTail9_38, scan, if_neg (by decide), innerCandidates9_38_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_39
    (if 3 ≤ innerCandidates9_38.length then search edge 3 [37, 4177] innerCandidates9_38 4214 best else best) = best
  rw [if_pos (by decide), innerChild9_38]
  exact innerScan9_39

def innerTail9_37 : List Nat := 4159 :: innerTail9_38
def innerCandidates9_37 : List Nat := [4789, 5779]
set_option maxRecDepth 4096 in
theorem innerCandidates9_37_checked : candidates edge 4 37 4159 innerTail9_38 best.bound = innerCandidates9_37 := by
  unfold candidates
  have row : edge 4159 = SearchData.row4159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_37 : scan edge (search edge 3) 4 [37] 37 innerTail9_37 best = best := by
  rw [innerTail9_37, scan, if_neg (by decide), innerCandidates9_37_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_38
    (if 3 ≤ innerCandidates9_37.length then search edge 3 [37, 4159] innerCandidates9_37 4196 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_38

def innerTail9_36 : List Nat := 4093 :: innerTail9_37
def innerCandidates9_36 : List Nat := [6823]
set_option maxRecDepth 4096 in
theorem innerCandidates9_36_checked : candidates edge 4 37 4093 innerTail9_37 best.bound = innerCandidates9_36 := by
  unfold candidates
  have row : edge 4093 = SearchData.row4093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_36 : scan edge (search edge 3) 4 [37] 37 innerTail9_36 best = best := by
  rw [innerTail9_36, scan, if_neg (by decide), innerCandidates9_36_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_37
    (if 3 ≤ innerCandidates9_36.length then search edge 3 [37, 4093] innerCandidates9_36 4130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_37

def innerTail9_35 : List Nat := 3613 :: innerTail9_36
def innerCandidates9_35 : List Nat := [5623]
set_option maxRecDepth 4096 in
theorem innerCandidates9_35_checked : candidates edge 4 37 3613 innerTail9_36 best.bound = innerCandidates9_35 := by
  unfold candidates
  have row : edge 3613 = SearchData.row3613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_35 : scan edge (search edge 3) 4 [37] 37 innerTail9_35 best = best := by
  rw [innerTail9_35, scan, if_neg (by decide), innerCandidates9_35_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_36
    (if 3 ≤ innerCandidates9_35.length then search edge 3 [37, 3613] innerCandidates9_35 3650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_36

def innerTail9_34 : List Nat := 3463 :: innerTail9_35
def innerCandidates9_34 : List Nat := [8011]
set_option maxRecDepth 4096 in
theorem innerCandidates9_34_checked : candidates edge 4 37 3463 innerTail9_35 best.bound = innerCandidates9_34 := by
  unfold candidates
  have row : edge 3463 = SearchData.row3463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_34 : scan edge (search edge 3) 4 [37] 37 innerTail9_34 best = best := by
  rw [innerTail9_34, scan, if_neg (by decide), innerCandidates9_34_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_35
    (if 3 ≤ innerCandidates9_34.length then search edge 3 [37, 3463] innerCandidates9_34 3500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_35

def innerTail9_33 : List Nat := 3187 :: innerTail9_34
def innerCandidates9_33 : List Nat := [4093, 4729, 5227, 7477, 8101]
set_option maxRecDepth 4096 in
theorem innerCandidates9_33_checked : candidates edge 4 37 3187 innerTail9_34 best.bound = innerCandidates9_33 := by
  unfold candidates
  have row : edge 3187 = SearchData.row3187.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild9_33 : search edge 3 [37, 3187] innerCandidates9_33 3224 best = best := by decide

theorem innerScan9_33 : scan edge (search edge 3) 4 [37] 37 innerTail9_33 best = best := by
  rw [innerTail9_33, scan, if_neg (by decide), innerCandidates9_33_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_34
    (if 3 ≤ innerCandidates9_33.length then search edge 3 [37, 3187] innerCandidates9_33 3224 best else best) = best
  rw [if_pos (by decide), innerChild9_33]
  exact innerScan9_34

def innerTail9_32 : List Nat := 3181 :: innerTail9_33
def innerCandidates9_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_32_checked : candidates edge 4 37 3181 innerTail9_33 best.bound = innerCandidates9_32 := by
  unfold candidates
  have row : edge 3181 = SearchData.row3181.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_32 : scan edge (search edge 3) 4 [37] 37 innerTail9_32 best = best := by
  rw [innerTail9_32, scan, if_neg (by decide), innerCandidates9_32_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_33
    (if 3 ≤ innerCandidates9_32.length then search edge 3 [37, 3181] innerCandidates9_32 3218 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_33

def innerTail9_31 : List Nat := 3049 :: innerTail9_32
def innerCandidates9_31 : List Nat := [7297, 8089]
set_option maxRecDepth 4096 in
theorem innerCandidates9_31_checked : candidates edge 4 37 3049 innerTail9_32 best.bound = innerCandidates9_31 := by
  unfold candidates
  have row : edge 3049 = SearchData.row3049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_31 : scan edge (search edge 3) 4 [37] 37 innerTail9_31 best = best := by
  rw [innerTail9_31, scan, if_neg (by decide), innerCandidates9_31_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_32
    (if 3 ≤ innerCandidates9_31.length then search edge 3 [37, 3049] innerCandidates9_31 3086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_32

def innerTail9_30 : List Nat := 2803 :: innerTail9_31
def innerCandidates9_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_30_checked : candidates edge 4 37 2803 innerTail9_31 best.bound = innerCandidates9_30 := by
  unfold candidates
  have row : edge 2803 = SearchData.row2803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_30 : scan edge (search edge 3) 4 [37] 37 innerTail9_30 best = best := by
  rw [innerTail9_30, scan, if_neg (by decide), innerCandidates9_30_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_31
    (if 3 ≤ innerCandidates9_30.length then search edge 3 [37, 2803] innerCandidates9_30 2840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_31

def innerTail9_29 : List Nat := 2707 :: innerTail9_30
def innerCandidates9_29 : List Nat := [3463, 4909, 5857]
set_option maxRecDepth 4096 in
theorem innerCandidates9_29_checked : candidates edge 4 37 2707 innerTail9_30 best.bound = innerCandidates9_29 := by
  unfold candidates
  have row : edge 2707 = SearchData.row2707.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild9_29 : search edge 3 [37, 2707] innerCandidates9_29 2744 best = best := by decide

theorem innerScan9_29 : scan edge (search edge 3) 4 [37] 37 innerTail9_29 best = best := by
  rw [innerTail9_29, scan, if_neg (by decide), innerCandidates9_29_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_30
    (if 3 ≤ innerCandidates9_29.length then search edge 3 [37, 2707] innerCandidates9_29 2744 best else best) = best
  rw [if_pos (by decide), innerChild9_29]
  exact innerScan9_30

def innerTail9_28 : List Nat := 2689 :: innerTail9_29
def innerCandidates9_28 : List Nat := [7477, 8089]
set_option maxRecDepth 4096 in
theorem innerCandidates9_28_checked : candidates edge 4 37 2689 innerTail9_29 best.bound = innerCandidates9_28 := by
  unfold candidates
  have row : edge 2689 = SearchData.row2689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_28 : scan edge (search edge 3) 4 [37] 37 innerTail9_28 best = best := by
  rw [innerTail9_28, scan, if_neg (by decide), innerCandidates9_28_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_29
    (if 3 ≤ innerCandidates9_28.length then search edge 3 [37, 2689] innerCandidates9_28 2726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_29

def innerTail9_27 : List Nat := 2677 :: innerTail9_28
def innerCandidates9_27 : List Nat := [3049, 7369]
set_option maxRecDepth 4096 in
theorem innerCandidates9_27_checked : candidates edge 4 37 2677 innerTail9_28 best.bound = innerCandidates9_27 := by
  unfold candidates
  have row : edge 2677 = SearchData.row2677.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_27 : scan edge (search edge 3) 4 [37] 37 innerTail9_27 best = best := by
  rw [innerTail9_27, scan, if_neg (by decide), innerCandidates9_27_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_28
    (if 3 ≤ innerCandidates9_27.length then search edge 3 [37, 2677] innerCandidates9_27 2714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_28

def innerTail9_26 : List Nat := 2539 :: innerTail9_27
def innerCandidates9_26 : List Nat := [3463, 5449, 6823]
set_option maxRecDepth 4096 in
theorem innerCandidates9_26_checked : candidates edge 4 37 2539 innerTail9_27 best.bound = innerCandidates9_26 := by
  unfold candidates
  have row : edge 2539 = PrimePairCache.smallRow2539.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild9_26 : search edge 3 [37, 2539] innerCandidates9_26 2576 best = best := by decide

theorem innerScan9_26 : scan edge (search edge 3) 4 [37] 37 innerTail9_26 best = best := by
  rw [innerTail9_26, scan, if_neg (by decide), innerCandidates9_26_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_27
    (if 3 ≤ innerCandidates9_26.length then search edge 3 [37, 2539] innerCandidates9_26 2576 best else best) = best
  rw [if_pos (by decide), innerChild9_26]
  exact innerScan9_27

def innerTail9_25 : List Nat := 2473 :: innerTail9_26
def innerCandidates9_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_25_checked : candidates edge 4 37 2473 innerTail9_26 best.bound = innerCandidates9_25 := by
  unfold candidates
  have row : edge 2473 = PrimePairCache.smallRow2473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_25 : scan edge (search edge 3) 4 [37] 37 innerTail9_25 best = best := by
  rw [innerTail9_25, scan, if_neg (by decide), innerCandidates9_25_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_26
    (if 3 ≤ innerCandidates9_25.length then search edge 3 [37, 2473] innerCandidates9_25 2510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_26

def innerTail9_24 : List Nat := 2377 :: innerTail9_25
def innerCandidates9_24 : List Nat := [4159]
set_option maxRecDepth 4096 in
theorem innerCandidates9_24_checked : candidates edge 4 37 2377 innerTail9_25 best.bound = innerCandidates9_24 := by
  unfold candidates
  have row : edge 2377 = PrimePairCache.smallRow2377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_24 : scan edge (search edge 3) 4 [37] 37 innerTail9_24 best = best := by
  rw [innerTail9_24, scan, if_neg (by decide), innerCandidates9_24_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_25
    (if 3 ≤ innerCandidates9_24.length then search edge 3 [37, 2377] innerCandidates9_24 2414 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_25

def innerTail9_23 : List Nat := 2371 :: innerTail9_24
def innerCandidates9_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates9_23_checked : candidates edge 4 37 2371 innerTail9_24 best.bound = innerCandidates9_23 := by
  unfold candidates
  have row : edge 2371 = PrimePairCache.smallRow2371.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_23 : scan edge (search edge 3) 4 [37] 37 innerTail9_23 best = best := by
  rw [innerTail9_23, scan, if_neg (by decide), innerCandidates9_23_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_24
    (if 3 ≤ innerCandidates9_23.length then search edge 3 [37, 2371] innerCandidates9_23 2408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_24

def innerTail9_22 : List Nat := 2269 :: innerTail9_23
def innerCandidates9_22 : List Nat := [3613, 4909, 5923]
set_option maxRecDepth 4096 in
theorem innerCandidates9_22_checked : candidates edge 4 37 2269 innerTail9_23 best.bound = innerCandidates9_22 := by
  unfold candidates
  have row : edge 2269 = PrimePairCache.smallRow2269.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild9_22 : search edge 3 [37, 2269] innerCandidates9_22 2306 best = best := by decide

theorem innerScan9_22 : scan edge (search edge 3) 4 [37] 37 innerTail9_22 best = best := by
  rw [innerTail9_22, scan, if_neg (by decide), innerCandidates9_22_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_23
    (if 3 ≤ innerCandidates9_22.length then search edge 3 [37, 2269] innerCandidates9_22 2306 best else best) = best
  rw [if_pos (by decide), innerChild9_22]
  exact innerScan9_23

def innerTail9_21 : List Nat := 2179 :: innerTail9_22
def innerCandidates9_21 : List Nat := [5227, 5857, 6529]
set_option maxRecDepth 4096 in
theorem innerCandidates9_21_checked : candidates edge 4 37 2179 innerTail9_22 best.bound = innerCandidates9_21 := by
  unfold candidates
  have row : edge 2179 = PrimePairCache.smallRow2179.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild9_21 : search edge 3 [37, 2179] innerCandidates9_21 2216 best = best := by decide

theorem innerScan9_21 : scan edge (search edge 3) 4 [37] 37 innerTail9_21 best = best := by
  rw [innerTail9_21, scan, if_neg (by decide), innerCandidates9_21_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_22
    (if 3 ≤ innerCandidates9_21.length then search edge 3 [37, 2179] innerCandidates9_21 2216 best else best) = best
  rw [if_pos (by decide), innerChild9_21]
  exact innerScan9_22

def innerTail9_20 : List Nat := 2137 :: innerTail9_21
def innerCandidates9_20 : List Nat := [2539, 6133]
set_option maxRecDepth 4096 in
theorem innerCandidates9_20_checked : candidates edge 4 37 2137 innerTail9_21 best.bound = innerCandidates9_20 := by
  unfold candidates
  have row : edge 2137 = PrimePairCache.smallRow2137.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_20 : scan edge (search edge 3) 4 [37] 37 innerTail9_20 best = best := by
  rw [innerTail9_20, scan, if_neg (by decide), innerCandidates9_20_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_21
    (if 3 ≤ innerCandidates9_20.length then search edge 3 [37, 2137] innerCandidates9_20 2174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_21

def innerTail9_19 : List Nat := 1951 :: innerTail9_20
def innerCandidates9_19 : List Nat := [4909, 5227, 6823]
set_option maxRecDepth 4096 in
theorem innerCandidates9_19_checked : candidates edge 4 37 1951 innerTail9_20 best.bound = innerCandidates9_19 := by
  unfold candidates
  have row : edge 1951 = PrimePairCache.smallRow1951.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild9_19 : search edge 3 [37, 1951] innerCandidates9_19 1988 best = best := by decide

theorem innerScan9_19 : scan edge (search edge 3) 4 [37] 37 innerTail9_19 best = best := by
  rw [innerTail9_19, scan, if_neg (by decide), innerCandidates9_19_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_20
    (if 3 ≤ innerCandidates9_19.length then search edge 3 [37, 1951] innerCandidates9_19 1988 best else best) = best
  rw [if_pos (by decide), innerChild9_19]
  exact innerScan9_20

def innerTail9_18 : List Nat := 1873 :: innerTail9_19
def innerCandidates9_18 : List Nat := [2269, 7963, 8011]
set_option maxRecDepth 4096 in
theorem innerCandidates9_18_checked : candidates edge 4 37 1873 innerTail9_19 best.bound = innerCandidates9_18 := by
  unfold candidates
  have row : edge 1873 = PrimePairCache.smallRow1873.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild9_18 : search edge 3 [37, 1873] innerCandidates9_18 1910 best = best := by decide

theorem innerScan9_18 : scan edge (search edge 3) 4 [37] 37 innerTail9_18 best = best := by
  rw [innerTail9_18, scan, if_neg (by decide), innerCandidates9_18_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_19
    (if 3 ≤ innerCandidates9_18.length then search edge 3 [37, 1873] innerCandidates9_18 1910 best else best) = best
  rw [if_pos (by decide), innerChild9_18]
  exact innerScan9_19

def innerTail9_17 : List Nat := 1699 :: innerTail9_18
def innerCandidates9_17 : List Nat := [2179, 4177]
set_option maxRecDepth 4096 in
theorem innerCandidates9_17_checked : candidates edge 4 37 1699 innerTail9_18 best.bound = innerCandidates9_17 := by
  unfold candidates
  have row : edge 1699 = PrimePairCache.smallRow1699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_17 : scan edge (search edge 3) 4 [37] 37 innerTail9_17 best = best := by
  rw [innerTail9_17, scan, if_neg (by decide), innerCandidates9_17_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_18
    (if 3 ≤ innerCandidates9_17.length then search edge 3 [37, 1699] innerCandidates9_17 1736 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_18

def innerTail9_16 : List Nat := 1549 :: innerTail9_17
def innerCandidates9_16 : List Nat := [2707, 3463, 4789, 7393]
set_option maxRecDepth 4096 in
theorem innerCandidates9_16_checked : candidates edge 4 37 1549 innerTail9_17 best.bound = innerCandidates9_16 := by
  unfold candidates
  have row : edge 1549 = PrimePairCache.smallRow1549.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild9_16 : search edge 3 [37, 1549] innerCandidates9_16 1586 best = best := by decide

theorem innerScan9_16 : scan edge (search edge 3) 4 [37] 37 innerTail9_16 best = best := by
  rw [innerTail9_16, scan, if_neg (by decide), innerCandidates9_16_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_17
    (if 3 ≤ innerCandidates9_16.length then search edge 3 [37, 1549] innerCandidates9_16 1586 best else best) = best
  rw [if_pos (by decide), innerChild9_16]
  exact innerScan9_17

def innerTail9_15 : List Nat := 1471 :: innerTail9_16
def innerCandidates9_15 : List Nat := [3613, 4159, 6133]
set_option maxRecDepth 4096 in
theorem innerCandidates9_15_checked : candidates edge 4 37 1471 innerTail9_16 best.bound = innerCandidates9_15 := by
  unfold candidates
  have row : edge 1471 = PrimePairCache.smallRow1471.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild9_15 : search edge 3 [37, 1471] innerCandidates9_15 1508 best = best := by decide

theorem innerScan9_15 : scan edge (search edge 3) 4 [37] 37 innerTail9_15 best = best := by
  rw [innerTail9_15, scan, if_neg (by decide), innerCandidates9_15_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_16
    (if 3 ≤ innerCandidates9_15.length then search edge 3 [37, 1471] innerCandidates9_15 1508 best else best) = best
  rw [if_pos (by decide), innerChild9_15]
  exact innerScan9_16

def innerTail9_14 : List Nat := 1447 :: innerTail9_15
def innerCandidates9_14 : List Nat := [1549, 2803, 4093]
set_option maxRecDepth 4096 in
theorem innerCandidates9_14_checked : candidates edge 4 37 1447 innerTail9_15 best.bound = innerCandidates9_14 := by
  unfold candidates
  have row : edge 1447 = PrimePairCache.smallRow1447.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild9_14 : search edge 3 [37, 1447] innerCandidates9_14 1484 best = best := by decide

theorem innerScan9_14 : scan edge (search edge 3) 4 [37] 37 innerTail9_14 best = best := by
  rw [innerTail9_14, scan, if_neg (by decide), innerCandidates9_14_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_15
    (if 3 ≤ innerCandidates9_14.length then search edge 3 [37, 1447] innerCandidates9_14 1484 best else best) = best
  rw [if_pos (by decide), innerChild9_14]
  exact innerScan9_15

def innerTail9_13 : List Nat := 1321 :: innerTail9_14
def innerCandidates9_13 : List Nat := [2803, 7537]
set_option maxRecDepth 4096 in
theorem innerCandidates9_13_checked : candidates edge 4 37 1321 innerTail9_14 best.bound = innerCandidates9_13 := by
  unfold candidates
  have row : edge 1321 = PrimePairCache.smallRow1321.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_13 : scan edge (search edge 3) 4 [37] 37 innerTail9_13 best = best := by
  rw [innerTail9_13, scan, if_neg (by decide), innerCandidates9_13_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_14
    (if 3 ≤ innerCandidates9_13.length then search edge 3 [37, 1321] innerCandidates9_13 1358 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_14

def innerTail9_12 : List Nat := 1303 :: innerTail9_13
def innerCandidates9_12 : List Nat := [8101]
set_option maxRecDepth 4096 in
theorem innerCandidates9_12_checked : candidates edge 4 37 1303 innerTail9_13 best.bound = innerCandidates9_12 := by
  unfold candidates
  have row : edge 1303 = PrimePairCache.smallRow1303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_12 : scan edge (search edge 3) 4 [37] 37 innerTail9_12 best = best := by
  rw [innerTail9_12, scan, if_neg (by decide), innerCandidates9_12_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_13
    (if 3 ≤ innerCandidates9_12.length then search edge 3 [37, 1303] innerCandidates9_12 1340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_13

def innerTail9_11 : List Nat := 1237 :: innerTail9_12
def innerCandidates9_11 : List Nat := [1549, 3187]
set_option maxRecDepth 4096 in
theorem innerCandidates9_11_checked : candidates edge 4 37 1237 innerTail9_12 best.bound = innerCandidates9_11 := by
  unfold candidates
  have row : edge 1237 = PrimePairCache.smallRow1237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_11 : scan edge (search edge 3) 4 [37] 37 innerTail9_11 best = best := by
  rw [innerTail9_11, scan, if_neg (by decide), innerCandidates9_11_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_12
    (if 3 ≤ innerCandidates9_11.length then search edge 3 [37, 1237] innerCandidates9_11 1274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_12

def innerTail9_10 : List Nat := 1153 :: innerTail9_11
def innerCandidates9_10 : List Nat := [1699, 2473, 2803, 4093]
set_option maxRecDepth 4096 in
theorem innerCandidates9_10_checked : candidates edge 4 37 1153 innerTail9_11 best.bound = innerCandidates9_10 := by
  unfold candidates
  have row : edge 1153 = PrimePairCache.smallRow1153.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild9_10 : search edge 3 [37, 1153] innerCandidates9_10 1190 best = best := by decide

theorem innerScan9_10 : scan edge (search edge 3) 4 [37] 37 innerTail9_10 best = best := by
  rw [innerTail9_10, scan, if_neg (by decide), innerCandidates9_10_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_11
    (if 3 ≤ innerCandidates9_10.length then search edge 3 [37, 1153] innerCandidates9_10 1190 best else best) = best
  rw [if_pos (by decide), innerChild9_10]
  exact innerScan9_11

def innerTail9_9 : List Nat := 1069 :: innerTail9_10
def innerCandidates9_9 : List Nat := [1321, 4909, 6529]
set_option maxRecDepth 4096 in
theorem innerCandidates9_9_checked : candidates edge 4 37 1069 innerTail9_10 best.bound = innerCandidates9_9 := by
  unfold candidates
  have row : edge 1069 = PrimePairCache.smallRow1069.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild9_9 : search edge 3 [37, 1069] innerCandidates9_9 1106 best = best := by decide

theorem innerScan9_9 : scan edge (search edge 3) 4 [37] 37 innerTail9_9 best = best := by
  rw [innerTail9_9, scan, if_neg (by decide), innerCandidates9_9_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_10
    (if 3 ≤ innerCandidates9_9.length then search edge 3 [37, 1069] innerCandidates9_9 1106 best else best) = best
  rw [if_pos (by decide), innerChild9_9]
  exact innerScan9_10

def innerTail9_8 : List Nat := 991 :: innerTail9_9
def innerCandidates9_8 : List Nat := [2269, 2689, 2803, 3613]
set_option maxRecDepth 4096 in
theorem innerCandidates9_8_checked : candidates edge 4 37 991 innerTail9_9 best.bound = innerCandidates9_8 := by
  unfold candidates
  have row : edge 991 = PrimePairCache.smallRow991.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild9_8 : search edge 3 [37, 991] innerCandidates9_8 1028 best = best := by decide

theorem innerScan9_8 : scan edge (search edge 3) 4 [37] 37 innerTail9_8 best = best := by
  rw [innerTail9_8, scan, if_neg (by decide), innerCandidates9_8_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_9
    (if 3 ≤ innerCandidates9_8.length then search edge 3 [37, 991] innerCandidates9_8 1028 best else best) = best
  rw [if_pos (by decide), innerChild9_8]
  exact innerScan9_9

def innerTail9_7 : List Nat := 967 :: innerTail9_8
def innerCandidates9_7 : List Nat := [1951, 3049, 5623, 5779, 5857]
set_option maxRecDepth 4096 in
theorem innerCandidates9_7_checked : candidates edge 4 37 967 innerTail9_8 best.bound = innerCandidates9_7 := by
  unfold candidates
  have row : edge 967 = PrimePairCache.smallRow967.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild9_7 : search edge 3 [37, 967] innerCandidates9_7 1004 best = best := by decide

theorem innerScan9_7 : scan edge (search edge 3) 4 [37] 37 innerTail9_7 best = best := by
  rw [innerTail9_7, scan, if_neg (by decide), innerCandidates9_7_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_8
    (if 3 ≤ innerCandidates9_7.length then search edge 3 [37, 967] innerCandidates9_7 1004 best else best) = best
  rw [if_pos (by decide), innerChild9_7]
  exact innerScan9_8

def innerTail9_6 : List Nat := 607 :: innerTail9_7
def innerCandidates9_6 : List Nat := [967, 991, 4603]
set_option maxRecDepth 4096 in
theorem innerCandidates9_6_checked : candidates edge 4 37 607 innerTail9_7 best.bound = innerCandidates9_6 := by
  unfold candidates
  have row : edge 607 = PrimePairCache.smallRow607.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild9_6 : search edge 3 [37, 607] innerCandidates9_6 644 best = best := by decide

theorem innerScan9_6 : scan edge (search edge 3) 4 [37] 37 innerTail9_6 best = best := by
  rw [innerTail9_6, scan, if_neg (by decide), innerCandidates9_6_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_7
    (if 3 ≤ innerCandidates9_6.length then search edge 3 [37, 607] innerCandidates9_6 644 best else best) = best
  rw [if_pos (by decide), innerChild9_6]
  exact innerScan9_7

def innerTail9_5 : List Nat := 463 :: innerTail9_6
def innerCandidates9_5 : List Nat := [2473, 5623]
set_option maxRecDepth 4096 in
theorem innerCandidates9_5_checked : candidates edge 4 37 463 innerTail9_6 best.bound = innerCandidates9_5 := by
  unfold candidates
  have row : edge 463 = PrimePairCache.smallRow463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_5 : scan edge (search edge 3) 4 [37] 37 innerTail9_5 best = best := by
  rw [innerTail9_5, scan, if_neg (by decide), innerCandidates9_5_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_6
    (if 3 ≤ innerCandidates9_5.length then search edge 3 [37, 463] innerCandidates9_5 500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_6

def innerTail9_4 : List Nat := 313 :: innerTail9_5
def innerCandidates9_4 : List Nat := [991]
set_option maxRecDepth 4096 in
theorem innerCandidates9_4_checked : candidates edge 4 37 313 innerTail9_5 best.bound = innerCandidates9_4 := by
  unfold candidates
  have row : edge 313 = PrimePairCache.smallRow313.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_4 : scan edge (search edge 3) 4 [37] 37 innerTail9_4 best = best := by
  rw [innerTail9_4, scan, if_neg (by decide), innerCandidates9_4_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_5
    (if 3 ≤ innerCandidates9_4.length then search edge 3 [37, 313] innerCandidates9_4 350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_5

def innerTail9_3 : List Nat := 277 :: innerTail9_4
def innerCandidates9_3 : List Nat := [1471]
set_option maxRecDepth 4096 in
theorem innerCandidates9_3_checked : candidates edge 4 37 277 innerTail9_4 best.bound = innerCandidates9_3 := by
  unfold candidates
  have row : edge 277 = PrimePairCache.smallRow277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_3 : scan edge (search edge 3) 4 [37] 37 innerTail9_3 best = best := by
  rw [innerTail9_3, scan, if_neg (by decide), innerCandidates9_3_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_4
    (if 3 ≤ innerCandidates9_3.length then search edge 3 [37, 277] innerCandidates9_3 314 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_4

def innerTail9_2 : List Nat := 199 :: innerTail9_3
def innerCandidates9_2 : List Nat := [1153]
set_option maxRecDepth 4096 in
theorem innerCandidates9_2_checked : candidates edge 4 37 199 innerTail9_3 best.bound = innerCandidates9_2 := by
  unfold candidates
  have row : edge 199 = PrimePairCache.smallRow199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan9_2 : scan edge (search edge 3) 4 [37] 37 innerTail9_2 best = best := by
  rw [innerTail9_2, scan, if_neg (by decide), innerCandidates9_2_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_3
    (if 3 ≤ innerCandidates9_2.length then search edge 3 [37, 199] innerCandidates9_2 236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan9_3

def innerTail9_1 : List Nat := 79 :: innerTail9_2
def innerCandidates9_1 : List Nat := [967, 1447, 2179, 2371, 2689, 3181, 5101, 5449, 6373]
set_option maxRecDepth 4096 in
theorem innerCandidates9_1_checked : candidates edge 4 37 79 innerTail9_2 best.bound = innerCandidates9_1 := by
  unfold candidates
  have row : edge 79 = PrimePairCache.smallRow79.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild9_1 : search edge 3 [37, 79] innerCandidates9_1 116 best = best := by decide

theorem innerScan9_1 : scan edge (search edge 3) 4 [37] 37 innerTail9_1 best = best := by
  rw [innerTail9_1, scan, if_neg (by decide), innerCandidates9_1_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_2
    (if 3 ≤ innerCandidates9_1.length then search edge 3 [37, 79] innerCandidates9_1 116 best else best) = best
  rw [if_pos (by decide), innerChild9_1]
  exact innerScan9_2

def innerTail9_0 : List Nat := 67 :: innerTail9_1
def innerCandidates9_0 : List Nat := [1303, 2377, 4789, 5923]
set_option maxRecDepth 4096 in
theorem innerCandidates9_0_checked : candidates edge 4 37 67 innerTail9_1 best.bound = innerCandidates9_0 := by
  unfold candidates
  have row : edge 67 = PrimePairCache.smallRow67.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild9_0 : search edge 3 [37, 67] innerCandidates9_0 104 best = best := by decide

theorem innerScan9_0 : scan edge (search edge 3) 4 [37] 37 innerTail9_0 best = best := by
  rw [innerTail9_0, scan, if_neg (by decide), innerCandidates9_0_checked]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_1
    (if 3 ≤ innerCandidates9_0.length then search edge 3 [37, 67] innerCandidates9_0 104 best else best) = best
  rw [if_pos (by decide), innerChild9_0]
  exact innerScan9_1

theorem rootBranch9 : search edge 4 [37] rootCandidates9 37 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [37] 37 innerTail9_0 best = best
  exact innerScan9_0

end ProjectEulerSolutions.PrimePairUpper.Search
