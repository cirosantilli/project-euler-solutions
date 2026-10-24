import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates13 : List Nat := [113, 197, 269, 353, 401, 419, 653, 1229, 1481, 1637, 1823, 1847, 1997, 2027, 2069, 2099, 2153, 2267, 2333, 2447, 2531, 2621, 2663, 3011, 3167, 3257, 3371, 3581, 3593, 4007, 4019, 4091, 4283, 4637, 4649, 4799, 4943, 5273, 5303, 5333, 5783, 5861, 6287, 6311, 6323, 6353, 6779, 6791, 6803, 6857, 7079, 7307, 7703, 7793, 7841, 7853, 7883, 8117, 8123]
set_option maxRecDepth 4096 in
theorem rootCandidates13_checked : candidates edge 5 0 53 tail14 best.bound = rootCandidates13 := by
  unfold candidates
  have row : edge 53 = PrimePairCache.smallRow53.contains := by funext q; rfl
  rw [row]
  decide

def innerTail13_59 : List Nat := []
theorem innerScan13_59 : scan edge (search edge 3) 4 [53] 53 innerTail13_59 best = best := rfl

def innerTail13_58 : List Nat := 8123 :: innerTail13_59
def innerCandidates13_58 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_58_checked : candidates edge 4 53 8123 innerTail13_59 best.bound = innerCandidates13_58 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_58 : scan edge (search edge 3) 4 [53] 53 innerTail13_58 best = best := by
  rw [innerTail13_58, scan, if_neg (by decide), innerCandidates13_58_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_59
    (if 3 ≤ innerCandidates13_58.length then search edge 3 [53, 8123] innerCandidates13_58 8176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_59

def innerTail13_57 : List Nat := 8117 :: innerTail13_58
def innerCandidates13_57 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_57_checked : candidates edge 4 53 8117 innerTail13_58 best.bound = innerCandidates13_57 := by
  unfold candidates
  have row : edge 8117 = SearchData.row8117.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_57 : scan edge (search edge 3) 4 [53] 53 innerTail13_57 best = best := by
  rw [innerTail13_57, scan, if_neg (by decide), innerCandidates13_57_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_58
    (if 3 ≤ innerCandidates13_57.length then search edge 3 [53, 8117] innerCandidates13_57 8170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_58

def innerTail13_56 : List Nat := 7883 :: innerTail13_57
def innerCandidates13_56 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_56_checked : candidates edge 4 53 7883 innerTail13_57 best.bound = innerCandidates13_56 := by
  unfold candidates
  have row : edge 7883 = SearchData.row7883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_56 : scan edge (search edge 3) 4 [53] 53 innerTail13_56 best = best := by
  rw [innerTail13_56, scan, if_neg (by decide), innerCandidates13_56_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_57
    (if 3 ≤ innerCandidates13_56.length then search edge 3 [53, 7883] innerCandidates13_56 7936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_57

def innerTail13_55 : List Nat := 7853 :: innerTail13_56
def innerCandidates13_55 : List Nat := [7883]
set_option maxRecDepth 4096 in
theorem innerCandidates13_55_checked : candidates edge 4 53 7853 innerTail13_56 best.bound = innerCandidates13_55 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_55 : scan edge (search edge 3) 4 [53] 53 innerTail13_55 best = best := by
  rw [innerTail13_55, scan, if_neg (by decide), innerCandidates13_55_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_56
    (if 3 ≤ innerCandidates13_55.length then search edge 3 [53, 7853] innerCandidates13_55 7906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_56

def innerTail13_54 : List Nat := 7841 :: innerTail13_55
def innerCandidates13_54 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_54_checked : candidates edge 4 53 7841 innerTail13_55 best.bound = innerCandidates13_54 := by
  unfold candidates
  have row : edge 7841 = SearchData.row7841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_54 : scan edge (search edge 3) 4 [53] 53 innerTail13_54 best = best := by
  rw [innerTail13_54, scan, if_neg (by decide), innerCandidates13_54_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_55
    (if 3 ≤ innerCandidates13_54.length then search edge 3 [53, 7841] innerCandidates13_54 7894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_55

def innerTail13_53 : List Nat := 7793 :: innerTail13_54
def innerCandidates13_53 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_53_checked : candidates edge 4 53 7793 innerTail13_54 best.bound = innerCandidates13_53 := by
  unfold candidates
  have row : edge 7793 = SearchData.row7793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_53 : scan edge (search edge 3) 4 [53] 53 innerTail13_53 best = best := by
  rw [innerTail13_53, scan, if_neg (by decide), innerCandidates13_53_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_54
    (if 3 ≤ innerCandidates13_53.length then search edge 3 [53, 7793] innerCandidates13_53 7846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_54

def innerTail13_52 : List Nat := 7703 :: innerTail13_53
def innerCandidates13_52 : List Nat := [7853]
set_option maxRecDepth 4096 in
theorem innerCandidates13_52_checked : candidates edge 4 53 7703 innerTail13_53 best.bound = innerCandidates13_52 := by
  unfold candidates
  have row : edge 7703 = SearchData.row7703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_52 : scan edge (search edge 3) 4 [53] 53 innerTail13_52 best = best := by
  rw [innerTail13_52, scan, if_neg (by decide), innerCandidates13_52_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_53
    (if 3 ≤ innerCandidates13_52.length then search edge 3 [53, 7703] innerCandidates13_52 7756 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_53

def innerTail13_51 : List Nat := 7307 :: innerTail13_52
def innerCandidates13_51 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_51_checked : candidates edge 4 53 7307 innerTail13_52 best.bound = innerCandidates13_51 := by
  unfold candidates
  have row : edge 7307 = SearchData.row7307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_51 : scan edge (search edge 3) 4 [53] 53 innerTail13_51 best = best := by
  rw [innerTail13_51, scan, if_neg (by decide), innerCandidates13_51_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_52
    (if 3 ≤ innerCandidates13_51.length then search edge 3 [53, 7307] innerCandidates13_51 7360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_52

def innerTail13_50 : List Nat := 7079 :: innerTail13_51
def innerCandidates13_50 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_50_checked : candidates edge 4 53 7079 innerTail13_51 best.bound = innerCandidates13_50 := by
  unfold candidates
  have row : edge 7079 = SearchData.row7079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_50 : scan edge (search edge 3) 4 [53] 53 innerTail13_50 best = best := by
  rw [innerTail13_50, scan, if_neg (by decide), innerCandidates13_50_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_51
    (if 3 ≤ innerCandidates13_50.length then search edge 3 [53, 7079] innerCandidates13_50 7132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_51

def innerTail13_49 : List Nat := 6857 :: innerTail13_50
def innerCandidates13_49 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_49_checked : candidates edge 4 53 6857 innerTail13_50 best.bound = innerCandidates13_49 := by
  unfold candidates
  have row : edge 6857 = SearchData.row6857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_49 : scan edge (search edge 3) 4 [53] 53 innerTail13_49 best = best := by
  rw [innerTail13_49, scan, if_neg (by decide), innerCandidates13_49_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_50
    (if 3 ≤ innerCandidates13_49.length then search edge 3 [53, 6857] innerCandidates13_49 6910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_50

def innerTail13_48 : List Nat := 6803 :: innerTail13_49
def innerCandidates13_48 : List Nat := [8123]
set_option maxRecDepth 4096 in
theorem innerCandidates13_48_checked : candidates edge 4 53 6803 innerTail13_49 best.bound = innerCandidates13_48 := by
  unfold candidates
  have row : edge 6803 = SearchData.row6803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_48 : scan edge (search edge 3) 4 [53] 53 innerTail13_48 best = best := by
  rw [innerTail13_48, scan, if_neg (by decide), innerCandidates13_48_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_49
    (if 3 ≤ innerCandidates13_48.length then search edge 3 [53, 6803] innerCandidates13_48 6856 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_49

def innerTail13_47 : List Nat := 6791 :: innerTail13_48
def innerCandidates13_47 : List Nat := [7841]
set_option maxRecDepth 4096 in
theorem innerCandidates13_47_checked : candidates edge 4 53 6791 innerTail13_48 best.bound = innerCandidates13_47 := by
  unfold candidates
  have row : edge 6791 = SearchData.row6791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_47 : scan edge (search edge 3) 4 [53] 53 innerTail13_47 best = best := by
  rw [innerTail13_47, scan, if_neg (by decide), innerCandidates13_47_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_48
    (if 3 ≤ innerCandidates13_47.length then search edge 3 [53, 6791] innerCandidates13_47 6844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_48

def innerTail13_46 : List Nat := 6779 :: innerTail13_47
def innerCandidates13_46 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_46_checked : candidates edge 4 53 6779 innerTail13_47 best.bound = innerCandidates13_46 := by
  unfold candidates
  have row : edge 6779 = SearchData.row6779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_46 : scan edge (search edge 3) 4 [53] 53 innerTail13_46 best = best := by
  rw [innerTail13_46, scan, if_neg (by decide), innerCandidates13_46_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_47
    (if 3 ≤ innerCandidates13_46.length then search edge 3 [53, 6779] innerCandidates13_46 6832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_47

def innerTail13_45 : List Nat := 6353 :: innerTail13_46
def innerCandidates13_45 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_45_checked : candidates edge 4 53 6353 innerTail13_46 best.bound = innerCandidates13_45 := by
  unfold candidates
  have row : edge 6353 = SearchData.row6353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_45 : scan edge (search edge 3) 4 [53] 53 innerTail13_45 best = best := by
  rw [innerTail13_45, scan, if_neg (by decide), innerCandidates13_45_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_46
    (if 3 ≤ innerCandidates13_45.length then search edge 3 [53, 6353] innerCandidates13_45 6406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_46

def innerTail13_44 : List Nat := 6323 :: innerTail13_45
def innerCandidates13_44 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_44_checked : candidates edge 4 53 6323 innerTail13_45 best.bound = innerCandidates13_44 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_44 : scan edge (search edge 3) 4 [53] 53 innerTail13_44 best = best := by
  rw [innerTail13_44, scan, if_neg (by decide), innerCandidates13_44_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_45
    (if 3 ≤ innerCandidates13_44.length then search edge 3 [53, 6323] innerCandidates13_44 6376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_45

def innerTail13_43 : List Nat := 6311 :: innerTail13_44
def innerCandidates13_43 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_43_checked : candidates edge 4 53 6311 innerTail13_44 best.bound = innerCandidates13_43 := by
  unfold candidates
  have row : edge 6311 = SearchData.row6311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_43 : scan edge (search edge 3) 4 [53] 53 innerTail13_43 best = best := by
  rw [innerTail13_43, scan, if_neg (by decide), innerCandidates13_43_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_44
    (if 3 ≤ innerCandidates13_43.length then search edge 3 [53, 6311] innerCandidates13_43 6364 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_44

def innerTail13_42 : List Nat := 6287 :: innerTail13_43
def innerCandidates13_42 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_42_checked : candidates edge 4 53 6287 innerTail13_43 best.bound = innerCandidates13_42 := by
  unfold candidates
  have row : edge 6287 = SearchData.row6287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_42 : scan edge (search edge 3) 4 [53] 53 innerTail13_42 best = best := by
  rw [innerTail13_42, scan, if_neg (by decide), innerCandidates13_42_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_43
    (if 3 ≤ innerCandidates13_42.length then search edge 3 [53, 6287] innerCandidates13_42 6340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_43

def innerTail13_41 : List Nat := 5861 :: innerTail13_42
def innerCandidates13_41 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_41_checked : candidates edge 4 53 5861 innerTail13_42 best.bound = innerCandidates13_41 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_41 : scan edge (search edge 3) 4 [53] 53 innerTail13_41 best = best := by
  rw [innerTail13_41, scan, if_neg (by decide), innerCandidates13_41_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_42
    (if 3 ≤ innerCandidates13_41.length then search edge 3 [53, 5861] innerCandidates13_41 5914 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_42

def innerTail13_40 : List Nat := 5783 :: innerTail13_41
def innerCandidates13_40 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_40_checked : candidates edge 4 53 5783 innerTail13_41 best.bound = innerCandidates13_40 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_40 : scan edge (search edge 3) 4 [53] 53 innerTail13_40 best = best := by
  rw [innerTail13_40, scan, if_neg (by decide), innerCandidates13_40_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_41
    (if 3 ≤ innerCandidates13_40.length then search edge 3 [53, 5783] innerCandidates13_40 5836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_41

def innerTail13_39 : List Nat := 5333 :: innerTail13_40
def innerCandidates13_39 : List Nat := [7703]
set_option maxRecDepth 4096 in
theorem innerCandidates13_39_checked : candidates edge 4 53 5333 innerTail13_40 best.bound = innerCandidates13_39 := by
  unfold candidates
  have row : edge 5333 = SearchData.row5333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_39 : scan edge (search edge 3) 4 [53] 53 innerTail13_39 best = best := by
  rw [innerTail13_39, scan, if_neg (by decide), innerCandidates13_39_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_40
    (if 3 ≤ innerCandidates13_39.length then search edge 3 [53, 5333] innerCandidates13_39 5386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_40

def innerTail13_38 : List Nat := 5303 :: innerTail13_39
def innerCandidates13_38 : List Nat := [8123]
set_option maxRecDepth 4096 in
theorem innerCandidates13_38_checked : candidates edge 4 53 5303 innerTail13_39 best.bound = innerCandidates13_38 := by
  unfold candidates
  have row : edge 5303 = SearchData.row5303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_38 : scan edge (search edge 3) 4 [53] 53 innerTail13_38 best = best := by
  rw [innerTail13_38, scan, if_neg (by decide), innerCandidates13_38_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_39
    (if 3 ≤ innerCandidates13_38.length then search edge 3 [53, 5303] innerCandidates13_38 5356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_39

def innerTail13_37 : List Nat := 5273 :: innerTail13_38
def innerCandidates13_37 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_37_checked : candidates edge 4 53 5273 innerTail13_38 best.bound = innerCandidates13_37 := by
  unfold candidates
  have row : edge 5273 = SearchData.row5273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_37 : scan edge (search edge 3) 4 [53] 53 innerTail13_37 best = best := by
  rw [innerTail13_37, scan, if_neg (by decide), innerCandidates13_37_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_38
    (if 3 ≤ innerCandidates13_37.length then search edge 3 [53, 5273] innerCandidates13_37 5326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_38

def innerTail13_36 : List Nat := 4943 :: innerTail13_37
def innerCandidates13_36 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_36_checked : candidates edge 4 53 4943 innerTail13_37 best.bound = innerCandidates13_36 := by
  unfold candidates
  have row : edge 4943 = SearchData.row4943.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_36 : scan edge (search edge 3) 4 [53] 53 innerTail13_36 best = best := by
  rw [innerTail13_36, scan, if_neg (by decide), innerCandidates13_36_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_37
    (if 3 ≤ innerCandidates13_36.length then search edge 3 [53, 4943] innerCandidates13_36 4996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_37

def innerTail13_35 : List Nat := 4799 :: innerTail13_36
def innerCandidates13_35 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_35_checked : candidates edge 4 53 4799 innerTail13_36 best.bound = innerCandidates13_35 := by
  unfold candidates
  have row : edge 4799 = SearchData.row4799.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_35 : scan edge (search edge 3) 4 [53] 53 innerTail13_35 best = best := by
  rw [innerTail13_35, scan, if_neg (by decide), innerCandidates13_35_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_36
    (if 3 ≤ innerCandidates13_35.length then search edge 3 [53, 4799] innerCandidates13_35 4852 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_36

def innerTail13_34 : List Nat := 4649 :: innerTail13_35
def innerCandidates13_34 : List Nat := [6323, 7853, 8117]
set_option maxRecDepth 4096 in
theorem innerCandidates13_34_checked : candidates edge 4 53 4649 innerTail13_35 best.bound = innerCandidates13_34 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild13_34 : search edge 3 [53, 4649] innerCandidates13_34 4702 best = best := by decide

theorem innerScan13_34 : scan edge (search edge 3) 4 [53] 53 innerTail13_34 best = best := by
  rw [innerTail13_34, scan, if_neg (by decide), innerCandidates13_34_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_35
    (if 3 ≤ innerCandidates13_34.length then search edge 3 [53, 4649] innerCandidates13_34 4702 best else best) = best
  rw [if_pos (by decide), innerChild13_34]
  exact innerScan13_35

def innerTail13_33 : List Nat := 4637 :: innerTail13_34
def innerCandidates13_33 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_33_checked : candidates edge 4 53 4637 innerTail13_34 best.bound = innerCandidates13_33 := by
  unfold candidates
  have row : edge 4637 = SearchData.row4637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_33 : scan edge (search edge 3) 4 [53] 53 innerTail13_33 best = best := by
  rw [innerTail13_33, scan, if_neg (by decide), innerCandidates13_33_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_34
    (if 3 ≤ innerCandidates13_33.length then search edge 3 [53, 4637] innerCandidates13_33 4690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_34

def innerTail13_32 : List Nat := 4283 :: innerTail13_33
def innerCandidates13_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_32_checked : candidates edge 4 53 4283 innerTail13_33 best.bound = innerCandidates13_32 := by
  unfold candidates
  have row : edge 4283 = SearchData.row4283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_32 : scan edge (search edge 3) 4 [53] 53 innerTail13_32 best = best := by
  rw [innerTail13_32, scan, if_neg (by decide), innerCandidates13_32_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_33
    (if 3 ≤ innerCandidates13_32.length then search edge 3 [53, 4283] innerCandidates13_32 4336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_33

def innerTail13_31 : List Nat := 4091 :: innerTail13_32
def innerCandidates13_31 : List Nat := [4283]
set_option maxRecDepth 4096 in
theorem innerCandidates13_31_checked : candidates edge 4 53 4091 innerTail13_32 best.bound = innerCandidates13_31 := by
  unfold candidates
  have row : edge 4091 = SearchData.row4091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_31 : scan edge (search edge 3) 4 [53] 53 innerTail13_31 best = best := by
  rw [innerTail13_31, scan, if_neg (by decide), innerCandidates13_31_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_32
    (if 3 ≤ innerCandidates13_31.length then search edge 3 [53, 4091] innerCandidates13_31 4144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_32

def innerTail13_30 : List Nat := 4019 :: innerTail13_31
def innerCandidates13_30 : List Nat := [6791]
set_option maxRecDepth 4096 in
theorem innerCandidates13_30_checked : candidates edge 4 53 4019 innerTail13_31 best.bound = innerCandidates13_30 := by
  unfold candidates
  have row : edge 4019 = SearchData.row4019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_30 : scan edge (search edge 3) 4 [53] 53 innerTail13_30 best = best := by
  rw [innerTail13_30, scan, if_neg (by decide), innerCandidates13_30_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_31
    (if 3 ≤ innerCandidates13_30.length then search edge 3 [53, 4019] innerCandidates13_30 4072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_31

def innerTail13_29 : List Nat := 4007 :: innerTail13_30
def innerCandidates13_29 : List Nat := [4091, 6353, 7853]
set_option maxRecDepth 4096 in
theorem innerCandidates13_29_checked : candidates edge 4 53 4007 innerTail13_30 best.bound = innerCandidates13_29 := by
  unfold candidates
  have row : edge 4007 = SearchData.row4007.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild13_29 : search edge 3 [53, 4007] innerCandidates13_29 4060 best = best := by decide

theorem innerScan13_29 : scan edge (search edge 3) 4 [53] 53 innerTail13_29 best = best := by
  rw [innerTail13_29, scan, if_neg (by decide), innerCandidates13_29_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_30
    (if 3 ≤ innerCandidates13_29.length then search edge 3 [53, 4007] innerCandidates13_29 4060 best else best) = best
  rw [if_pos (by decide), innerChild13_29]
  exact innerScan13_30

def innerTail13_28 : List Nat := 3593 :: innerTail13_29
def innerCandidates13_28 : List Nat := [6311]
set_option maxRecDepth 4096 in
theorem innerCandidates13_28_checked : candidates edge 4 53 3593 innerTail13_29 best.bound = innerCandidates13_28 := by
  unfold candidates
  have row : edge 3593 = SearchData.row3593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_28 : scan edge (search edge 3) 4 [53] 53 innerTail13_28 best = best := by
  rw [innerTail13_28, scan, if_neg (by decide), innerCandidates13_28_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_29
    (if 3 ≤ innerCandidates13_28.length then search edge 3 [53, 3593] innerCandidates13_28 3646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_29

def innerTail13_27 : List Nat := 3581 :: innerTail13_28
def innerCandidates13_27 : List Nat := [4091, 5303]
set_option maxRecDepth 4096 in
theorem innerCandidates13_27_checked : candidates edge 4 53 3581 innerTail13_28 best.bound = innerCandidates13_27 := by
  unfold candidates
  have row : edge 3581 = SearchData.row3581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_27 : scan edge (search edge 3) 4 [53] 53 innerTail13_27 best = best := by
  rw [innerTail13_27, scan, if_neg (by decide), innerCandidates13_27_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_28
    (if 3 ≤ innerCandidates13_27.length then search edge 3 [53, 3581] innerCandidates13_27 3634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_28

def innerTail13_26 : List Nat := 3371 :: innerTail13_27
def innerCandidates13_26 : List Nat := [4649, 4799]
set_option maxRecDepth 4096 in
theorem innerCandidates13_26_checked : candidates edge 4 53 3371 innerTail13_27 best.bound = innerCandidates13_26 := by
  unfold candidates
  have row : edge 3371 = SearchData.row3371.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_26 : scan edge (search edge 3) 4 [53] 53 innerTail13_26 best = best := by
  rw [innerTail13_26, scan, if_neg (by decide), innerCandidates13_26_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_27
    (if 3 ≤ innerCandidates13_26.length then search edge 3 [53, 3371] innerCandidates13_26 3424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_27

def innerTail13_25 : List Nat := 3257 :: innerTail13_26
def innerCandidates13_25 : List Nat := [4649]
set_option maxRecDepth 4096 in
theorem innerCandidates13_25_checked : candidates edge 4 53 3257 innerTail13_26 best.bound = innerCandidates13_25 := by
  unfold candidates
  have row : edge 3257 = SearchData.row3257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_25 : scan edge (search edge 3) 4 [53] 53 innerTail13_25 best = best := by
  rw [innerTail13_25, scan, if_neg (by decide), innerCandidates13_25_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_26
    (if 3 ≤ innerCandidates13_25.length then search edge 3 [53, 3257] innerCandidates13_25 3310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_26

def innerTail13_24 : List Nat := 3167 :: innerTail13_25
def innerCandidates13_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_24_checked : candidates edge 4 53 3167 innerTail13_25 best.bound = innerCandidates13_24 := by
  unfold candidates
  have row : edge 3167 = SearchData.row3167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_24 : scan edge (search edge 3) 4 [53] 53 innerTail13_24 best = best := by
  rw [innerTail13_24, scan, if_neg (by decide), innerCandidates13_24_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_25
    (if 3 ≤ innerCandidates13_24.length then search edge 3 [53, 3167] innerCandidates13_24 3220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_25

def innerTail13_23 : List Nat := 3011 :: innerTail13_24
def innerCandidates13_23 : List Nat := [6353, 7883]
set_option maxRecDepth 4096 in
theorem innerCandidates13_23_checked : candidates edge 4 53 3011 innerTail13_24 best.bound = innerCandidates13_23 := by
  unfold candidates
  have row : edge 3011 = SearchData.row3011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_23 : scan edge (search edge 3) 4 [53] 53 innerTail13_23 best = best := by
  rw [innerTail13_23, scan, if_neg (by decide), innerCandidates13_23_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_24
    (if 3 ≤ innerCandidates13_23.length then search edge 3 [53, 3011] innerCandidates13_23 3064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_24

def innerTail13_22 : List Nat := 2663 :: innerTail13_23
def innerCandidates13_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_22_checked : candidates edge 4 53 2663 innerTail13_23 best.bound = innerCandidates13_22 := by
  unfold candidates
  have row : edge 2663 = PrimePairCache.smallRow2663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_22 : scan edge (search edge 3) 4 [53] 53 innerTail13_22 best = best := by
  rw [innerTail13_22, scan, if_neg (by decide), innerCandidates13_22_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_23
    (if 3 ≤ innerCandidates13_22.length then search edge 3 [53, 2663] innerCandidates13_22 2716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_23

def innerTail13_21 : List Nat := 2621 :: innerTail13_22
def innerCandidates13_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_21_checked : candidates edge 4 53 2621 innerTail13_22 best.bound = innerCandidates13_21 := by
  unfold candidates
  have row : edge 2621 = PrimePairCache.smallRow2621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_21 : scan edge (search edge 3) 4 [53] 53 innerTail13_21 best = best := by
  rw [innerTail13_21, scan, if_neg (by decide), innerCandidates13_21_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_22
    (if 3 ≤ innerCandidates13_21.length then search edge 3 [53, 2621] innerCandidates13_21 2674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_22

def innerTail13_20 : List Nat := 2531 :: innerTail13_21
def innerCandidates13_20 : List Nat := [5273, 7883, 8123]
set_option maxRecDepth 4096 in
theorem innerCandidates13_20_checked : candidates edge 4 53 2531 innerTail13_21 best.bound = innerCandidates13_20 := by
  unfold candidates
  have row : edge 2531 = PrimePairCache.smallRow2531.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild13_20 : search edge 3 [53, 2531] innerCandidates13_20 2584 best = best := by decide

theorem innerScan13_20 : scan edge (search edge 3) 4 [53] 53 innerTail13_20 best = best := by
  rw [innerTail13_20, scan, if_neg (by decide), innerCandidates13_20_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_21
    (if 3 ≤ innerCandidates13_20.length then search edge 3 [53, 2531] innerCandidates13_20 2584 best else best) = best
  rw [if_pos (by decide), innerChild13_20]
  exact innerScan13_21

def innerTail13_19 : List Nat := 2447 :: innerTail13_20
def innerCandidates13_19 : List Nat := [3011, 4019, 6323, 8123]
set_option maxRecDepth 4096 in
theorem innerCandidates13_19_checked : candidates edge 4 53 2447 innerTail13_20 best.bound = innerCandidates13_19 := by
  unfold candidates
  have row : edge 2447 = PrimePairCache.smallRow2447.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild13_19 : search edge 3 [53, 2447] innerCandidates13_19 2500 best = best := by decide

theorem innerScan13_19 : scan edge (search edge 3) 4 [53] 53 innerTail13_19 best = best := by
  rw [innerTail13_19, scan, if_neg (by decide), innerCandidates13_19_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_20
    (if 3 ≤ innerCandidates13_19.length then search edge 3 [53, 2447] innerCandidates13_19 2500 best else best) = best
  rw [if_pos (by decide), innerChild13_19]
  exact innerScan13_20

def innerTail13_18 : List Nat := 2333 :: innerTail13_19
def innerCandidates13_18 : List Nat := [6311]
set_option maxRecDepth 4096 in
theorem innerCandidates13_18_checked : candidates edge 4 53 2333 innerTail13_19 best.bound = innerCandidates13_18 := by
  unfold candidates
  have row : edge 2333 = PrimePairCache.smallRow2333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_18 : scan edge (search edge 3) 4 [53] 53 innerTail13_18 best = best := by
  rw [innerTail13_18, scan, if_neg (by decide), innerCandidates13_18_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_19
    (if 3 ≤ innerCandidates13_18.length then search edge 3 [53, 2333] innerCandidates13_18 2386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_19

def innerTail13_17 : List Nat := 2267 :: innerTail13_18
def innerCandidates13_17 : List Nat := [4637, 6791]
set_option maxRecDepth 4096 in
theorem innerCandidates13_17_checked : candidates edge 4 53 2267 innerTail13_18 best.bound = innerCandidates13_17 := by
  unfold candidates
  have row : edge 2267 = PrimePairCache.smallRow2267.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_17 : scan edge (search edge 3) 4 [53] 53 innerTail13_17 best = best := by
  rw [innerTail13_17, scan, if_neg (by decide), innerCandidates13_17_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_18
    (if 3 ≤ innerCandidates13_17.length then search edge 3 [53, 2267] innerCandidates13_17 2320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_18

def innerTail13_16 : List Nat := 2153 :: innerTail13_17
def innerCandidates13_16 : List Nat := [2663, 5273]
set_option maxRecDepth 4096 in
theorem innerCandidates13_16_checked : candidates edge 4 53 2153 innerTail13_17 best.bound = innerCandidates13_16 := by
  unfold candidates
  have row : edge 2153 = PrimePairCache.smallRow2153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_16 : scan edge (search edge 3) 4 [53] 53 innerTail13_16 best = best := by
  rw [innerTail13_16, scan, if_neg (by decide), innerCandidates13_16_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_17
    (if 3 ≤ innerCandidates13_16.length then search edge 3 [53, 2153] innerCandidates13_16 2206 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_17

def innerTail13_15 : List Nat := 2099 :: innerTail13_16
def innerCandidates13_15 : List Nat := [7841]
set_option maxRecDepth 4096 in
theorem innerCandidates13_15_checked : candidates edge 4 53 2099 innerTail13_16 best.bound = innerCandidates13_15 := by
  unfold candidates
  have row : edge 2099 = PrimePairCache.smallRow2099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_15 : scan edge (search edge 3) 4 [53] 53 innerTail13_15 best = best := by
  rw [innerTail13_15, scan, if_neg (by decide), innerCandidates13_15_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_16
    (if 3 ≤ innerCandidates13_15.length then search edge 3 [53, 2099] innerCandidates13_15 2152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_16

def innerTail13_14 : List Nat := 2069 :: innerTail13_15
def innerCandidates13_14 : List Nat := [4637]
set_option maxRecDepth 4096 in
theorem innerCandidates13_14_checked : candidates edge 4 53 2069 innerTail13_15 best.bound = innerCandidates13_14 := by
  unfold candidates
  have row : edge 2069 = PrimePairCache.smallRow2069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_14 : scan edge (search edge 3) 4 [53] 53 innerTail13_14 best = best := by
  rw [innerTail13_14, scan, if_neg (by decide), innerCandidates13_14_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_15
    (if 3 ≤ innerCandidates13_14.length then search edge 3 [53, 2069] innerCandidates13_14 2122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_15

def innerTail13_13 : List Nat := 2027 :: innerTail13_14
def innerCandidates13_13 : List Nat := [7703]
set_option maxRecDepth 4096 in
theorem innerCandidates13_13_checked : candidates edge 4 53 2027 innerTail13_14 best.bound = innerCandidates13_13 := by
  unfold candidates
  have row : edge 2027 = PrimePairCache.smallRow2027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_13 : scan edge (search edge 3) 4 [53] 53 innerTail13_13 best = best := by
  rw [innerTail13_13, scan, if_neg (by decide), innerCandidates13_13_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_14
    (if 3 ≤ innerCandidates13_13.length then search edge 3 [53, 2027] innerCandidates13_13 2080 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_14

def innerTail13_12 : List Nat := 1997 :: innerTail13_13
def innerCandidates13_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_12_checked : candidates edge 4 53 1997 innerTail13_13 best.bound = innerCandidates13_12 := by
  unfold candidates
  have row : edge 1997 = PrimePairCache.smallRow1997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_12 : scan edge (search edge 3) 4 [53] 53 innerTail13_12 best = best := by
  rw [innerTail13_12, scan, if_neg (by decide), innerCandidates13_12_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_13
    (if 3 ≤ innerCandidates13_12.length then search edge 3 [53, 1997] innerCandidates13_12 2050 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_13

def innerTail13_11 : List Nat := 1847 :: innerTail13_12
def innerCandidates13_11 : List Nat := [3011, 4007, 4283, 7307, 7703, 8123]
set_option maxRecDepth 4096 in
theorem innerCandidates13_11_checked : candidates edge 4 53 1847 innerTail13_12 best.bound = innerCandidates13_11 := by
  unfold candidates
  have row : edge 1847 = PrimePairCache.smallRow1847.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild13_11 : search edge 3 [53, 1847] innerCandidates13_11 1900 best = best := by decide

theorem innerScan13_11 : scan edge (search edge 3) 4 [53] 53 innerTail13_11 best = best := by
  rw [innerTail13_11, scan, if_neg (by decide), innerCandidates13_11_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_12
    (if 3 ≤ innerCandidates13_11.length then search edge 3 [53, 1847] innerCandidates13_11 1900 best else best) = best
  rw [if_pos (by decide), innerChild13_11]
  exact innerScan13_12

def innerTail13_10 : List Nat := 1823 :: innerTail13_11
def innerCandidates13_10 : List Nat := [2069, 6353]
set_option maxRecDepth 4096 in
theorem innerCandidates13_10_checked : candidates edge 4 53 1823 innerTail13_11 best.bound = innerCandidates13_10 := by
  unfold candidates
  have row : edge 1823 = PrimePairCache.smallRow1823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_10 : scan edge (search edge 3) 4 [53] 53 innerTail13_10 best = best := by
  rw [innerTail13_10, scan, if_neg (by decide), innerCandidates13_10_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_11
    (if 3 ≤ innerCandidates13_10.length then search edge 3 [53, 1823] innerCandidates13_10 1876 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_11

def innerTail13_9 : List Nat := 1637 :: innerTail13_10
def innerCandidates13_9 : List Nat := [4649, 7841]
set_option maxRecDepth 4096 in
theorem innerCandidates13_9_checked : candidates edge 4 53 1637 innerTail13_10 best.bound = innerCandidates13_9 := by
  unfold candidates
  have row : edge 1637 = PrimePairCache.smallRow1637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_9 : scan edge (search edge 3) 4 [53] 53 innerTail13_9 best = best := by
  rw [innerTail13_9, scan, if_neg (by decide), innerCandidates13_9_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_10
    (if 3 ≤ innerCandidates13_9.length then search edge 3 [53, 1637] innerCandidates13_9 1690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_10

def innerTail13_8 : List Nat := 1481 :: innerTail13_9
def innerCandidates13_8 : List Nat := [2663, 3167, 7703]
set_option maxRecDepth 4096 in
theorem innerCandidates13_8_checked : candidates edge 4 53 1481 innerTail13_9 best.bound = innerCandidates13_8 := by
  unfold candidates
  have row : edge 1481 = PrimePairCache.smallRow1481.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild13_8 : search edge 3 [53, 1481] innerCandidates13_8 1534 best = best := by decide

theorem innerScan13_8 : scan edge (search edge 3) 4 [53] 53 innerTail13_8 best = best := by
  rw [innerTail13_8, scan, if_neg (by decide), innerCandidates13_8_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_9
    (if 3 ≤ innerCandidates13_8.length then search edge 3 [53, 1481] innerCandidates13_8 1534 best else best) = best
  rw [if_pos (by decide), innerChild13_8]
  exact innerScan13_9

def innerTail13_7 : List Nat := 1229 :: innerTail13_8
def innerCandidates13_7 : List Nat := [2663, 5861, 6353]
set_option maxRecDepth 4096 in
theorem innerCandidates13_7_checked : candidates edge 4 53 1229 innerTail13_8 best.bound = innerCandidates13_7 := by
  unfold candidates
  have row : edge 1229 = PrimePairCache.smallRow1229.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild13_7 : search edge 3 [53, 1229] innerCandidates13_7 1282 best = best := by decide

theorem innerScan13_7 : scan edge (search edge 3) 4 [53] 53 innerTail13_7 best = best := by
  rw [innerTail13_7, scan, if_neg (by decide), innerCandidates13_7_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_8
    (if 3 ≤ innerCandidates13_7.length then search edge 3 [53, 1229] innerCandidates13_7 1282 best else best) = best
  rw [if_pos (by decide), innerChild13_7]
  exact innerScan13_8

def innerTail13_6 : List Nat := 653 :: innerTail13_7
def innerCandidates13_6 : List Nat := [2531, 3167]
set_option maxRecDepth 4096 in
theorem innerCandidates13_6_checked : candidates edge 4 53 653 innerTail13_7 best.bound = innerCandidates13_6 := by
  unfold candidates
  have row : edge 653 = PrimePairCache.smallRow653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_6 : scan edge (search edge 3) 4 [53] 53 innerTail13_6 best = best := by
  rw [innerTail13_6, scan, if_neg (by decide), innerCandidates13_6_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_7
    (if 3 ≤ innerCandidates13_6.length then search edge 3 [53, 653] innerCandidates13_6 706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_7

def innerTail13_5 : List Nat := 419 :: innerTail13_6
def innerCandidates13_5 : List Nat := [5273, 7883]
set_option maxRecDepth 4096 in
theorem innerCandidates13_5_checked : candidates edge 4 53 419 innerTail13_6 best.bound = innerCandidates13_5 := by
  unfold candidates
  have row : edge 419 = PrimePairCache.smallRow419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_5 : scan edge (search edge 3) 4 [53] 53 innerTail13_5 best = best := by
  rw [innerTail13_5, scan, if_neg (by decide), innerCandidates13_5_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_6
    (if 3 ≤ innerCandidates13_5.length then search edge 3 [53, 419] innerCandidates13_5 472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_6

def innerTail13_4 : List Nat := 401 :: innerTail13_5
def innerCandidates13_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_4_checked : candidates edge 4 53 401 innerTail13_5 best.bound = innerCandidates13_4 := by
  unfold candidates
  have row : edge 401 = PrimePairCache.smallRow401.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_4 : scan edge (search edge 3) 4 [53] 53 innerTail13_4 best = best := by
  rw [innerTail13_4, scan, if_neg (by decide), innerCandidates13_4_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_5
    (if 3 ≤ innerCandidates13_4.length then search edge 3 [53, 401] innerCandidates13_4 454 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_5

def innerTail13_3 : List Nat := 353 :: innerTail13_4
def innerCandidates13_3 : List Nat := [1481, 3371, 6791]
set_option maxRecDepth 4096 in
theorem innerCandidates13_3_checked : candidates edge 4 53 353 innerTail13_4 best.bound = innerCandidates13_3 := by
  unfold candidates
  have row : edge 353 = PrimePairCache.smallRow353.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild13_3 : search edge 3 [53, 353] innerCandidates13_3 406 best = best := by decide

theorem innerScan13_3 : scan edge (search edge 3) 4 [53] 53 innerTail13_3 best = best := by
  rw [innerTail13_3, scan, if_neg (by decide), innerCandidates13_3_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_4
    (if 3 ≤ innerCandidates13_3.length then search edge 3 [53, 353] innerCandidates13_3 406 best else best) = best
  rw [if_pos (by decide), innerChild13_3]
  exact innerScan13_4

def innerTail13_2 : List Nat := 269 :: innerTail13_3
def innerCandidates13_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates13_2_checked : candidates edge 4 53 269 innerTail13_3 best.bound = innerCandidates13_2 := by
  unfold candidates
  have row : edge 269 = PrimePairCache.smallRow269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_2 : scan edge (search edge 3) 4 [53] 53 innerTail13_2 best = best := by
  rw [innerTail13_2, scan, if_neg (by decide), innerCandidates13_2_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_3
    (if 3 ≤ innerCandidates13_2.length then search edge 3 [53, 269] innerCandidates13_2 322 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_3

def innerTail13_1 : List Nat := 197 :: innerTail13_2
def innerCandidates13_1 : List Nat := [1481, 1637, 2099]
set_option maxRecDepth 4096 in
theorem innerCandidates13_1_checked : candidates edge 4 53 197 innerTail13_2 best.bound = innerCandidates13_1 := by
  unfold candidates
  have row : edge 197 = PrimePairCache.smallRow197.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild13_1 : search edge 3 [53, 197] innerCandidates13_1 250 best = best := by decide

theorem innerScan13_1 : scan edge (search edge 3) 4 [53] 53 innerTail13_1 best = best := by
  rw [innerTail13_1, scan, if_neg (by decide), innerCandidates13_1_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_2
    (if 3 ≤ innerCandidates13_1.length then search edge 3 [53, 197] innerCandidates13_1 250 best else best) = best
  rw [if_pos (by decide), innerChild13_1]
  exact innerScan13_2

def innerTail13_0 : List Nat := 113 :: innerTail13_1
def innerCandidates13_0 : List Nat := [5333]
set_option maxRecDepth 4096 in
theorem innerCandidates13_0_checked : candidates edge 4 53 113 innerTail13_1 best.bound = innerCandidates13_0 := by
  unfold candidates
  have row : edge 113 = PrimePairCache.smallRow113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan13_0 : scan edge (search edge 3) 4 [53] 53 innerTail13_0 best = best := by
  rw [innerTail13_0, scan, if_neg (by decide), innerCandidates13_0_checked]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_1
    (if 3 ≤ innerCandidates13_0.length then search edge 3 [53, 113] innerCandidates13_0 166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan13_1

theorem rootBranch13 : search edge 4 [53] rootCandidates13 53 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [53] 53 innerTail13_0 best = best
  exact innerScan13_0

end ProjectEulerSolutions.PrimePairUpper.Search
