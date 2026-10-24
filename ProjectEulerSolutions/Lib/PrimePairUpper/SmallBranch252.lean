import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates15 : List Nat := [151, 331, 409, 487, 751, 1069, 1453, 1483, 1693, 1753, 1801, 1873, 1951, 1999, 2341, 2371, 2383, 2797, 2971, 3049, 3169, 3181, 3463, 3469, 3637, 3733, 3889, 4657, 5413, 5431, 5449, 5557, 5623, 5821, 5827, 5869, 6079, 6367, 6673, 6793, 6871, 6997, 7129, 7237, 7369, 7681, 7699, 7873, 7951, 7963, 8161, 8311]
set_option maxRecDepth 4096 in
theorem rootCandidates15_checked : candidates edge 5 0 61 tail16 best.bound = rootCandidates15 := by
  unfold candidates
  have row : edge 61 = PrimePairCache.smallRow61.contains := by funext q; rfl
  rw [row]
  decide

def innerTail15_52 : List Nat := []
theorem innerScan15_52 : scan edge (search edge 3) 4 [61] 61 innerTail15_52 best = best := rfl

def innerTail15_51 : List Nat := 8311 :: innerTail15_52
def innerCandidates15_51 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_51_checked : candidates edge 4 61 8311 innerTail15_52 best.bound = innerCandidates15_51 := by
  unfold candidates
  have row : edge 8311 = SearchData.row8311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_51 : scan edge (search edge 3) 4 [61] 61 innerTail15_51 best = best := by
  rw [innerTail15_51, scan, if_neg (by decide), innerCandidates15_51_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_52
    (if 3 ≤ innerCandidates15_51.length then search edge 3 [61, 8311] innerCandidates15_51 8372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_52

def innerTail15_50 : List Nat := 8161 :: innerTail15_51
def innerCandidates15_50 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_50_checked : candidates edge 4 61 8161 innerTail15_51 best.bound = innerCandidates15_50 := by
  unfold candidates
  have row : edge 8161 = SearchData.row8161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_50 : scan edge (search edge 3) 4 [61] 61 innerTail15_50 best = best := by
  rw [innerTail15_50, scan, if_neg (by decide), innerCandidates15_50_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_51
    (if 3 ≤ innerCandidates15_50.length then search edge 3 [61, 8161] innerCandidates15_50 8222 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_51

def innerTail15_49 : List Nat := 7963 :: innerTail15_50
def innerCandidates15_49 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_49_checked : candidates edge 4 61 7963 innerTail15_50 best.bound = innerCandidates15_49 := by
  unfold candidates
  have row : edge 7963 = SearchData.row7963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_49 : scan edge (search edge 3) 4 [61] 61 innerTail15_49 best = best := by
  rw [innerTail15_49, scan, if_neg (by decide), innerCandidates15_49_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_50
    (if 3 ≤ innerCandidates15_49.length then search edge 3 [61, 7963] innerCandidates15_49 8024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_50

def innerTail15_48 : List Nat := 7951 :: innerTail15_49
def innerCandidates15_48 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_48_checked : candidates edge 4 61 7951 innerTail15_49 best.bound = innerCandidates15_48 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_48 : scan edge (search edge 3) 4 [61] 61 innerTail15_48 best = best := by
  rw [innerTail15_48, scan, if_neg (by decide), innerCandidates15_48_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_49
    (if 3 ≤ innerCandidates15_48.length then search edge 3 [61, 7951] innerCandidates15_48 8012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_49

def innerTail15_47 : List Nat := 7873 :: innerTail15_48
def innerCandidates15_47 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_47_checked : candidates edge 4 61 7873 innerTail15_48 best.bound = innerCandidates15_47 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_47 : scan edge (search edge 3) 4 [61] 61 innerTail15_47 best = best := by
  rw [innerTail15_47, scan, if_neg (by decide), innerCandidates15_47_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_48
    (if 3 ≤ innerCandidates15_47.length then search edge 3 [61, 7873] innerCandidates15_47 7934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_48

def innerTail15_46 : List Nat := 7699 :: innerTail15_47
def innerCandidates15_46 : List Nat := [7873]
set_option maxRecDepth 4096 in
theorem innerCandidates15_46_checked : candidates edge 4 61 7699 innerTail15_47 best.bound = innerCandidates15_46 := by
  unfold candidates
  have row : edge 7699 = SearchData.row7699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_46 : scan edge (search edge 3) 4 [61] 61 innerTail15_46 best = best := by
  rw [innerTail15_46, scan, if_neg (by decide), innerCandidates15_46_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_47
    (if 3 ≤ innerCandidates15_46.length then search edge 3 [61, 7699] innerCandidates15_46 7760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_47

def innerTail15_45 : List Nat := 7681 :: innerTail15_46
def innerCandidates15_45 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_45_checked : candidates edge 4 61 7681 innerTail15_46 best.bound = innerCandidates15_45 := by
  unfold candidates
  have row : edge 7681 = SearchData.row7681.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_45 : scan edge (search edge 3) 4 [61] 61 innerTail15_45 best = best := by
  rw [innerTail15_45, scan, if_neg (by decide), innerCandidates15_45_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_46
    (if 3 ≤ innerCandidates15_45.length then search edge 3 [61, 7681] innerCandidates15_45 7742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_46

def innerTail15_44 : List Nat := 7369 :: innerTail15_45
def innerCandidates15_44 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_44_checked : candidates edge 4 61 7369 innerTail15_45 best.bound = innerCandidates15_44 := by
  unfold candidates
  have row : edge 7369 = SearchData.row7369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_44 : scan edge (search edge 3) 4 [61] 61 innerTail15_44 best = best := by
  rw [innerTail15_44, scan, if_neg (by decide), innerCandidates15_44_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_45
    (if 3 ≤ innerCandidates15_44.length then search edge 3 [61, 7369] innerCandidates15_44 7430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_45

def innerTail15_43 : List Nat := 7237 :: innerTail15_44
def innerCandidates15_43 : List Nat := [7699]
set_option maxRecDepth 4096 in
theorem innerCandidates15_43_checked : candidates edge 4 61 7237 innerTail15_44 best.bound = innerCandidates15_43 := by
  unfold candidates
  have row : edge 7237 = SearchData.row7237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_43 : scan edge (search edge 3) 4 [61] 61 innerTail15_43 best = best := by
  rw [innerTail15_43, scan, if_neg (by decide), innerCandidates15_43_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_44
    (if 3 ≤ innerCandidates15_43.length then search edge 3 [61, 7237] innerCandidates15_43 7298 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_44

def innerTail15_42 : List Nat := 7129 :: innerTail15_43
def innerCandidates15_42 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_42_checked : candidates edge 4 61 7129 innerTail15_43 best.bound = innerCandidates15_42 := by
  unfold candidates
  have row : edge 7129 = SearchData.row7129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_42 : scan edge (search edge 3) 4 [61] 61 innerTail15_42 best = best := by
  rw [innerTail15_42, scan, if_neg (by decide), innerCandidates15_42_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_43
    (if 3 ≤ innerCandidates15_42.length then search edge 3 [61, 7129] innerCandidates15_42 7190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_43

def innerTail15_41 : List Nat := 6997 :: innerTail15_42
def innerCandidates15_41 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_41_checked : candidates edge 4 61 6997 innerTail15_42 best.bound = innerCandidates15_41 := by
  unfold candidates
  have row : edge 6997 = SearchData.row6997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_41 : scan edge (search edge 3) 4 [61] 61 innerTail15_41 best = best := by
  rw [innerTail15_41, scan, if_neg (by decide), innerCandidates15_41_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_42
    (if 3 ≤ innerCandidates15_41.length then search edge 3 [61, 6997] innerCandidates15_41 7058 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_42

def innerTail15_40 : List Nat := 6871 :: innerTail15_41
def innerCandidates15_40 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_40_checked : candidates edge 4 61 6871 innerTail15_41 best.bound = innerCandidates15_40 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_40 : scan edge (search edge 3) 4 [61] 61 innerTail15_40 best = best := by
  rw [innerTail15_40, scan, if_neg (by decide), innerCandidates15_40_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_41
    (if 3 ≤ innerCandidates15_40.length then search edge 3 [61, 6871] innerCandidates15_40 6932 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_41

def innerTail15_39 : List Nat := 6793 :: innerTail15_40
def innerCandidates15_39 : List Nat := [7951]
set_option maxRecDepth 4096 in
theorem innerCandidates15_39_checked : candidates edge 4 61 6793 innerTail15_40 best.bound = innerCandidates15_39 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_39 : scan edge (search edge 3) 4 [61] 61 innerTail15_39 best = best := by
  rw [innerTail15_39, scan, if_neg (by decide), innerCandidates15_39_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_40
    (if 3 ≤ innerCandidates15_39.length then search edge 3 [61, 6793] innerCandidates15_39 6854 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_40

def innerTail15_38 : List Nat := 6673 :: innerTail15_39
def innerCandidates15_38 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_38_checked : candidates edge 4 61 6673 innerTail15_39 best.bound = innerCandidates15_38 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_38 : scan edge (search edge 3) 4 [61] 61 innerTail15_38 best = best := by
  rw [innerTail15_38, scan, if_neg (by decide), innerCandidates15_38_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_39
    (if 3 ≤ innerCandidates15_38.length then search edge 3 [61, 6673] innerCandidates15_38 6734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_39

def innerTail15_37 : List Nat := 6367 :: innerTail15_38
def innerCandidates15_37 : List Nat := [6793]
set_option maxRecDepth 4096 in
theorem innerCandidates15_37_checked : candidates edge 4 61 6367 innerTail15_38 best.bound = innerCandidates15_37 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_37 : scan edge (search edge 3) 4 [61] 61 innerTail15_37 best = best := by
  rw [innerTail15_37, scan, if_neg (by decide), innerCandidates15_37_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_38
    (if 3 ≤ innerCandidates15_37.length then search edge 3 [61, 6367] innerCandidates15_37 6428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_38

def innerTail15_36 : List Nat := 6079 :: innerTail15_37
def innerCandidates15_36 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_36_checked : candidates edge 4 61 6079 innerTail15_37 best.bound = innerCandidates15_36 := by
  unfold candidates
  have row : edge 6079 = SearchData.row6079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_36 : scan edge (search edge 3) 4 [61] 61 innerTail15_36 best = best := by
  rw [innerTail15_36, scan, if_neg (by decide), innerCandidates15_36_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_37
    (if 3 ≤ innerCandidates15_36.length then search edge 3 [61, 6079] innerCandidates15_36 6140 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_37

def innerTail15_35 : List Nat := 5869 :: innerTail15_36
def innerCandidates15_35 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_35_checked : candidates edge 4 61 5869 innerTail15_36 best.bound = innerCandidates15_35 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_35 : scan edge (search edge 3) 4 [61] 61 innerTail15_35 best = best := by
  rw [innerTail15_35, scan, if_neg (by decide), innerCandidates15_35_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_36
    (if 3 ≤ innerCandidates15_35.length then search edge 3 [61, 5869] innerCandidates15_35 5930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_36

def innerTail15_34 : List Nat := 5827 :: innerTail15_35
def innerCandidates15_34 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_34_checked : candidates edge 4 61 5827 innerTail15_35 best.bound = innerCandidates15_34 := by
  unfold candidates
  have row : edge 5827 = SearchData.row5827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_34 : scan edge (search edge 3) 4 [61] 61 innerTail15_34 best = best := by
  rw [innerTail15_34, scan, if_neg (by decide), innerCandidates15_34_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_35
    (if 3 ≤ innerCandidates15_34.length then search edge 3 [61, 5827] innerCandidates15_34 5888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_35

def innerTail15_33 : List Nat := 5821 :: innerTail15_34
def innerCandidates15_33 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_33_checked : candidates edge 4 61 5821 innerTail15_34 best.bound = innerCandidates15_33 := by
  unfold candidates
  have row : edge 5821 = SearchData.row5821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_33 : scan edge (search edge 3) 4 [61] 61 innerTail15_33 best = best := by
  rw [innerTail15_33, scan, if_neg (by decide), innerCandidates15_33_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_34
    (if 3 ≤ innerCandidates15_33.length then search edge 3 [61, 5821] innerCandidates15_33 5882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_34

def innerTail15_32 : List Nat := 5623 :: innerTail15_33
def innerCandidates15_32 : List Nat := [5821, 7681, 8311]
set_option maxRecDepth 4096 in
theorem innerCandidates15_32_checked : candidates edge 4 61 5623 innerTail15_33 best.bound = innerCandidates15_32 := by
  unfold candidates
  have row : edge 5623 = SearchData.row5623.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild15_32 : search edge 3 [61, 5623] innerCandidates15_32 5684 best = best := by decide

theorem innerScan15_32 : scan edge (search edge 3) 4 [61] 61 innerTail15_32 best = best := by
  rw [innerTail15_32, scan, if_neg (by decide), innerCandidates15_32_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_33
    (if 3 ≤ innerCandidates15_32.length then search edge 3 [61, 5623] innerCandidates15_32 5684 best else best) = best
  rw [if_pos (by decide), innerChild15_32]
  exact innerScan15_33

def innerTail15_31 : List Nat := 5557 :: innerTail15_32
def innerCandidates15_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_31_checked : candidates edge 4 61 5557 innerTail15_32 best.bound = innerCandidates15_31 := by
  unfold candidates
  have row : edge 5557 = SearchData.row5557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_31 : scan edge (search edge 3) 4 [61] 61 innerTail15_31 best = best := by
  rw [innerTail15_31, scan, if_neg (by decide), innerCandidates15_31_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_32
    (if 3 ≤ innerCandidates15_31.length then search edge 3 [61, 5557] innerCandidates15_31 5618 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_32

def innerTail15_30 : List Nat := 5449 :: innerTail15_31
def innerCandidates15_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_30_checked : candidates edge 4 61 5449 innerTail15_31 best.bound = innerCandidates15_30 := by
  unfold candidates
  have row : edge 5449 = SearchData.row5449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_30 : scan edge (search edge 3) 4 [61] 61 innerTail15_30 best = best := by
  rw [innerTail15_30, scan, if_neg (by decide), innerCandidates15_30_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_31
    (if 3 ≤ innerCandidates15_30.length then search edge 3 [61, 5449] innerCandidates15_30 5510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_31

def innerTail15_29 : List Nat := 5431 :: innerTail15_30
def innerCandidates15_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_29_checked : candidates edge 4 61 5431 innerTail15_30 best.bound = innerCandidates15_29 := by
  unfold candidates
  have row : edge 5431 = SearchData.row5431.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_29 : scan edge (search edge 3) 4 [61] 61 innerTail15_29 best = best := by
  rw [innerTail15_29, scan, if_neg (by decide), innerCandidates15_29_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_30
    (if 3 ≤ innerCandidates15_29.length then search edge 3 [61, 5431] innerCandidates15_29 5492 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_30

def innerTail15_28 : List Nat := 5413 :: innerTail15_29
def innerCandidates15_28 : List Nat := [5821]
set_option maxRecDepth 4096 in
theorem innerCandidates15_28_checked : candidates edge 4 61 5413 innerTail15_29 best.bound = innerCandidates15_28 := by
  unfold candidates
  have row : edge 5413 = SearchData.row5413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_28 : scan edge (search edge 3) 4 [61] 61 innerTail15_28 best = best := by
  rw [innerTail15_28, scan, if_neg (by decide), innerCandidates15_28_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_29
    (if 3 ≤ innerCandidates15_28.length then search edge 3 [61, 5413] innerCandidates15_28 5474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_29

def innerTail15_27 : List Nat := 4657 :: innerTail15_28
def innerCandidates15_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_27_checked : candidates edge 4 61 4657 innerTail15_28 best.bound = innerCandidates15_27 := by
  unfold candidates
  have row : edge 4657 = SearchData.row4657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_27 : scan edge (search edge 3) 4 [61] 61 innerTail15_27 best = best := by
  rw [innerTail15_27, scan, if_neg (by decide), innerCandidates15_27_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_28
    (if 3 ≤ innerCandidates15_27.length then search edge 3 [61, 4657] innerCandidates15_27 4718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_28

def innerTail15_26 : List Nat := 3889 :: innerTail15_27
def innerCandidates15_26 : List Nat := [5821]
set_option maxRecDepth 4096 in
theorem innerCandidates15_26_checked : candidates edge 4 61 3889 innerTail15_27 best.bound = innerCandidates15_26 := by
  unfold candidates
  have row : edge 3889 = SearchData.row3889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_26 : scan edge (search edge 3) 4 [61] 61 innerTail15_26 best = best := by
  rw [innerTail15_26, scan, if_neg (by decide), innerCandidates15_26_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_27
    (if 3 ≤ innerCandidates15_26.length then search edge 3 [61, 3889] innerCandidates15_26 3950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_27

def innerTail15_25 : List Nat := 3733 :: innerTail15_26
def innerCandidates15_25 : List Nat := [4657]
set_option maxRecDepth 4096 in
theorem innerCandidates15_25_checked : candidates edge 4 61 3733 innerTail15_26 best.bound = innerCandidates15_25 := by
  unfold candidates
  have row : edge 3733 = SearchData.row3733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_25 : scan edge (search edge 3) 4 [61] 61 innerTail15_25 best = best := by
  rw [innerTail15_25, scan, if_neg (by decide), innerCandidates15_25_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_26
    (if 3 ≤ innerCandidates15_25.length then search edge 3 [61, 3733] innerCandidates15_25 3794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_26

def innerTail15_24 : List Nat := 3637 :: innerTail15_25
def innerCandidates15_24 : List Nat := [6871]
set_option maxRecDepth 4096 in
theorem innerCandidates15_24_checked : candidates edge 4 61 3637 innerTail15_25 best.bound = innerCandidates15_24 := by
  unfold candidates
  have row : edge 3637 = SearchData.row3637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_24 : scan edge (search edge 3) 4 [61] 61 innerTail15_24 best = best := by
  rw [innerTail15_24, scan, if_neg (by decide), innerCandidates15_24_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_25
    (if 3 ≤ innerCandidates15_24.length then search edge 3 [61, 3637] innerCandidates15_24 3698 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_25

def innerTail15_23 : List Nat := 3469 :: innerTail15_24
def innerCandidates15_23 : List Nat := [8311]
set_option maxRecDepth 4096 in
theorem innerCandidates15_23_checked : candidates edge 4 61 3469 innerTail15_24 best.bound = innerCandidates15_23 := by
  unfold candidates
  have row : edge 3469 = SearchData.row3469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_23 : scan edge (search edge 3) 4 [61] 61 innerTail15_23 best = best := by
  rw [innerTail15_23, scan, if_neg (by decide), innerCandidates15_23_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_24
    (if 3 ≤ innerCandidates15_23.length then search edge 3 [61, 3469] innerCandidates15_23 3530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_24

def innerTail15_22 : List Nat := 3463 :: innerTail15_23
def innerCandidates15_22 : List Nat := [5869, 6079]
set_option maxRecDepth 4096 in
theorem innerCandidates15_22_checked : candidates edge 4 61 3463 innerTail15_23 best.bound = innerCandidates15_22 := by
  unfold candidates
  have row : edge 3463 = SearchData.row3463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_22 : scan edge (search edge 3) 4 [61] 61 innerTail15_22 best = best := by
  rw [innerTail15_22, scan, if_neg (by decide), innerCandidates15_22_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_23
    (if 3 ≤ innerCandidates15_22.length then search edge 3 [61, 3463] innerCandidates15_22 3524 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_23

def innerTail15_21 : List Nat := 3181 :: innerTail15_22
def innerCandidates15_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_21_checked : candidates edge 4 61 3181 innerTail15_22 best.bound = innerCandidates15_21 := by
  unfold candidates
  have row : edge 3181 = SearchData.row3181.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_21 : scan edge (search edge 3) 4 [61] 61 innerTail15_21 best = best := by
  rw [innerTail15_21, scan, if_neg (by decide), innerCandidates15_21_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_22
    (if 3 ≤ innerCandidates15_21.length then search edge 3 [61, 3181] innerCandidates15_21 3242 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_22

def innerTail15_20 : List Nat := 3169 :: innerTail15_21
def innerCandidates15_20 : List Nat := [3733]
set_option maxRecDepth 4096 in
theorem innerCandidates15_20_checked : candidates edge 4 61 3169 innerTail15_21 best.bound = innerCandidates15_20 := by
  unfold candidates
  have row : edge 3169 = SearchData.row3169.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_20 : scan edge (search edge 3) 4 [61] 61 innerTail15_20 best = best := by
  rw [innerTail15_20, scan, if_neg (by decide), innerCandidates15_20_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_21
    (if 3 ≤ innerCandidates15_20.length then search edge 3 [61, 3169] innerCandidates15_20 3230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_21

def innerTail15_19 : List Nat := 3049 :: innerTail15_20
def innerCandidates15_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_19_checked : candidates edge 4 61 3049 innerTail15_20 best.bound = innerCandidates15_19 := by
  unfold candidates
  have row : edge 3049 = SearchData.row3049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_19 : scan edge (search edge 3) 4 [61] 61 innerTail15_19 best = best := by
  rw [innerTail15_19, scan, if_neg (by decide), innerCandidates15_19_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_20
    (if 3 ≤ innerCandidates15_19.length then search edge 3 [61, 3049] innerCandidates15_19 3110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_20

def innerTail15_18 : List Nat := 2971 :: innerTail15_19
def innerCandidates15_18 : List Nat := [5557, 6871]
set_option maxRecDepth 4096 in
theorem innerCandidates15_18_checked : candidates edge 4 61 2971 innerTail15_19 best.bound = innerCandidates15_18 := by
  unfold candidates
  have row : edge 2971 = SearchData.row2971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_18 : scan edge (search edge 3) 4 [61] 61 innerTail15_18 best = best := by
  rw [innerTail15_18, scan, if_neg (by decide), innerCandidates15_18_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_19
    (if 3 ≤ innerCandidates15_18.length then search edge 3 [61, 2971] innerCandidates15_18 3032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_19

def innerTail15_17 : List Nat := 2797 :: innerTail15_18
def innerCandidates15_17 : List Nat := [5449, 6367, 7681]
set_option maxRecDepth 4096 in
theorem innerCandidates15_17_checked : candidates edge 4 61 2797 innerTail15_18 best.bound = innerCandidates15_17 := by
  unfold candidates
  have row : edge 2797 = SearchData.row2797.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild15_17 : search edge 3 [61, 2797] innerCandidates15_17 2858 best = best := by decide

theorem innerScan15_17 : scan edge (search edge 3) 4 [61] 61 innerTail15_17 best = best := by
  rw [innerTail15_17, scan, if_neg (by decide), innerCandidates15_17_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_18
    (if 3 ≤ innerCandidates15_17.length then search edge 3 [61, 2797] innerCandidates15_17 2858 best else best) = best
  rw [if_pos (by decide), innerChild15_17]
  exact innerScan15_18

def innerTail15_16 : List Nat := 2383 :: innerTail15_17
def innerCandidates15_16 : List Nat := [5431, 7129, 7873]
set_option maxRecDepth 4096 in
theorem innerCandidates15_16_checked : candidates edge 4 61 2383 innerTail15_17 best.bound = innerCandidates15_16 := by
  unfold candidates
  have row : edge 2383 = PrimePairCache.smallRow2383.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild15_16 : search edge 3 [61, 2383] innerCandidates15_16 2444 best = best := by decide

theorem innerScan15_16 : scan edge (search edge 3) 4 [61] 61 innerTail15_16 best = best := by
  rw [innerTail15_16, scan, if_neg (by decide), innerCandidates15_16_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_17
    (if 3 ≤ innerCandidates15_16.length then search edge 3 [61, 2383] innerCandidates15_16 2444 best else best) = best
  rw [if_pos (by decide), innerChild15_16]
  exact innerScan15_17

def innerTail15_15 : List Nat := 2371 :: innerTail15_16
def innerCandidates15_15 : List Nat := [3169]
set_option maxRecDepth 4096 in
theorem innerCandidates15_15_checked : candidates edge 4 61 2371 innerTail15_16 best.bound = innerCandidates15_15 := by
  unfold candidates
  have row : edge 2371 = PrimePairCache.smallRow2371.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_15 : scan edge (search edge 3) 4 [61] 61 innerTail15_15 best = best := by
  rw [innerTail15_15, scan, if_neg (by decide), innerCandidates15_15_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_16
    (if 3 ≤ innerCandidates15_15.length then search edge 3 [61, 2371] innerCandidates15_15 2432 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_16

def innerTail15_14 : List Nat := 2341 :: innerTail15_15
def innerCandidates15_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_14_checked : candidates edge 4 61 2341 innerTail15_15 best.bound = innerCandidates15_14 := by
  unfold candidates
  have row : edge 2341 = PrimePairCache.smallRow2341.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_14 : scan edge (search edge 3) 4 [61] 61 innerTail15_14 best = best := by
  rw [innerTail15_14, scan, if_neg (by decide), innerCandidates15_14_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_15
    (if 3 ≤ innerCandidates15_14.length then search edge 3 [61, 2341] innerCandidates15_14 2402 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_15

def innerTail15_13 : List Nat := 1999 :: innerTail15_14
def innerCandidates15_13 : List Nat := [3469, 3733, 5869]
set_option maxRecDepth 4096 in
theorem innerCandidates15_13_checked : candidates edge 4 61 1999 innerTail15_14 best.bound = innerCandidates15_13 := by
  unfold candidates
  have row : edge 1999 = PrimePairCache.smallRow1999.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild15_13 : search edge 3 [61, 1999] innerCandidates15_13 2060 best = best := by decide

theorem innerScan15_13 : scan edge (search edge 3) 4 [61] 61 innerTail15_13 best = best := by
  rw [innerTail15_13, scan, if_neg (by decide), innerCandidates15_13_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_14
    (if 3 ≤ innerCandidates15_13.length then search edge 3 [61, 1999] innerCandidates15_13 2060 best else best) = best
  rw [if_pos (by decide), innerChild15_13]
  exact innerScan15_14

def innerTail15_12 : List Nat := 1951 :: innerTail15_13
def innerCandidates15_12 : List Nat := [2341, 3733]
set_option maxRecDepth 4096 in
theorem innerCandidates15_12_checked : candidates edge 4 61 1951 innerTail15_13 best.bound = innerCandidates15_12 := by
  unfold candidates
  have row : edge 1951 = PrimePairCache.smallRow1951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_12 : scan edge (search edge 3) 4 [61] 61 innerTail15_12 best = best := by
  rw [innerTail15_12, scan, if_neg (by decide), innerCandidates15_12_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_13
    (if 3 ≤ innerCandidates15_12.length then search edge 3 [61, 1951] innerCandidates15_12 2012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_13

def innerTail15_11 : List Nat := 1873 :: innerTail15_12
def innerCandidates15_11 : List Nat := [6997, 7963]
set_option maxRecDepth 4096 in
theorem innerCandidates15_11_checked : candidates edge 4 61 1873 innerTail15_12 best.bound = innerCandidates15_11 := by
  unfold candidates
  have row : edge 1873 = PrimePairCache.smallRow1873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_11 : scan edge (search edge 3) 4 [61] 61 innerTail15_11 best = best := by
  rw [innerTail15_11, scan, if_neg (by decide), innerCandidates15_11_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_12
    (if 3 ≤ innerCandidates15_11.length then search edge 3 [61, 1873] innerCandidates15_11 1934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_12

def innerTail15_10 : List Nat := 1801 :: innerTail15_11
def innerCandidates15_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_10_checked : candidates edge 4 61 1801 innerTail15_11 best.bound = innerCandidates15_10 := by
  unfold candidates
  have row : edge 1801 = PrimePairCache.smallRow1801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_10 : scan edge (search edge 3) 4 [61] 61 innerTail15_10 best = best := by
  rw [innerTail15_10, scan, if_neg (by decide), innerCandidates15_10_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_11
    (if 3 ≤ innerCandidates15_10.length then search edge 3 [61, 1801] innerCandidates15_10 1862 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_11

def innerTail15_9 : List Nat := 1753 :: innerTail15_10
def innerCandidates15_9 : List Nat := [1873, 1951, 5827, 7237, 8161]
set_option maxRecDepth 4096 in
theorem innerCandidates15_9_checked : candidates edge 4 61 1753 innerTail15_10 best.bound = innerCandidates15_9 := by
  unfold candidates
  have row : edge 1753 = PrimePairCache.smallRow1753.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild15_9 : search edge 3 [61, 1753] innerCandidates15_9 1814 best = best := by decide

theorem innerScan15_9 : scan edge (search edge 3) 4 [61] 61 innerTail15_9 best = best := by
  rw [innerTail15_9, scan, if_neg (by decide), innerCandidates15_9_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_10
    (if 3 ≤ innerCandidates15_9.length then search edge 3 [61, 1753] innerCandidates15_9 1814 best else best) = best
  rw [if_pos (by decide), innerChild15_9]
  exact innerScan15_10

def innerTail15_8 : List Nat := 1693 :: innerTail15_9
def innerCandidates15_8 : List Nat := [1753, 2371, 3181]
set_option maxRecDepth 4096 in
theorem innerCandidates15_8_checked : candidates edge 4 61 1693 innerTail15_9 best.bound = innerCandidates15_8 := by
  unfold candidates
  have row : edge 1693 = PrimePairCache.smallRow1693.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild15_8 : search edge 3 [61, 1693] innerCandidates15_8 1754 best = best := by decide

theorem innerScan15_8 : scan edge (search edge 3) 4 [61] 61 innerTail15_8 best = best := by
  rw [innerTail15_8, scan, if_neg (by decide), innerCandidates15_8_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_9
    (if 3 ≤ innerCandidates15_8.length then search edge 3 [61, 1693] innerCandidates15_8 1754 best else best) = best
  rw [if_pos (by decide), innerChild15_8]
  exact innerScan15_9

def innerTail15_7 : List Nat := 1483 :: innerTail15_8
def innerCandidates15_7 : List Nat := [2341, 5869, 7873]
set_option maxRecDepth 4096 in
theorem innerCandidates15_7_checked : candidates edge 4 61 1483 innerTail15_8 best.bound = innerCandidates15_7 := by
  unfold candidates
  have row : edge 1483 = PrimePairCache.smallRow1483.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild15_7 : search edge 3 [61, 1483] innerCandidates15_7 1544 best = best := by decide

theorem innerScan15_7 : scan edge (search edge 3) 4 [61] 61 innerTail15_7 best = best := by
  rw [innerTail15_7, scan, if_neg (by decide), innerCandidates15_7_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_8
    (if 3 ≤ innerCandidates15_7.length then search edge 3 [61, 1483] innerCandidates15_7 1544 best else best) = best
  rw [if_pos (by decide), innerChild15_7]
  exact innerScan15_8

def innerTail15_6 : List Nat := 1453 :: innerTail15_7
def innerCandidates15_6 : List Nat := [2797, 3049, 7129]
set_option maxRecDepth 4096 in
theorem innerCandidates15_6_checked : candidates edge 4 61 1453 innerTail15_7 best.bound = innerCandidates15_6 := by
  unfold candidates
  have row : edge 1453 = PrimePairCache.smallRow1453.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild15_6 : search edge 3 [61, 1453] innerCandidates15_6 1514 best = best := by decide

theorem innerScan15_6 : scan edge (search edge 3) 4 [61] 61 innerTail15_6 best = best := by
  rw [innerTail15_6, scan, if_neg (by decide), innerCandidates15_6_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_7
    (if 3 ≤ innerCandidates15_6.length then search edge 3 [61, 1453] innerCandidates15_6 1514 best else best) = best
  rw [if_pos (by decide), innerChild15_6]
  exact innerScan15_7

def innerTail15_5 : List Nat := 1069 :: innerTail15_6
def innerCandidates15_5 : List Nat := [8161]
set_option maxRecDepth 4096 in
theorem innerCandidates15_5_checked : candidates edge 4 61 1069 innerTail15_6 best.bound = innerCandidates15_5 := by
  unfold candidates
  have row : edge 1069 = PrimePairCache.smallRow1069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_5 : scan edge (search edge 3) 4 [61] 61 innerTail15_5 best = best := by
  rw [innerTail15_5, scan, if_neg (by decide), innerCandidates15_5_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_6
    (if 3 ≤ innerCandidates15_5.length then search edge 3 [61, 1069] innerCandidates15_5 1130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_6

def innerTail15_4 : List Nat := 751 :: innerTail15_5
def innerCandidates15_4 : List Nat := [2383]
set_option maxRecDepth 4096 in
theorem innerCandidates15_4_checked : candidates edge 4 61 751 innerTail15_5 best.bound = innerCandidates15_4 := by
  unfold candidates
  have row : edge 751 = PrimePairCache.smallRow751.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_4 : scan edge (search edge 3) 4 [61] 61 innerTail15_4 best = best := by
  rw [innerTail15_4, scan, if_neg (by decide), innerCandidates15_4_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_5
    (if 3 ≤ innerCandidates15_4.length then search edge 3 [61, 751] innerCandidates15_4 812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_5

def innerTail15_3 : List Nat := 487 :: innerTail15_4
def innerCandidates15_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates15_3_checked : candidates edge 4 61 487 innerTail15_4 best.bound = innerCandidates15_3 := by
  unfold candidates
  have row : edge 487 = PrimePairCache.smallRow487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_3 : scan edge (search edge 3) 4 [61] 61 innerTail15_3 best = best := by
  rw [innerTail15_3, scan, if_neg (by decide), innerCandidates15_3_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_4
    (if 3 ≤ innerCandidates15_3.length then search edge 3 [61, 487] innerCandidates15_3 548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_4

def innerTail15_2 : List Nat := 409 :: innerTail15_3
def innerCandidates15_2 : List Nat := [1873, 7699]
set_option maxRecDepth 4096 in
theorem innerCandidates15_2_checked : candidates edge 4 61 409 innerTail15_3 best.bound = innerCandidates15_2 := by
  unfold candidates
  have row : edge 409 = PrimePairCache.smallRow409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_2 : scan edge (search edge 3) 4 [61] 61 innerTail15_2 best = best := by
  rw [innerTail15_2, scan, if_neg (by decide), innerCandidates15_2_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_3
    (if 3 ≤ innerCandidates15_2.length then search edge 3 [61, 409] innerCandidates15_2 470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_3

def innerTail15_1 : List Nat := 331 :: innerTail15_2
def innerCandidates15_1 : List Nat := [1999, 3637]
set_option maxRecDepth 4096 in
theorem innerCandidates15_1_checked : candidates edge 4 61 331 innerTail15_2 best.bound = innerCandidates15_1 := by
  unfold candidates
  have row : edge 331 = PrimePairCache.smallRow331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_1 : scan edge (search edge 3) 4 [61] 61 innerTail15_1 best = best := by
  rw [innerTail15_1, scan, if_neg (by decide), innerCandidates15_1_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_2
    (if 3 ≤ innerCandidates15_1.length then search edge 3 [61, 331] innerCandidates15_1 392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_2

def innerTail15_0 : List Nat := 151 :: innerTail15_1
def innerCandidates15_0 : List Nat := [4657, 5413]
set_option maxRecDepth 4096 in
theorem innerCandidates15_0_checked : candidates edge 4 61 151 innerTail15_1 best.bound = innerCandidates15_0 := by
  unfold candidates
  have row : edge 151 = PrimePairCache.smallRow151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan15_0 : scan edge (search edge 3) 4 [61] 61 innerTail15_0 best = best := by
  rw [innerTail15_0, scan, if_neg (by decide), innerCandidates15_0_checked]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_1
    (if 3 ≤ innerCandidates15_0.length then search edge 3 [61, 151] innerCandidates15_0 212 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan15_1

theorem rootBranch15 : search edge 4 [61] rootCandidates15 61 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [61] 61 innerTail15_0 best = best
  exact innerScan15_0

end ProjectEulerSolutions.PrimePairUpper.Search
