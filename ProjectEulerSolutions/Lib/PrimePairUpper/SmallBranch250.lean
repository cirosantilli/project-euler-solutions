import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates17 : List Nat := [233, 257, 263, 317, 389, 443, 719, 821, 887, 947, 971, 1049, 1163, 1181, 1427, 1499, 1583, 1709, 1973, 2417, 2477, 2687, 2819, 2837, 2909, 3347, 3491, 3833, 3863, 4517, 4673, 5087, 5153, 5237, 5417, 5441, 5639, 5651, 5657, 5849, 5879, 6389, 6563, 6869, 7001, 7103, 7109, 7151, 7229, 7589, 7817, 7841, 7919, 8087, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates17_checked : candidates edge 5 0 71 tail18 best.bound = rootCandidates17 := by
  unfold candidates
  have row : edge 71 = PrimePairCache.smallRow71.contains := by funext q; rfl
  rw [row]
  decide

def innerTail17_55 : List Nat := []
theorem innerScan17_55 : scan edge (search edge 3) 4 [71] 71 innerTail17_55 best = best := rfl

def innerTail17_54 : List Nat := 8387 :: innerTail17_55
def innerCandidates17_54 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates17_54_checked : candidates edge 4 71 8387 innerTail17_55 best.bound = innerCandidates17_54 := by
  unfold candidates
  have row : edge 8387 = SearchData.row8387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_54 : scan edge (search edge 3) 4 [71] 71 innerTail17_54 best = best := by
  rw [innerTail17_54, scan, if_neg (by decide), innerCandidates17_54_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_55
    (if 3 ≤ innerCandidates17_54.length then search edge 3 [71, 8387] innerCandidates17_54 8458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_55

def innerTail17_53 : List Nat := 8087 :: innerTail17_54
def innerCandidates17_53 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates17_53_checked : candidates edge 4 71 8087 innerTail17_54 best.bound = innerCandidates17_53 := by
  unfold candidates
  have row : edge 8087 = SearchData.row8087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_53 : scan edge (search edge 3) 4 [71] 71 innerTail17_53 best = best := by
  rw [innerTail17_53, scan, if_neg (by decide), innerCandidates17_53_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_54
    (if 3 ≤ innerCandidates17_53.length then search edge 3 [71, 8087] innerCandidates17_53 8158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_54

def innerTail17_52 : List Nat := 7919 :: innerTail17_53
def innerCandidates17_52 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates17_52_checked : candidates edge 4 71 7919 innerTail17_53 best.bound = innerCandidates17_52 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_52 : scan edge (search edge 3) 4 [71] 71 innerTail17_52 best = best := by
  rw [innerTail17_52, scan, if_neg (by decide), innerCandidates17_52_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_53
    (if 3 ≤ innerCandidates17_52.length then search edge 3 [71, 7919] innerCandidates17_52 7990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_53

def innerTail17_51 : List Nat := 7841 :: innerTail17_52
def innerCandidates17_51 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates17_51_checked : candidates edge 4 71 7841 innerTail17_52 best.bound = innerCandidates17_51 := by
  unfold candidates
  have row : edge 7841 = SearchData.row7841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_51 : scan edge (search edge 3) 4 [71] 71 innerTail17_51 best = best := by
  rw [innerTail17_51, scan, if_neg (by decide), innerCandidates17_51_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_52
    (if 3 ≤ innerCandidates17_51.length then search edge 3 [71, 7841] innerCandidates17_51 7912 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_52

def innerTail17_50 : List Nat := 7817 :: innerTail17_51
def innerCandidates17_50 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates17_50_checked : candidates edge 4 71 7817 innerTail17_51 best.bound = innerCandidates17_50 := by
  unfold candidates
  have row : edge 7817 = SearchData.row7817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_50 : scan edge (search edge 3) 4 [71] 71 innerTail17_50 best = best := by
  rw [innerTail17_50, scan, if_neg (by decide), innerCandidates17_50_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_51
    (if 3 ≤ innerCandidates17_50.length then search edge 3 [71, 7817] innerCandidates17_50 7888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_51

def innerTail17_49 : List Nat := 7589 :: innerTail17_50
def innerCandidates17_49 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates17_49_checked : candidates edge 4 71 7589 innerTail17_50 best.bound = innerCandidates17_49 := by
  unfold candidates
  have row : edge 7589 = SearchData.row7589.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_49 : scan edge (search edge 3) 4 [71] 71 innerTail17_49 best = best := by
  rw [innerTail17_49, scan, if_neg (by decide), innerCandidates17_49_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_50
    (if 3 ≤ innerCandidates17_49.length then search edge 3 [71, 7589] innerCandidates17_49 7660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_50

def innerTail17_48 : List Nat := 7229 :: innerTail17_49
def innerCandidates17_48 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates17_48_checked : candidates edge 4 71 7229 innerTail17_49 best.bound = innerCandidates17_48 := by
  unfold candidates
  have row : edge 7229 = SearchData.row7229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_48 : scan edge (search edge 3) 4 [71] 71 innerTail17_48 best = best := by
  rw [innerTail17_48, scan, if_neg (by decide), innerCandidates17_48_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_49
    (if 3 ≤ innerCandidates17_48.length then search edge 3 [71, 7229] innerCandidates17_48 7300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_49

def innerTail17_47 : List Nat := 7151 :: innerTail17_48
def innerCandidates17_47 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates17_47_checked : candidates edge 4 71 7151 innerTail17_48 best.bound = innerCandidates17_47 := by
  unfold candidates
  have row : edge 7151 = SearchData.row7151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_47 : scan edge (search edge 3) 4 [71] 71 innerTail17_47 best = best := by
  rw [innerTail17_47, scan, if_neg (by decide), innerCandidates17_47_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_48
    (if 3 ≤ innerCandidates17_47.length then search edge 3 [71, 7151] innerCandidates17_47 7222 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_48

def innerTail17_46 : List Nat := 7109 :: innerTail17_47
def innerCandidates17_46 : List Nat := [7841]
set_option maxRecDepth 4096 in
theorem innerCandidates17_46_checked : candidates edge 4 71 7109 innerTail17_47 best.bound = innerCandidates17_46 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_46 : scan edge (search edge 3) 4 [71] 71 innerTail17_46 best = best := by
  rw [innerTail17_46, scan, if_neg (by decide), innerCandidates17_46_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_47
    (if 3 ≤ innerCandidates17_46.length then search edge 3 [71, 7109] innerCandidates17_46 7180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_47

def innerTail17_45 : List Nat := 7103 :: innerTail17_46
def innerCandidates17_45 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates17_45_checked : candidates edge 4 71 7103 innerTail17_46 best.bound = innerCandidates17_45 := by
  unfold candidates
  have row : edge 7103 = SearchData.row7103.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_45 : scan edge (search edge 3) 4 [71] 71 innerTail17_45 best = best := by
  rw [innerTail17_45, scan, if_neg (by decide), innerCandidates17_45_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_46
    (if 3 ≤ innerCandidates17_45.length then search edge 3 [71, 7103] innerCandidates17_45 7174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_46

def innerTail17_44 : List Nat := 7001 :: innerTail17_45
def innerCandidates17_44 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates17_44_checked : candidates edge 4 71 7001 innerTail17_45 best.bound = innerCandidates17_44 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_44 : scan edge (search edge 3) 4 [71] 71 innerTail17_44 best = best := by
  rw [innerTail17_44, scan, if_neg (by decide), innerCandidates17_44_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_45
    (if 3 ≤ innerCandidates17_44.length then search edge 3 [71, 7001] innerCandidates17_44 7072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_45

def innerTail17_43 : List Nat := 6869 :: innerTail17_44
def innerCandidates17_43 : List Nat := [8087]
set_option maxRecDepth 4096 in
theorem innerCandidates17_43_checked : candidates edge 4 71 6869 innerTail17_44 best.bound = innerCandidates17_43 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_43 : scan edge (search edge 3) 4 [71] 71 innerTail17_43 best = best := by
  rw [innerTail17_43, scan, if_neg (by decide), innerCandidates17_43_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_44
    (if 3 ≤ innerCandidates17_43.length then search edge 3 [71, 6869] innerCandidates17_43 6940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_44

def innerTail17_42 : List Nat := 6563 :: innerTail17_43
def innerCandidates17_42 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates17_42_checked : candidates edge 4 71 6563 innerTail17_43 best.bound = innerCandidates17_42 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_42 : scan edge (search edge 3) 4 [71] 71 innerTail17_42 best = best := by
  rw [innerTail17_42, scan, if_neg (by decide), innerCandidates17_42_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_43
    (if 3 ≤ innerCandidates17_42.length then search edge 3 [71, 6563] innerCandidates17_42 6634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_43

def innerTail17_41 : List Nat := 6389 :: innerTail17_42
def innerCandidates17_41 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates17_41_checked : candidates edge 4 71 6389 innerTail17_42 best.bound = innerCandidates17_41 := by
  unfold candidates
  have row : edge 6389 = SearchData.row6389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_41 : scan edge (search edge 3) 4 [71] 71 innerTail17_41 best = best := by
  rw [innerTail17_41, scan, if_neg (by decide), innerCandidates17_41_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_42
    (if 3 ≤ innerCandidates17_41.length then search edge 3 [71, 6389] innerCandidates17_41 6460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_42

def innerTail17_40 : List Nat := 5879 :: innerTail17_41
def innerCandidates17_40 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates17_40_checked : candidates edge 4 71 5879 innerTail17_41 best.bound = innerCandidates17_40 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_40 : scan edge (search edge 3) 4 [71] 71 innerTail17_40 best = best := by
  rw [innerTail17_40, scan, if_neg (by decide), innerCandidates17_40_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_41
    (if 3 ≤ innerCandidates17_40.length then search edge 3 [71, 5879] innerCandidates17_40 5950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_41

def innerTail17_39 : List Nat := 5849 :: innerTail17_40
def innerCandidates17_39 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates17_39_checked : candidates edge 4 71 5849 innerTail17_40 best.bound = innerCandidates17_39 := by
  unfold candidates
  have row : edge 5849 = SearchData.row5849.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_39 : scan edge (search edge 3) 4 [71] 71 innerTail17_39 best = best := by
  rw [innerTail17_39, scan, if_neg (by decide), innerCandidates17_39_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_40
    (if 3 ≤ innerCandidates17_39.length then search edge 3 [71, 5849] innerCandidates17_39 5920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_40

def innerTail17_38 : List Nat := 5657 :: innerTail17_39
def innerCandidates17_38 : List Nat := [8387]
set_option maxRecDepth 4096 in
theorem innerCandidates17_38_checked : candidates edge 4 71 5657 innerTail17_39 best.bound = innerCandidates17_38 := by
  unfold candidates
  have row : edge 5657 = SearchData.row5657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_38 : scan edge (search edge 3) 4 [71] 71 innerTail17_38 best = best := by
  rw [innerTail17_38, scan, if_neg (by decide), innerCandidates17_38_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_39
    (if 3 ≤ innerCandidates17_38.length then search edge 3 [71, 5657] innerCandidates17_38 5728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_39

def innerTail17_37 : List Nat := 5651 :: innerTail17_38
def innerCandidates17_37 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates17_37_checked : candidates edge 4 71 5651 innerTail17_38 best.bound = innerCandidates17_37 := by
  unfold candidates
  have row : edge 5651 = SearchData.row5651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_37 : scan edge (search edge 3) 4 [71] 71 innerTail17_37 best = best := by
  rw [innerTail17_37, scan, if_neg (by decide), innerCandidates17_37_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_38
    (if 3 ≤ innerCandidates17_37.length then search edge 3 [71, 5651] innerCandidates17_37 5722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_38

def innerTail17_36 : List Nat := 5639 :: innerTail17_37
def innerCandidates17_36 : List Nat := [6563]
set_option maxRecDepth 4096 in
theorem innerCandidates17_36_checked : candidates edge 4 71 5639 innerTail17_37 best.bound = innerCandidates17_36 := by
  unfold candidates
  have row : edge 5639 = SearchData.row5639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_36 : scan edge (search edge 3) 4 [71] 71 innerTail17_36 best = best := by
  rw [innerTail17_36, scan, if_neg (by decide), innerCandidates17_36_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_37
    (if 3 ≤ innerCandidates17_36.length then search edge 3 [71, 5639] innerCandidates17_36 5710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_37

def innerTail17_35 : List Nat := 5441 :: innerTail17_36
def innerCandidates17_35 : List Nat := [5849]
set_option maxRecDepth 4096 in
theorem innerCandidates17_35_checked : candidates edge 4 71 5441 innerTail17_36 best.bound = innerCandidates17_35 := by
  unfold candidates
  have row : edge 5441 = SearchData.row5441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_35 : scan edge (search edge 3) 4 [71] 71 innerTail17_35 best = best := by
  rw [innerTail17_35, scan, if_neg (by decide), innerCandidates17_35_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_36
    (if 3 ≤ innerCandidates17_35.length then search edge 3 [71, 5441] innerCandidates17_35 5512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_36

def innerTail17_34 : List Nat := 5417 :: innerTail17_35
def innerCandidates17_34 : List Nat := [7001]
set_option maxRecDepth 4096 in
theorem innerCandidates17_34_checked : candidates edge 4 71 5417 innerTail17_35 best.bound = innerCandidates17_34 := by
  unfold candidates
  have row : edge 5417 = SearchData.row5417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_34 : scan edge (search edge 3) 4 [71] 71 innerTail17_34 best = best := by
  rw [innerTail17_34, scan, if_neg (by decide), innerCandidates17_34_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_35
    (if 3 ≤ innerCandidates17_34.length then search edge 3 [71, 5417] innerCandidates17_34 5488 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_35

def innerTail17_33 : List Nat := 5237 :: innerTail17_34
def innerCandidates17_33 : List Nat := [6563, 7151]
set_option maxRecDepth 4096 in
theorem innerCandidates17_33_checked : candidates edge 4 71 5237 innerTail17_34 best.bound = innerCandidates17_33 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_33 : scan edge (search edge 3) 4 [71] 71 innerTail17_33 best = best := by
  rw [innerTail17_33, scan, if_neg (by decide), innerCandidates17_33_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_34
    (if 3 ≤ innerCandidates17_33.length then search edge 3 [71, 5237] innerCandidates17_33 5308 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_34

def innerTail17_32 : List Nat := 5153 :: innerTail17_33
def innerCandidates17_32 : List Nat := [5849, 7589]
set_option maxRecDepth 4096 in
theorem innerCandidates17_32_checked : candidates edge 4 71 5153 innerTail17_33 best.bound = innerCandidates17_32 := by
  unfold candidates
  have row : edge 5153 = SearchData.row5153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_32 : scan edge (search edge 3) 4 [71] 71 innerTail17_32 best = best := by
  rw [innerTail17_32, scan, if_neg (by decide), innerCandidates17_32_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_33
    (if 3 ≤ innerCandidates17_32.length then search edge 3 [71, 5153] innerCandidates17_32 5224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_33

def innerTail17_31 : List Nat := 5087 :: innerTail17_32
def innerCandidates17_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates17_31_checked : candidates edge 4 71 5087 innerTail17_32 best.bound = innerCandidates17_31 := by
  unfold candidates
  have row : edge 5087 = SearchData.row5087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_31 : scan edge (search edge 3) 4 [71] 71 innerTail17_31 best = best := by
  rw [innerTail17_31, scan, if_neg (by decide), innerCandidates17_31_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_32
    (if 3 ≤ innerCandidates17_31.length then search edge 3 [71, 5087] innerCandidates17_31 5158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_32

def innerTail17_30 : List Nat := 4673 :: innerTail17_31
def innerCandidates17_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates17_30_checked : candidates edge 4 71 4673 innerTail17_31 best.bound = innerCandidates17_30 := by
  unfold candidates
  have row : edge 4673 = SearchData.row4673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_30 : scan edge (search edge 3) 4 [71] 71 innerTail17_30 best = best := by
  rw [innerTail17_30, scan, if_neg (by decide), innerCandidates17_30_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_31
    (if 3 ≤ innerCandidates17_30.length then search edge 3 [71, 4673] innerCandidates17_30 4744 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_31

def innerTail17_29 : List Nat := 4517 :: innerTail17_30
def innerCandidates17_29 : List Nat := [5087, 5849, 6869, 8087, 8387]
set_option maxRecDepth 4096 in
theorem innerCandidates17_29_checked : candidates edge 4 71 4517 innerTail17_30 best.bound = innerCandidates17_29 := by
  unfold candidates
  have row : edge 4517 = SearchData.row4517.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild17_29 : search edge 3 [71, 4517] innerCandidates17_29 4588 best = best := by decide

theorem innerScan17_29 : scan edge (search edge 3) 4 [71] 71 innerTail17_29 best = best := by
  rw [innerTail17_29, scan, if_neg (by decide), innerCandidates17_29_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_30
    (if 3 ≤ innerCandidates17_29.length then search edge 3 [71, 4517] innerCandidates17_29 4588 best else best) = best
  rw [if_pos (by decide), innerChild17_29]
  exact innerScan17_30

def innerTail17_28 : List Nat := 3863 :: innerTail17_29
def innerCandidates17_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates17_28_checked : candidates edge 4 71 3863 innerTail17_29 best.bound = innerCandidates17_28 := by
  unfold candidates
  have row : edge 3863 = SearchData.row3863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_28 : scan edge (search edge 3) 4 [71] 71 innerTail17_28 best = best := by
  rw [innerTail17_28, scan, if_neg (by decide), innerCandidates17_28_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_29
    (if 3 ≤ innerCandidates17_28.length then search edge 3 [71, 3863] innerCandidates17_28 3934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_29

def innerTail17_27 : List Nat := 3833 :: innerTail17_28
def innerCandidates17_27 : List Nat := [5657, 7841]
set_option maxRecDepth 4096 in
theorem innerCandidates17_27_checked : candidates edge 4 71 3833 innerTail17_28 best.bound = innerCandidates17_27 := by
  unfold candidates
  have row : edge 3833 = SearchData.row3833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_27 : scan edge (search edge 3) 4 [71] 71 innerTail17_27 best = best := by
  rw [innerTail17_27, scan, if_neg (by decide), innerCandidates17_27_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_28
    (if 3 ≤ innerCandidates17_27.length then search edge 3 [71, 3833] innerCandidates17_27 3904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_28

def innerTail17_26 : List Nat := 3491 :: innerTail17_27
def innerCandidates17_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates17_26_checked : candidates edge 4 71 3491 innerTail17_27 best.bound = innerCandidates17_26 := by
  unfold candidates
  have row : edge 3491 = SearchData.row3491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_26 : scan edge (search edge 3) 4 [71] 71 innerTail17_26 best = best := by
  rw [innerTail17_26, scan, if_neg (by decide), innerCandidates17_26_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_27
    (if 3 ≤ innerCandidates17_26.length then search edge 3 [71, 3491] innerCandidates17_26 3562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_27

def innerTail17_25 : List Nat := 3347 :: innerTail17_26
def innerCandidates17_25 : List Nat := [4673]
set_option maxRecDepth 4096 in
theorem innerCandidates17_25_checked : candidates edge 4 71 3347 innerTail17_26 best.bound = innerCandidates17_25 := by
  unfold candidates
  have row : edge 3347 = SearchData.row3347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_25 : scan edge (search edge 3) 4 [71] 71 innerTail17_25 best = best := by
  rw [innerTail17_25, scan, if_neg (by decide), innerCandidates17_25_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_26
    (if 3 ≤ innerCandidates17_25.length then search edge 3 [71, 3347] innerCandidates17_25 3418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_26

def innerTail17_24 : List Nat := 2909 :: innerTail17_25
def innerCandidates17_24 : List Nat := [5153]
set_option maxRecDepth 4096 in
theorem innerCandidates17_24_checked : candidates edge 4 71 2909 innerTail17_25 best.bound = innerCandidates17_24 := by
  unfold candidates
  have row : edge 2909 = SearchData.row2909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_24 : scan edge (search edge 3) 4 [71] 71 innerTail17_24 best = best := by
  rw [innerTail17_24, scan, if_neg (by decide), innerCandidates17_24_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_25
    (if 3 ≤ innerCandidates17_24.length then search edge 3 [71, 2909] innerCandidates17_24 2980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_25

def innerTail17_23 : List Nat := 2837 :: innerTail17_24
def innerCandidates17_23 : List Nat := [5153, 7001]
set_option maxRecDepth 4096 in
theorem innerCandidates17_23_checked : candidates edge 4 71 2837 innerTail17_24 best.bound = innerCandidates17_23 := by
  unfold candidates
  have row : edge 2837 = SearchData.row2837.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_23 : scan edge (search edge 3) 4 [71] 71 innerTail17_23 best = best := by
  rw [innerTail17_23, scan, if_neg (by decide), innerCandidates17_23_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_24
    (if 3 ≤ innerCandidates17_23.length then search edge 3 [71, 2837] innerCandidates17_23 2908 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_24

def innerTail17_22 : List Nat := 2819 :: innerTail17_23
def innerCandidates17_22 : List Nat := [5879]
set_option maxRecDepth 4096 in
theorem innerCandidates17_22_checked : candidates edge 4 71 2819 innerTail17_23 best.bound = innerCandidates17_22 := by
  unfold candidates
  have row : edge 2819 = SearchData.row2819.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_22 : scan edge (search edge 3) 4 [71] 71 innerTail17_22 best = best := by
  rw [innerTail17_22, scan, if_neg (by decide), innerCandidates17_22_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_23
    (if 3 ≤ innerCandidates17_22.length then search edge 3 [71, 2819] innerCandidates17_22 2890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_23

def innerTail17_21 : List Nat := 2687 :: innerTail17_22
def innerCandidates17_21 : List Nat := [5237]
set_option maxRecDepth 4096 in
theorem innerCandidates17_21_checked : candidates edge 4 71 2687 innerTail17_22 best.bound = innerCandidates17_21 := by
  unfold candidates
  have row : edge 2687 = SearchData.row2687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_21 : scan edge (search edge 3) 4 [71] 71 innerTail17_21 best = best := by
  rw [innerTail17_21, scan, if_neg (by decide), innerCandidates17_21_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_22
    (if 3 ≤ innerCandidates17_21.length then search edge 3 [71, 2687] innerCandidates17_21 2758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_22

def innerTail17_20 : List Nat := 2477 :: innerTail17_21
def innerCandidates17_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates17_20_checked : candidates edge 4 71 2477 innerTail17_21 best.bound = innerCandidates17_20 := by
  unfold candidates
  have row : edge 2477 = PrimePairCache.smallRow2477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_20 : scan edge (search edge 3) 4 [71] 71 innerTail17_20 best = best := by
  rw [innerTail17_20, scan, if_neg (by decide), innerCandidates17_20_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_21
    (if 3 ≤ innerCandidates17_20.length then search edge 3 [71, 2477] innerCandidates17_20 2548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_21

def innerTail17_19 : List Nat := 2417 :: innerTail17_20
def innerCandidates17_19 : List Nat := [5153, 6389, 6563, 7103]
set_option maxRecDepth 4096 in
theorem innerCandidates17_19_checked : candidates edge 4 71 2417 innerTail17_20 best.bound = innerCandidates17_19 := by
  unfold candidates
  have row : edge 2417 = PrimePairCache.smallRow2417.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild17_19 : search edge 3 [71, 2417] innerCandidates17_19 2488 best = best := by decide

theorem innerScan17_19 : scan edge (search edge 3) 4 [71] 71 innerTail17_19 best = best := by
  rw [innerTail17_19, scan, if_neg (by decide), innerCandidates17_19_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_20
    (if 3 ≤ innerCandidates17_19.length then search edge 3 [71, 2417] innerCandidates17_19 2488 best else best) = best
  rw [if_pos (by decide), innerChild17_19]
  exact innerScan17_20

def innerTail17_18 : List Nat := 1973 :: innerTail17_19
def innerCandidates17_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates17_18_checked : candidates edge 4 71 1973 innerTail17_19 best.bound = innerCandidates17_18 := by
  unfold candidates
  have row : edge 1973 = PrimePairCache.smallRow1973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_18 : scan edge (search edge 3) 4 [71] 71 innerTail17_18 best = best := by
  rw [innerTail17_18, scan, if_neg (by decide), innerCandidates17_18_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_19
    (if 3 ≤ innerCandidates17_18.length then search edge 3 [71, 1973] innerCandidates17_18 2044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_19

def innerTail17_17 : List Nat := 1709 :: innerTail17_18
def innerCandidates17_17 : List Nat := [6869, 7001]
set_option maxRecDepth 4096 in
theorem innerCandidates17_17_checked : candidates edge 4 71 1709 innerTail17_18 best.bound = innerCandidates17_17 := by
  unfold candidates
  have row : edge 1709 = PrimePairCache.smallRow1709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_17 : scan edge (search edge 3) 4 [71] 71 innerTail17_17 best = best := by
  rw [innerTail17_17, scan, if_neg (by decide), innerCandidates17_17_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_18
    (if 3 ≤ innerCandidates17_17.length then search edge 3 [71, 1709] innerCandidates17_17 1780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_18

def innerTail17_16 : List Nat := 1583 :: innerTail17_17
def innerCandidates17_16 : List Nat := [1973, 3347, 5879, 6869]
set_option maxRecDepth 4096 in
theorem innerCandidates17_16_checked : candidates edge 4 71 1583 innerTail17_17 best.bound = innerCandidates17_16 := by
  unfold candidates
  have row : edge 1583 = PrimePairCache.smallRow1583.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild17_16 : search edge 3 [71, 1583] innerCandidates17_16 1654 best = best := by decide

theorem innerScan17_16 : scan edge (search edge 3) 4 [71] 71 innerTail17_16 best = best := by
  rw [innerTail17_16, scan, if_neg (by decide), innerCandidates17_16_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_17
    (if 3 ≤ innerCandidates17_16.length then search edge 3 [71, 1583] innerCandidates17_16 1654 best else best) = best
  rw [if_pos (by decide), innerChild17_16]
  exact innerScan17_17

def innerTail17_15 : List Nat := 1499 :: innerTail17_16
def innerCandidates17_15 : List Nat := [6563, 7589]
set_option maxRecDepth 4096 in
theorem innerCandidates17_15_checked : candidates edge 4 71 1499 innerTail17_16 best.bound = innerCandidates17_15 := by
  unfold candidates
  have row : edge 1499 = PrimePairCache.smallRow1499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_15 : scan edge (search edge 3) 4 [71] 71 innerTail17_15 best = best := by
  rw [innerTail17_15, scan, if_neg (by decide), innerCandidates17_15_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_16
    (if 3 ≤ innerCandidates17_15.length then search edge 3 [71, 1499] innerCandidates17_15 1570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_16

def innerTail17_14 : List Nat := 1427 :: innerTail17_15
def innerCandidates17_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates17_14_checked : candidates edge 4 71 1427 innerTail17_15 best.bound = innerCandidates17_14 := by
  unfold candidates
  have row : edge 1427 = PrimePairCache.smallRow1427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_14 : scan edge (search edge 3) 4 [71] 71 innerTail17_14 best = best := by
  rw [innerTail17_14, scan, if_neg (by decide), innerCandidates17_14_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_15
    (if 3 ≤ innerCandidates17_14.length then search edge 3 [71, 1427] innerCandidates17_14 1498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_15

def innerTail17_13 : List Nat := 1181 :: innerTail17_14
def innerCandidates17_13 : List Nat := [1427, 2687]
set_option maxRecDepth 4096 in
theorem innerCandidates17_13_checked : candidates edge 4 71 1181 innerTail17_14 best.bound = innerCandidates17_13 := by
  unfold candidates
  have row : edge 1181 = PrimePairCache.smallRow1181.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_13 : scan edge (search edge 3) 4 [71] 71 innerTail17_13 best = best := by
  rw [innerTail17_13, scan, if_neg (by decide), innerCandidates17_13_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_14
    (if 3 ≤ innerCandidates17_13.length then search edge 3 [71, 1181] innerCandidates17_13 1252 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_14

def innerTail17_12 : List Nat := 1163 :: innerTail17_13
def innerCandidates17_12 : List Nat := [2417, 4673, 5639, 5849, 7919]
set_option maxRecDepth 4096 in
theorem innerCandidates17_12_checked : candidates edge 4 71 1163 innerTail17_13 best.bound = innerCandidates17_12 := by
  unfold candidates
  have row : edge 1163 = PrimePairCache.smallRow1163.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild17_12 : search edge 3 [71, 1163] innerCandidates17_12 1234 best = best := by decide

theorem innerScan17_12 : scan edge (search edge 3) 4 [71] 71 innerTail17_12 best = best := by
  rw [innerTail17_12, scan, if_neg (by decide), innerCandidates17_12_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_13
    (if 3 ≤ innerCandidates17_12.length then search edge 3 [71, 1163] innerCandidates17_12 1234 best else best) = best
  rw [if_pos (by decide), innerChild17_12]
  exact innerScan17_13

def innerTail17_11 : List Nat := 1049 :: innerTail17_12
def innerCandidates17_11 : List Nat := [1181, 7229, 7841]
set_option maxRecDepth 4096 in
theorem innerCandidates17_11_checked : candidates edge 4 71 1049 innerTail17_12 best.bound = innerCandidates17_11 := by
  unfold candidates
  have row : edge 1049 = PrimePairCache.smallRow1049.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild17_11 : search edge 3 [71, 1049] innerCandidates17_11 1120 best = best := by decide

theorem innerScan17_11 : scan edge (search edge 3) 4 [71] 71 innerTail17_11 best = best := by
  rw [innerTail17_11, scan, if_neg (by decide), innerCandidates17_11_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_12
    (if 3 ≤ innerCandidates17_11.length then search edge 3 [71, 1049] innerCandidates17_11 1120 best else best) = best
  rw [if_pos (by decide), innerChild17_11]
  exact innerScan17_12

def innerTail17_10 : List Nat := 971 :: innerTail17_11
def innerCandidates17_10 : List Nat := [3863, 7919]
set_option maxRecDepth 4096 in
theorem innerCandidates17_10_checked : candidates edge 4 71 971 innerTail17_11 best.bound = innerCandidates17_10 := by
  unfold candidates
  have row : edge 971 = PrimePairCache.smallRow971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_10 : scan edge (search edge 3) 4 [71] 71 innerTail17_10 best = best := by
  rw [innerTail17_10, scan, if_neg (by decide), innerCandidates17_10_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_11
    (if 3 ≤ innerCandidates17_10.length then search edge 3 [71, 971] innerCandidates17_10 1042 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_11

def innerTail17_9 : List Nat := 947 :: innerTail17_10
def innerCandidates17_9 : List Nat := [3491, 8387]
set_option maxRecDepth 4096 in
theorem innerCandidates17_9_checked : candidates edge 4 71 947 innerTail17_10 best.bound = innerCandidates17_9 := by
  unfold candidates
  have row : edge 947 = PrimePairCache.smallRow947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_9 : scan edge (search edge 3) 4 [71] 71 innerTail17_9 best = best := by
  rw [innerTail17_9, scan, if_neg (by decide), innerCandidates17_9_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_10
    (if 3 ≤ innerCandidates17_9.length then search edge 3 [71, 947] innerCandidates17_9 1018 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_10

def innerTail17_8 : List Nat := 887 :: innerTail17_9
def innerCandidates17_8 : List Nat := [1427, 5417, 5651, 7109, 8087]
set_option maxRecDepth 4096 in
theorem innerCandidates17_8_checked : candidates edge 4 71 887 innerTail17_9 best.bound = innerCandidates17_8 := by
  unfold candidates
  have row : edge 887 = PrimePairCache.smallRow887.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild17_8 : search edge 3 [71, 887] innerCandidates17_8 958 best = best := by decide

theorem innerScan17_8 : scan edge (search edge 3) 4 [71] 71 innerTail17_8 best = best := by
  rw [innerTail17_8, scan, if_neg (by decide), innerCandidates17_8_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_9
    (if 3 ≤ innerCandidates17_8.length then search edge 3 [71, 887] innerCandidates17_8 958 best else best) = best
  rw [if_pos (by decide), innerChild17_8]
  exact innerScan17_9

def innerTail17_7 : List Nat := 821 :: innerTail17_8
def innerCandidates17_7 : List Nat := [971, 1163, 5849]
set_option maxRecDepth 4096 in
theorem innerCandidates17_7_checked : candidates edge 4 71 821 innerTail17_8 best.bound = innerCandidates17_7 := by
  unfold candidates
  have row : edge 821 = PrimePairCache.smallRow821.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild17_7 : search edge 3 [71, 821] innerCandidates17_7 892 best = best := by decide

theorem innerScan17_7 : scan edge (search edge 3) 4 [71] 71 innerTail17_7 best = best := by
  rw [innerTail17_7, scan, if_neg (by decide), innerCandidates17_7_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_8
    (if 3 ≤ innerCandidates17_7.length then search edge 3 [71, 821] innerCandidates17_7 892 best else best) = best
  rw [if_pos (by decide), innerChild17_7]
  exact innerScan17_8

def innerTail17_6 : List Nat := 719 :: innerTail17_7
def innerCandidates17_6 : List Nat := [947, 3833, 6869]
set_option maxRecDepth 4096 in
theorem innerCandidates17_6_checked : candidates edge 4 71 719 innerTail17_7 best.bound = innerCandidates17_6 := by
  unfold candidates
  have row : edge 719 = PrimePairCache.smallRow719.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild17_6 : search edge 3 [71, 719] innerCandidates17_6 790 best = best := by decide

theorem innerScan17_6 : scan edge (search edge 3) 4 [71] 71 innerTail17_6 best = best := by
  rw [innerTail17_6, scan, if_neg (by decide), innerCandidates17_6_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_7
    (if 3 ≤ innerCandidates17_6.length then search edge 3 [71, 719] innerCandidates17_6 790 best else best) = best
  rw [if_pos (by decide), innerChild17_6]
  exact innerScan17_7

def innerTail17_5 : List Nat := 443 :: innerTail17_6
def innerCandidates17_5 : List Nat := [3833, 4673, 6389]
set_option maxRecDepth 4096 in
theorem innerCandidates17_5_checked : candidates edge 4 71 443 innerTail17_6 best.bound = innerCandidates17_5 := by
  unfold candidates
  have row : edge 443 = PrimePairCache.smallRow443.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild17_5 : search edge 3 [71, 443] innerCandidates17_5 514 best = best := by decide

theorem innerScan17_5 : scan edge (search edge 3) 4 [71] 71 innerTail17_5 best = best := by
  rw [innerTail17_5, scan, if_neg (by decide), innerCandidates17_5_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_6
    (if 3 ≤ innerCandidates17_5.length then search edge 3 [71, 443] innerCandidates17_5 514 best else best) = best
  rw [if_pos (by decide), innerChild17_5]
  exact innerScan17_6

def innerTail17_4 : List Nat := 389 :: innerTail17_5
def innerCandidates17_4 : List Nat := [947, 971, 8087]
set_option maxRecDepth 4096 in
theorem innerCandidates17_4_checked : candidates edge 4 71 389 innerTail17_5 best.bound = innerCandidates17_4 := by
  unfold candidates
  have row : edge 389 = PrimePairCache.smallRow389.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild17_4 : search edge 3 [71, 389] innerCandidates17_4 460 best = best := by decide

theorem innerScan17_4 : scan edge (search edge 3) 4 [71] 71 innerTail17_4 best = best := by
  rw [innerTail17_4, scan, if_neg (by decide), innerCandidates17_4_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_5
    (if 3 ≤ innerCandidates17_4.length then search edge 3 [71, 389] innerCandidates17_4 460 best else best) = best
  rw [if_pos (by decide), innerChild17_4]
  exact innerScan17_5

def innerTail17_3 : List Nat := 317 :: innerTail17_4
def innerCandidates17_3 : List Nat := [1709]
set_option maxRecDepth 4096 in
theorem innerCandidates17_3_checked : candidates edge 4 71 317 innerTail17_4 best.bound = innerCandidates17_3 := by
  unfold candidates
  have row : edge 317 = PrimePairCache.smallRow317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_3 : scan edge (search edge 3) 4 [71] 71 innerTail17_3 best = best := by
  rw [innerTail17_3, scan, if_neg (by decide), innerCandidates17_3_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_4
    (if 3 ≤ innerCandidates17_3.length then search edge 3 [71, 317] innerCandidates17_3 388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_4

def innerTail17_2 : List Nat := 263 :: innerTail17_3
def innerCandidates17_2 : List Nat := [443, 821, 1049, 2477, 8087]
set_option maxRecDepth 4096 in
theorem innerCandidates17_2_checked : candidates edge 4 71 263 innerTail17_3 best.bound = innerCandidates17_2 := by
  unfold candidates
  have row : edge 263 = PrimePairCache.smallRow263.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild17_2 : search edge 3 [71, 263] innerCandidates17_2 334 best = best := by decide

theorem innerScan17_2 : scan edge (search edge 3) 4 [71] 71 innerTail17_2 best = best := by
  rw [innerTail17_2, scan, if_neg (by decide), innerCandidates17_2_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_3
    (if 3 ≤ innerCandidates17_2.length then search edge 3 [71, 263] innerCandidates17_2 334 best else best) = best
  rw [if_pos (by decide), innerChild17_2]
  exact innerScan17_3

def innerTail17_1 : List Nat := 257 :: innerTail17_2
def innerCandidates17_1 : List Nat := [263, 3833]
set_option maxRecDepth 4096 in
theorem innerCandidates17_1_checked : candidates edge 4 71 257 innerTail17_2 best.bound = innerCandidates17_1 := by
  unfold candidates
  have row : edge 257 = PrimePairCache.smallRow257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan17_1 : scan edge (search edge 3) 4 [71] 71 innerTail17_1 best = best := by
  rw [innerTail17_1, scan, if_neg (by decide), innerCandidates17_1_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_2
    (if 3 ≤ innerCandidates17_1.length then search edge 3 [71, 257] innerCandidates17_1 328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan17_2

def innerTail17_0 : List Nat := 233 :: innerTail17_1
def innerCandidates17_0 : List Nat := [1163, 1583, 7001]
set_option maxRecDepth 4096 in
theorem innerCandidates17_0_checked : candidates edge 4 71 233 innerTail17_1 best.bound = innerCandidates17_0 := by
  unfold candidates
  have row : edge 233 = PrimePairCache.smallRow233.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild17_0 : search edge 3 [71, 233] innerCandidates17_0 304 best = best := by decide

theorem innerScan17_0 : scan edge (search edge 3) 4 [71] 71 innerTail17_0 best = best := by
  rw [innerTail17_0, scan, if_neg (by decide), innerCandidates17_0_checked]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_1
    (if 3 ≤ innerCandidates17_0.length then search edge 3 [71, 233] innerCandidates17_0 304 best else best) = best
  rw [if_pos (by decide), innerChild17_0]
  exact innerScan17_1

theorem rootBranch17 : search edge 4 [71] rootCandidates17 71 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [71] 71 innerTail17_0 best = best
  exact innerScan17_0

end ProjectEulerSolutions.PrimePairUpper.Search
