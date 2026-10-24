import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates8 : List Nat := [139, 151, 181, 859, 907, 991, 1009, 1123, 1153, 1237, 1279, 1447, 1951, 2029, 2089, 2281, 2311, 2383, 2857, 2887, 3037, 3109, 3301, 3307, 3331, 3343, 3517, 3571, 3583, 3637, 3853, 3889, 4003, 4021, 4159, 4243, 4261, 4339, 4357, 4441, 4513, 4549, 4723, 5011, 5179, 5281, 5527, 5569, 5701, 5779, 5851, 5881, 6037, 6067, 6073, 6271, 6343, 6571, 7333, 7411, 7459, 7591, 7741, 8161, 8191, 8287, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates8_checked : candidates edge 5 0 31 tail9 best.bound = rootCandidates8 := by
  unfold candidates
  have row : edge 31 = PrimePairCache.smallRow31.contains := by funext q; rfl
  rw [row]
  decide

def innerTail8_67 : List Nat := []
theorem innerScan8_67 : scan edge (search edge 3) 4 [31] 31 innerTail8_67 best = best := rfl

def innerTail8_66 : List Nat := 8377 :: innerTail8_67
def innerCandidates8_66 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates8_66_checked : candidates edge 4 31 8377 innerTail8_67 best.bound = innerCandidates8_66 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_66 : scan edge (search edge 3) 4 [31] 31 innerTail8_66 best = best := by
  rw [innerTail8_66, scan, if_neg (by decide), innerCandidates8_66_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_67
    (if 3 ≤ innerCandidates8_66.length then search edge 3 [31, 8377] innerCandidates8_66 8408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_67

def innerTail8_65 : List Nat := 8287 :: innerTail8_66
def innerCandidates8_65 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates8_65_checked : candidates edge 4 31 8287 innerTail8_66 best.bound = innerCandidates8_65 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_65 : scan edge (search edge 3) 4 [31] 31 innerTail8_65 best = best := by
  rw [innerTail8_65, scan, if_neg (by decide), innerCandidates8_65_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_66
    (if 3 ≤ innerCandidates8_65.length then search edge 3 [31, 8287] innerCandidates8_65 8318 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_66

def innerTail8_64 : List Nat := 8191 :: innerTail8_65
def innerCandidates8_64 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates8_64_checked : candidates edge 4 31 8191 innerTail8_65 best.bound = innerCandidates8_64 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_64 : scan edge (search edge 3) 4 [31] 31 innerTail8_64 best = best := by
  rw [innerTail8_64, scan, if_neg (by decide), innerCandidates8_64_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_65
    (if 3 ≤ innerCandidates8_64.length then search edge 3 [31, 8191] innerCandidates8_64 8222 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_65

def innerTail8_63 : List Nat := 8161 :: innerTail8_64
def innerCandidates8_63 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates8_63_checked : candidates edge 4 31 8161 innerTail8_64 best.bound = innerCandidates8_63 := by
  unfold candidates
  have row : edge 8161 = SearchData.row8161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_63 : scan edge (search edge 3) 4 [31] 31 innerTail8_63 best = best := by
  rw [innerTail8_63, scan, if_neg (by decide), innerCandidates8_63_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_64
    (if 3 ≤ innerCandidates8_63.length then search edge 3 [31, 8161] innerCandidates8_63 8192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_64

def innerTail8_62 : List Nat := 7741 :: innerTail8_63
def innerCandidates8_62 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates8_62_checked : candidates edge 4 31 7741 innerTail8_63 best.bound = innerCandidates8_62 := by
  unfold candidates
  have row : edge 7741 = SearchData.row7741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_62 : scan edge (search edge 3) 4 [31] 31 innerTail8_62 best = best := by
  rw [innerTail8_62, scan, if_neg (by decide), innerCandidates8_62_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_63
    (if 3 ≤ innerCandidates8_62.length then search edge 3 [31, 7741] innerCandidates8_62 7772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_63

def innerTail8_61 : List Nat := 7591 :: innerTail8_62
def innerCandidates8_61 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates8_61_checked : candidates edge 4 31 7591 innerTail8_62 best.bound = innerCandidates8_61 := by
  unfold candidates
  have row : edge 7591 = SearchData.row7591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_61 : scan edge (search edge 3) 4 [31] 31 innerTail8_61 best = best := by
  rw [innerTail8_61, scan, if_neg (by decide), innerCandidates8_61_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_62
    (if 3 ≤ innerCandidates8_61.length then search edge 3 [31, 7591] innerCandidates8_61 7622 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_62

def innerTail8_60 : List Nat := 7459 :: innerTail8_61
def innerCandidates8_60 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates8_60_checked : candidates edge 4 31 7459 innerTail8_61 best.bound = innerCandidates8_60 := by
  unfold candidates
  have row : edge 7459 = SearchData.row7459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_60 : scan edge (search edge 3) 4 [31] 31 innerTail8_60 best = best := by
  rw [innerTail8_60, scan, if_neg (by decide), innerCandidates8_60_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_61
    (if 3 ≤ innerCandidates8_60.length then search edge 3 [31, 7459] innerCandidates8_60 7490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_61

def innerTail8_59 : List Nat := 7411 :: innerTail8_60
def innerCandidates8_59 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates8_59_checked : candidates edge 4 31 7411 innerTail8_60 best.bound = innerCandidates8_59 := by
  unfold candidates
  have row : edge 7411 = SearchData.row7411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_59 : scan edge (search edge 3) 4 [31] 31 innerTail8_59 best = best := by
  rw [innerTail8_59, scan, if_neg (by decide), innerCandidates8_59_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_60
    (if 3 ≤ innerCandidates8_59.length then search edge 3 [31, 7411] innerCandidates8_59 7442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_60

def innerTail8_58 : List Nat := 7333 :: innerTail8_59
def innerCandidates8_58 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates8_58_checked : candidates edge 4 31 7333 innerTail8_59 best.bound = innerCandidates8_58 := by
  unfold candidates
  have row : edge 7333 = SearchData.row7333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_58 : scan edge (search edge 3) 4 [31] 31 innerTail8_58 best = best := by
  rw [innerTail8_58, scan, if_neg (by decide), innerCandidates8_58_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_59
    (if 3 ≤ innerCandidates8_58.length then search edge 3 [31, 7333] innerCandidates8_58 7364 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_59

def innerTail8_57 : List Nat := 6571 :: innerTail8_58
def innerCandidates8_57 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates8_57_checked : candidates edge 4 31 6571 innerTail8_58 best.bound = innerCandidates8_57 := by
  unfold candidates
  have row : edge 6571 = SearchData.row6571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_57 : scan edge (search edge 3) 4 [31] 31 innerTail8_57 best = best := by
  rw [innerTail8_57, scan, if_neg (by decide), innerCandidates8_57_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_58
    (if 3 ≤ innerCandidates8_57.length then search edge 3 [31, 6571] innerCandidates8_57 6602 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_58

def innerTail8_56 : List Nat := 6343 :: innerTail8_57
def innerCandidates8_56 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates8_56_checked : candidates edge 4 31 6343 innerTail8_57 best.bound = innerCandidates8_56 := by
  unfold candidates
  have row : edge 6343 = SearchData.row6343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_56 : scan edge (search edge 3) 4 [31] 31 innerTail8_56 best = best := by
  rw [innerTail8_56, scan, if_neg (by decide), innerCandidates8_56_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_57
    (if 3 ≤ innerCandidates8_56.length then search edge 3 [31, 6343] innerCandidates8_56 6374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_57

def innerTail8_55 : List Nat := 6271 :: innerTail8_56
def innerCandidates8_55 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates8_55_checked : candidates edge 4 31 6271 innerTail8_56 best.bound = innerCandidates8_55 := by
  unfold candidates
  have row : edge 6271 = SearchData.row6271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_55 : scan edge (search edge 3) 4 [31] 31 innerTail8_55 best = best := by
  rw [innerTail8_55, scan, if_neg (by decide), innerCandidates8_55_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_56
    (if 3 ≤ innerCandidates8_55.length then search edge 3 [31, 6271] innerCandidates8_55 6302 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_56

def innerTail8_54 : List Nat := 6073 :: innerTail8_55
def innerCandidates8_54 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates8_54_checked : candidates edge 4 31 6073 innerTail8_55 best.bound = innerCandidates8_54 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_54 : scan edge (search edge 3) 4 [31] 31 innerTail8_54 best = best := by
  rw [innerTail8_54, scan, if_neg (by decide), innerCandidates8_54_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_55
    (if 3 ≤ innerCandidates8_54.length then search edge 3 [31, 6073] innerCandidates8_54 6104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_55

def innerTail8_53 : List Nat := 6067 :: innerTail8_54
def innerCandidates8_53 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates8_53_checked : candidates edge 4 31 6067 innerTail8_54 best.bound = innerCandidates8_53 := by
  unfold candidates
  have row : edge 6067 = SearchData.row6067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_53 : scan edge (search edge 3) 4 [31] 31 innerTail8_53 best = best := by
  rw [innerTail8_53, scan, if_neg (by decide), innerCandidates8_53_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_54
    (if 3 ≤ innerCandidates8_53.length then search edge 3 [31, 6067] innerCandidates8_53 6098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_54

def innerTail8_52 : List Nat := 6037 :: innerTail8_53
def innerCandidates8_52 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates8_52_checked : candidates edge 4 31 6037 innerTail8_53 best.bound = innerCandidates8_52 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_52 : scan edge (search edge 3) 4 [31] 31 innerTail8_52 best = best := by
  rw [innerTail8_52, scan, if_neg (by decide), innerCandidates8_52_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_53
    (if 3 ≤ innerCandidates8_52.length then search edge 3 [31, 6037] innerCandidates8_52 6068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_53

def innerTail8_51 : List Nat := 5881 :: innerTail8_52
def innerCandidates8_51 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates8_51_checked : candidates edge 4 31 5881 innerTail8_52 best.bound = innerCandidates8_51 := by
  unfold candidates
  have row : edge 5881 = SearchData.row5881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_51 : scan edge (search edge 3) 4 [31] 31 innerTail8_51 best = best := by
  rw [innerTail8_51, scan, if_neg (by decide), innerCandidates8_51_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_52
    (if 3 ≤ innerCandidates8_51.length then search edge 3 [31, 5881] innerCandidates8_51 5912 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_52

def innerTail8_50 : List Nat := 5851 :: innerTail8_51
def innerCandidates8_50 : List Nat := [8377]
set_option maxRecDepth 4096 in
theorem innerCandidates8_50_checked : candidates edge 4 31 5851 innerTail8_51 best.bound = innerCandidates8_50 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_50 : scan edge (search edge 3) 4 [31] 31 innerTail8_50 best = best := by
  rw [innerTail8_50, scan, if_neg (by decide), innerCandidates8_50_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_51
    (if 3 ≤ innerCandidates8_50.length then search edge 3 [31, 5851] innerCandidates8_50 5882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_51

def innerTail8_49 : List Nat := 5779 :: innerTail8_50
def innerCandidates8_49 : List Nat := [7591]
set_option maxRecDepth 4096 in
theorem innerCandidates8_49_checked : candidates edge 4 31 5779 innerTail8_50 best.bound = innerCandidates8_49 := by
  unfold candidates
  have row : edge 5779 = SearchData.row5779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_49 : scan edge (search edge 3) 4 [31] 31 innerTail8_49 best = best := by
  rw [innerTail8_49, scan, if_neg (by decide), innerCandidates8_49_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_50
    (if 3 ≤ innerCandidates8_49.length then search edge 3 [31, 5779] innerCandidates8_49 5810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_50

def innerTail8_48 : List Nat := 5701 :: innerTail8_49
def innerCandidates8_48 : List Nat := [7459]
set_option maxRecDepth 4096 in
theorem innerCandidates8_48_checked : candidates edge 4 31 5701 innerTail8_49 best.bound = innerCandidates8_48 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_48 : scan edge (search edge 3) 4 [31] 31 innerTail8_48 best = best := by
  rw [innerTail8_48, scan, if_neg (by decide), innerCandidates8_48_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_49
    (if 3 ≤ innerCandidates8_48.length then search edge 3 [31, 5701] innerCandidates8_48 5732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_49

def innerTail8_47 : List Nat := 5569 :: innerTail8_48
def innerCandidates8_47 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates8_47_checked : candidates edge 4 31 5569 innerTail8_48 best.bound = innerCandidates8_47 := by
  unfold candidates
  have row : edge 5569 = SearchData.row5569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_47 : scan edge (search edge 3) 4 [31] 31 innerTail8_47 best = best := by
  rw [innerTail8_47, scan, if_neg (by decide), innerCandidates8_47_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_48
    (if 3 ≤ innerCandidates8_47.length then search edge 3 [31, 5569] innerCandidates8_47 5600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_48

def innerTail8_46 : List Nat := 5527 :: innerTail8_47
def innerCandidates8_46 : List Nat := [6343]
set_option maxRecDepth 4096 in
theorem innerCandidates8_46_checked : candidates edge 4 31 5527 innerTail8_47 best.bound = innerCandidates8_46 := by
  unfold candidates
  have row : edge 5527 = SearchData.row5527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_46 : scan edge (search edge 3) 4 [31] 31 innerTail8_46 best = best := by
  rw [innerTail8_46, scan, if_neg (by decide), innerCandidates8_46_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_47
    (if 3 ≤ innerCandidates8_46.length then search edge 3 [31, 5527] innerCandidates8_46 5558 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_47

def innerTail8_45 : List Nat := 5281 :: innerTail8_46
def innerCandidates8_45 : List Nat := [8287]
set_option maxRecDepth 4096 in
theorem innerCandidates8_45_checked : candidates edge 4 31 5281 innerTail8_46 best.bound = innerCandidates8_45 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_45 : scan edge (search edge 3) 4 [31] 31 innerTail8_45 best = best := by
  rw [innerTail8_45, scan, if_neg (by decide), innerCandidates8_45_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_46
    (if 3 ≤ innerCandidates8_45.length then search edge 3 [31, 5281] innerCandidates8_45 5312 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_46

def innerTail8_44 : List Nat := 5179 :: innerTail8_45
def innerCandidates8_44 : List Nat := [5851]
set_option maxRecDepth 4096 in
theorem innerCandidates8_44_checked : candidates edge 4 31 5179 innerTail8_45 best.bound = innerCandidates8_44 := by
  unfold candidates
  have row : edge 5179 = SearchData.row5179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_44 : scan edge (search edge 3) 4 [31] 31 innerTail8_44 best = best := by
  rw [innerTail8_44, scan, if_neg (by decide), innerCandidates8_44_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_45
    (if 3 ≤ innerCandidates8_44.length then search edge 3 [31, 5179] innerCandidates8_44 5210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_45

def innerTail8_43 : List Nat := 5011 :: innerTail8_44
def innerCandidates8_43 : List Nat := [6271, 8191]
set_option maxRecDepth 4096 in
theorem innerCandidates8_43_checked : candidates edge 4 31 5011 innerTail8_44 best.bound = innerCandidates8_43 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_43 : scan edge (search edge 3) 4 [31] 31 innerTail8_43 best = best := by
  rw [innerTail8_43, scan, if_neg (by decide), innerCandidates8_43_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_44
    (if 3 ≤ innerCandidates8_43.length then search edge 3 [31, 5011] innerCandidates8_43 5042 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_44

def innerTail8_42 : List Nat := 4723 :: innerTail8_43
def innerCandidates8_42 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates8_42_checked : candidates edge 4 31 4723 innerTail8_43 best.bound = innerCandidates8_42 := by
  unfold candidates
  have row : edge 4723 = SearchData.row4723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_42 : scan edge (search edge 3) 4 [31] 31 innerTail8_42 best = best := by
  rw [innerTail8_42, scan, if_neg (by decide), innerCandidates8_42_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_43
    (if 3 ≤ innerCandidates8_42.length then search edge 3 [31, 4723] innerCandidates8_42 4754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_43

def innerTail8_41 : List Nat := 4549 :: innerTail8_42
def innerCandidates8_41 : List Nat := [5851, 6343, 6571, 8287]
set_option maxRecDepth 4096 in
theorem innerCandidates8_41_checked : candidates edge 4 31 4549 innerTail8_42 best.bound = innerCandidates8_41 := by
  unfold candidates
  have row : edge 4549 = SearchData.row4549.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild8_41 : search edge 3 [31, 4549] innerCandidates8_41 4580 best = best := by decide

theorem innerScan8_41 : scan edge (search edge 3) 4 [31] 31 innerTail8_41 best = best := by
  rw [innerTail8_41, scan, if_neg (by decide), innerCandidates8_41_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_42
    (if 3 ≤ innerCandidates8_41.length then search edge 3 [31, 4549] innerCandidates8_41 4580 best else best) = best
  rw [if_pos (by decide), innerChild8_41]
  exact innerScan8_42

def innerTail8_40 : List Nat := 4513 :: innerTail8_41
def innerCandidates8_40 : List Nat := [8287]
set_option maxRecDepth 4096 in
theorem innerCandidates8_40_checked : candidates edge 4 31 4513 innerTail8_41 best.bound = innerCandidates8_40 := by
  unfold candidates
  have row : edge 4513 = SearchData.row4513.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_40 : scan edge (search edge 3) 4 [31] 31 innerTail8_40 best = best := by
  rw [innerTail8_40, scan, if_neg (by decide), innerCandidates8_40_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_41
    (if 3 ≤ innerCandidates8_40.length then search edge 3 [31, 4513] innerCandidates8_40 4544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_41

def innerTail8_39 : List Nat := 4441 :: innerTail8_40
def innerCandidates8_39 : List Nat := [6271]
set_option maxRecDepth 4096 in
theorem innerCandidates8_39_checked : candidates edge 4 31 4441 innerTail8_40 best.bound = innerCandidates8_39 := by
  unfold candidates
  have row : edge 4441 = SearchData.row4441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_39 : scan edge (search edge 3) 4 [31] 31 innerTail8_39 best = best := by
  rw [innerTail8_39, scan, if_neg (by decide), innerCandidates8_39_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_40
    (if 3 ≤ innerCandidates8_39.length then search edge 3 [31, 4441] innerCandidates8_39 4472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_40

def innerTail8_38 : List Nat := 4357 :: innerTail8_39
def innerCandidates8_38 : List Nat := [7411]
set_option maxRecDepth 4096 in
theorem innerCandidates8_38_checked : candidates edge 4 31 4357 innerTail8_39 best.bound = innerCandidates8_38 := by
  unfold candidates
  have row : edge 4357 = SearchData.row4357.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_38 : scan edge (search edge 3) 4 [31] 31 innerTail8_38 best = best := by
  rw [innerTail8_38, scan, if_neg (by decide), innerCandidates8_38_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_39
    (if 3 ≤ innerCandidates8_38.length then search edge 3 [31, 4357] innerCandidates8_38 4388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_39

def innerTail8_37 : List Nat := 4339 :: innerTail8_38
def innerCandidates8_37 : List Nat := [5701]
set_option maxRecDepth 4096 in
theorem innerCandidates8_37_checked : candidates edge 4 31 4339 innerTail8_38 best.bound = innerCandidates8_37 := by
  unfold candidates
  have row : edge 4339 = SearchData.row4339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_37 : scan edge (search edge 3) 4 [31] 31 innerTail8_37 best = best := by
  rw [innerTail8_37, scan, if_neg (by decide), innerCandidates8_37_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_38
    (if 3 ≤ innerCandidates8_37.length then search edge 3 [31, 4339] innerCandidates8_37 4370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_38

def innerTail8_36 : List Nat := 4261 :: innerTail8_37
def innerCandidates8_36 : List Nat := [5011, 5281]
set_option maxRecDepth 4096 in
theorem innerCandidates8_36_checked : candidates edge 4 31 4261 innerTail8_37 best.bound = innerCandidates8_36 := by
  unfold candidates
  have row : edge 4261 = SearchData.row4261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_36 : scan edge (search edge 3) 4 [31] 31 innerTail8_36 best = best := by
  rw [innerTail8_36, scan, if_neg (by decide), innerCandidates8_36_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_37
    (if 3 ≤ innerCandidates8_36.length then search edge 3 [31, 4261] innerCandidates8_36 4292 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_37

def innerTail8_35 : List Nat := 4243 :: innerTail8_36
def innerCandidates8_35 : List Nat := [4339, 5527]
set_option maxRecDepth 4096 in
theorem innerCandidates8_35_checked : candidates edge 4 31 4243 innerTail8_36 best.bound = innerCandidates8_35 := by
  unfold candidates
  have row : edge 4243 = SearchData.row4243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_35 : scan edge (search edge 3) 4 [31] 31 innerTail8_35 best = best := by
  rw [innerTail8_35, scan, if_neg (by decide), innerCandidates8_35_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_36
    (if 3 ≤ innerCandidates8_35.length then search edge 3 [31, 4243] innerCandidates8_35 4274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_36

def innerTail8_34 : List Nat := 4159 :: innerTail8_35
def innerCandidates8_34 : List Nat := [5011, 5779]
set_option maxRecDepth 4096 in
theorem innerCandidates8_34_checked : candidates edge 4 31 4159 innerTail8_35 best.bound = innerCandidates8_34 := by
  unfold candidates
  have row : edge 4159 = SearchData.row4159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_34 : scan edge (search edge 3) 4 [31] 31 innerTail8_34 best = best := by
  rw [innerTail8_34, scan, if_neg (by decide), innerCandidates8_34_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_35
    (if 3 ≤ innerCandidates8_34.length then search edge 3 [31, 4159] innerCandidates8_34 4190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_35

def innerTail8_33 : List Nat := 4021 :: innerTail8_34
def innerCandidates8_33 : List Nat := [4441]
set_option maxRecDepth 4096 in
theorem innerCandidates8_33_checked : candidates edge 4 31 4021 innerTail8_34 best.bound = innerCandidates8_33 := by
  unfold candidates
  have row : edge 4021 = SearchData.row4021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_33 : scan edge (search edge 3) 4 [31] 31 innerTail8_33 best = best := by
  rw [innerTail8_33, scan, if_neg (by decide), innerCandidates8_33_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_34
    (if 3 ≤ innerCandidates8_33.length then search edge 3 [31, 4021] innerCandidates8_33 4052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_34

def innerTail8_32 : List Nat := 4003 :: innerTail8_33
def innerCandidates8_32 : List Nat := [4723, 5851]
set_option maxRecDepth 4096 in
theorem innerCandidates8_32_checked : candidates edge 4 31 4003 innerTail8_33 best.bound = innerCandidates8_32 := by
  unfold candidates
  have row : edge 4003 = SearchData.row4003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_32 : scan edge (search edge 3) 4 [31] 31 innerTail8_32 best = best := by
  rw [innerTail8_32, scan, if_neg (by decide), innerCandidates8_32_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_33
    (if 3 ≤ innerCandidates8_32.length then search edge 3 [31, 4003] innerCandidates8_32 4034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_33

def innerTail8_31 : List Nat := 3889 :: innerTail8_32
def innerCandidates8_31 : List Nat := [4021]
set_option maxRecDepth 4096 in
theorem innerCandidates8_31_checked : candidates edge 4 31 3889 innerTail8_32 best.bound = innerCandidates8_31 := by
  unfold candidates
  have row : edge 3889 = SearchData.row3889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_31 : scan edge (search edge 3) 4 [31] 31 innerTail8_31 best = best := by
  rw [innerTail8_31, scan, if_neg (by decide), innerCandidates8_31_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_32
    (if 3 ≤ innerCandidates8_31.length then search edge 3 [31, 3889] innerCandidates8_31 3920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_32

def innerTail8_30 : List Nat := 3853 :: innerTail8_31
def innerCandidates8_30 : List Nat := [7591]
set_option maxRecDepth 4096 in
theorem innerCandidates8_30_checked : candidates edge 4 31 3853 innerTail8_31 best.bound = innerCandidates8_30 := by
  unfold candidates
  have row : edge 3853 = SearchData.row3853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_30 : scan edge (search edge 3) 4 [31] 31 innerTail8_30 best = best := by
  rw [innerTail8_30, scan, if_neg (by decide), innerCandidates8_30_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_31
    (if 3 ≤ innerCandidates8_30.length then search edge 3 [31, 3853] innerCandidates8_30 3884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_31

def innerTail8_29 : List Nat := 3637 :: innerTail8_30
def innerCandidates8_29 : List Nat := [6343]
set_option maxRecDepth 4096 in
theorem innerCandidates8_29_checked : candidates edge 4 31 3637 innerTail8_30 best.bound = innerCandidates8_29 := by
  unfold candidates
  have row : edge 3637 = SearchData.row3637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_29 : scan edge (search edge 3) 4 [31] 31 innerTail8_29 best = best := by
  rw [innerTail8_29, scan, if_neg (by decide), innerCandidates8_29_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_30
    (if 3 ≤ innerCandidates8_29.length then search edge 3 [31, 3637] innerCandidates8_29 3668 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_30

def innerTail8_28 : List Nat := 3583 :: innerTail8_29
def innerCandidates8_28 : List Nat := [5179, 6067, 8191]
set_option maxRecDepth 4096 in
theorem innerCandidates8_28_checked : candidates edge 4 31 3583 innerTail8_29 best.bound = innerCandidates8_28 := by
  unfold candidates
  have row : edge 3583 = SearchData.row3583.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild8_28 : search edge 3 [31, 3583] innerCandidates8_28 3614 best = best := by decide

theorem innerScan8_28 : scan edge (search edge 3) 4 [31] 31 innerTail8_28 best = best := by
  rw [innerTail8_28, scan, if_neg (by decide), innerCandidates8_28_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_29
    (if 3 ≤ innerCandidates8_28.length then search edge 3 [31, 3583] innerCandidates8_28 3614 best else best) = best
  rw [if_pos (by decide), innerChild8_28]
  exact innerScan8_29

def innerTail8_27 : List Nat := 3571 :: innerTail8_28
def innerCandidates8_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates8_27_checked : candidates edge 4 31 3571 innerTail8_28 best.bound = innerCandidates8_27 := by
  unfold candidates
  have row : edge 3571 = SearchData.row3571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_27 : scan edge (search edge 3) 4 [31] 31 innerTail8_27 best = best := by
  rw [innerTail8_27, scan, if_neg (by decide), innerCandidates8_27_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_28
    (if 3 ≤ innerCandidates8_27.length then search edge 3 [31, 3571] innerCandidates8_27 3602 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_28

def innerTail8_26 : List Nat := 3517 :: innerTail8_27
def innerCandidates8_26 : List Nat := [8161]
set_option maxRecDepth 4096 in
theorem innerCandidates8_26_checked : candidates edge 4 31 3517 innerTail8_27 best.bound = innerCandidates8_26 := by
  unfold candidates
  have row : edge 3517 = SearchData.row3517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_26 : scan edge (search edge 3) 4 [31] 31 innerTail8_26 best = best := by
  rw [innerTail8_26, scan, if_neg (by decide), innerCandidates8_26_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_27
    (if 3 ≤ innerCandidates8_26.length then search edge 3 [31, 3517] innerCandidates8_26 3548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_27

def innerTail8_25 : List Nat := 3343 :: innerTail8_26
def innerCandidates8_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates8_25_checked : candidates edge 4 31 3343 innerTail8_26 best.bound = innerCandidates8_25 := by
  unfold candidates
  have row : edge 3343 = SearchData.row3343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_25 : scan edge (search edge 3) 4 [31] 31 innerTail8_25 best = best := by
  rw [innerTail8_25, scan, if_neg (by decide), innerCandidates8_25_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_26
    (if 3 ≤ innerCandidates8_25.length then search edge 3 [31, 3343] innerCandidates8_25 3374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_26

def innerTail8_24 : List Nat := 3331 :: innerTail8_25
def innerCandidates8_24 : List Nat := [4261, 5701]
set_option maxRecDepth 4096 in
theorem innerCandidates8_24_checked : candidates edge 4 31 3331 innerTail8_25 best.bound = innerCandidates8_24 := by
  unfold candidates
  have row : edge 3331 = SearchData.row3331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_24 : scan edge (search edge 3) 4 [31] 31 innerTail8_24 best = best := by
  rw [innerTail8_24, scan, if_neg (by decide), innerCandidates8_24_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_25
    (if 3 ≤ innerCandidates8_24.length then search edge 3 [31, 3331] innerCandidates8_24 3362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_25

def innerTail8_23 : List Nat := 3307 :: innerTail8_24
def innerCandidates8_23 : List Nat := [3331, 3583, 7411]
set_option maxRecDepth 4096 in
theorem innerCandidates8_23_checked : candidates edge 4 31 3307 innerTail8_24 best.bound = innerCandidates8_23 := by
  unfold candidates
  have row : edge 3307 = SearchData.row3307.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild8_23 : search edge 3 [31, 3307] innerCandidates8_23 3338 best = best := by decide

theorem innerScan8_23 : scan edge (search edge 3) 4 [31] 31 innerTail8_23 best = best := by
  rw [innerTail8_23, scan, if_neg (by decide), innerCandidates8_23_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_24
    (if 3 ≤ innerCandidates8_23.length then search edge 3 [31, 3307] innerCandidates8_23 3338 best else best) = best
  rw [if_pos (by decide), innerChild8_23]
  exact innerScan8_24

def innerTail8_22 : List Nat := 3301 :: innerTail8_23
def innerCandidates8_22 : List Nat := [3517, 5779, 5881, 6343]
set_option maxRecDepth 4096 in
theorem innerCandidates8_22_checked : candidates edge 4 31 3301 innerTail8_23 best.bound = innerCandidates8_22 := by
  unfold candidates
  have row : edge 3301 = SearchData.row3301.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild8_22 : search edge 3 [31, 3301] innerCandidates8_22 3332 best = best := by decide

theorem innerScan8_22 : scan edge (search edge 3) 4 [31] 31 innerTail8_22 best = best := by
  rw [innerTail8_22, scan, if_neg (by decide), innerCandidates8_22_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_23
    (if 3 ≤ innerCandidates8_22.length then search edge 3 [31, 3301] innerCandidates8_22 3332 best else best) = best
  rw [if_pos (by decide), innerChild8_22]
  exact innerScan8_23

def innerTail8_21 : List Nat := 3109 :: innerTail8_22
def innerCandidates8_21 : List Nat := [4339, 6073]
set_option maxRecDepth 4096 in
theorem innerCandidates8_21_checked : candidates edge 4 31 3109 innerTail8_22 best.bound = innerCandidates8_21 := by
  unfold candidates
  have row : edge 3109 = SearchData.row3109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_21 : scan edge (search edge 3) 4 [31] 31 innerTail8_21 best = best := by
  rw [innerTail8_21, scan, if_neg (by decide), innerCandidates8_21_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_22
    (if 3 ≤ innerCandidates8_21.length then search edge 3 [31, 3109] innerCandidates8_21 3140 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_22

def innerTail8_20 : List Nat := 3037 :: innerTail8_21
def innerCandidates8_20 : List Nat := [3331, 3517, 5701]
set_option maxRecDepth 4096 in
theorem innerCandidates8_20_checked : candidates edge 4 31 3037 innerTail8_21 best.bound = innerCandidates8_20 := by
  unfold candidates
  have row : edge 3037 = SearchData.row3037.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild8_20 : search edge 3 [31, 3037] innerCandidates8_20 3068 best = best := by decide

theorem innerScan8_20 : scan edge (search edge 3) 4 [31] 31 innerTail8_20 best = best := by
  rw [innerTail8_20, scan, if_neg (by decide), innerCandidates8_20_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_21
    (if 3 ≤ innerCandidates8_20.length then search edge 3 [31, 3037] innerCandidates8_20 3068 best else best) = best
  rw [if_pos (by decide), innerChild8_20]
  exact innerScan8_21

def innerTail8_19 : List Nat := 2887 :: innerTail8_20
def innerCandidates8_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates8_19_checked : candidates edge 4 31 2887 innerTail8_20 best.bound = innerCandidates8_19 := by
  unfold candidates
  have row : edge 2887 = SearchData.row2887.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_19 : scan edge (search edge 3) 4 [31] 31 innerTail8_19 best = best := by
  rw [innerTail8_19, scan, if_neg (by decide), innerCandidates8_19_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_20
    (if 3 ≤ innerCandidates8_19.length then search edge 3 [31, 2887] innerCandidates8_19 2918 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_20

def innerTail8_18 : List Nat := 2857 :: innerTail8_19
def innerCandidates8_18 : List Nat := [4549, 5881]
set_option maxRecDepth 4096 in
theorem innerCandidates8_18_checked : candidates edge 4 31 2857 innerTail8_19 best.bound = innerCandidates8_18 := by
  unfold candidates
  have row : edge 2857 = SearchData.row2857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_18 : scan edge (search edge 3) 4 [31] 31 innerTail8_18 best = best := by
  rw [innerTail8_18, scan, if_neg (by decide), innerCandidates8_18_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_19
    (if 3 ≤ innerCandidates8_18.length then search edge 3 [31, 2857] innerCandidates8_18 2888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_19

def innerTail8_17 : List Nat := 2383 :: innerTail8_18
def innerCandidates8_17 : List Nat := [3583, 4021, 5527]
set_option maxRecDepth 4096 in
theorem innerCandidates8_17_checked : candidates edge 4 31 2383 innerTail8_18 best.bound = innerCandidates8_17 := by
  unfold candidates
  have row : edge 2383 = PrimePairCache.smallRow2383.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild8_17 : search edge 3 [31, 2383] innerCandidates8_17 2414 best = best := by decide

theorem innerScan8_17 : scan edge (search edge 3) 4 [31] 31 innerTail8_17 best = best := by
  rw [innerTail8_17, scan, if_neg (by decide), innerCandidates8_17_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_18
    (if 3 ≤ innerCandidates8_17.length then search edge 3 [31, 2383] innerCandidates8_17 2414 best else best) = best
  rw [if_pos (by decide), innerChild8_17]
  exact innerScan8_18

def innerTail8_16 : List Nat := 2311 :: innerTail8_17
def innerCandidates8_16 : List Nat := [4549, 5179, 8287]
set_option maxRecDepth 4096 in
theorem innerCandidates8_16_checked : candidates edge 4 31 2311 innerTail8_17 best.bound = innerCandidates8_16 := by
  unfold candidates
  have row : edge 2311 = PrimePairCache.smallRow2311.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild8_16 : search edge 3 [31, 2311] innerCandidates8_16 2342 best = best := by decide

theorem innerScan8_16 : scan edge (search edge 3) 4 [31] 31 innerTail8_16 best = best := by
  rw [innerTail8_16, scan, if_neg (by decide), innerCandidates8_16_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_17
    (if 3 ≤ innerCandidates8_16.length then search edge 3 [31, 2311] innerCandidates8_16 2342 best else best) = best
  rw [if_pos (by decide), innerChild8_16]
  exact innerScan8_17

def innerTail8_15 : List Nat := 2281 :: innerTail8_16
def innerCandidates8_15 : List Nat := [2887]
set_option maxRecDepth 4096 in
theorem innerCandidates8_15_checked : candidates edge 4 31 2281 innerTail8_16 best.bound = innerCandidates8_15 := by
  unfold candidates
  have row : edge 2281 = PrimePairCache.smallRow2281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_15 : scan edge (search edge 3) 4 [31] 31 innerTail8_15 best = best := by
  rw [innerTail8_15, scan, if_neg (by decide), innerCandidates8_15_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_16
    (if 3 ≤ innerCandidates8_15.length then search edge 3 [31, 2281] innerCandidates8_15 2312 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_16

def innerTail8_14 : List Nat := 2089 :: innerTail8_15
def innerCandidates8_14 : List Nat := [7411]
set_option maxRecDepth 4096 in
theorem innerCandidates8_14_checked : candidates edge 4 31 2089 innerTail8_15 best.bound = innerCandidates8_14 := by
  unfold candidates
  have row : edge 2089 = PrimePairCache.smallRow2089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_14 : scan edge (search edge 3) 4 [31] 31 innerTail8_14 best = best := by
  rw [innerTail8_14, scan, if_neg (by decide), innerCandidates8_14_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_15
    (if 3 ≤ innerCandidates8_14.length then search edge 3 [31, 2089] innerCandidates8_14 2120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_15

def innerTail8_13 : List Nat := 2029 :: innerTail8_14
def innerCandidates8_13 : List Nat := [2281, 5281, 5701]
set_option maxRecDepth 4096 in
theorem innerCandidates8_13_checked : candidates edge 4 31 2029 innerTail8_14 best.bound = innerCandidates8_13 := by
  unfold candidates
  have row : edge 2029 = PrimePairCache.smallRow2029.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild8_13 : search edge 3 [31, 2029] innerCandidates8_13 2060 best = best := by decide

theorem innerScan8_13 : scan edge (search edge 3) 4 [31] 31 innerTail8_13 best = best := by
  rw [innerTail8_13, scan, if_neg (by decide), innerCandidates8_13_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_14
    (if 3 ≤ innerCandidates8_13.length then search edge 3 [31, 2029] innerCandidates8_13 2060 best else best) = best
  rw [if_pos (by decide), innerChild8_13]
  exact innerScan8_14

def innerTail8_12 : List Nat := 1951 :: innerTail8_13
def innerCandidates8_12 : List Nat := [4549, 6037]
set_option maxRecDepth 4096 in
theorem innerCandidates8_12_checked : candidates edge 4 31 1951 innerTail8_13 best.bound = innerCandidates8_12 := by
  unfold candidates
  have row : edge 1951 = PrimePairCache.smallRow1951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_12 : scan edge (search edge 3) 4 [31] 31 innerTail8_12 best = best := by
  rw [innerTail8_12, scan, if_neg (by decide), innerCandidates8_12_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_13
    (if 3 ≤ innerCandidates8_12.length then search edge 3 [31, 1951] innerCandidates8_12 1982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_13

def innerTail8_11 : List Nat := 1447 :: innerTail8_12
def innerCandidates8_11 : List Nat := [4441]
set_option maxRecDepth 4096 in
theorem innerCandidates8_11_checked : candidates edge 4 31 1447 innerTail8_12 best.bound = innerCandidates8_11 := by
  unfold candidates
  have row : edge 1447 = PrimePairCache.smallRow1447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_11 : scan edge (search edge 3) 4 [31] 31 innerTail8_11 best = best := by
  rw [innerTail8_11, scan, if_neg (by decide), innerCandidates8_11_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_12
    (if 3 ≤ innerCandidates8_11.length then search edge 3 [31, 1447] innerCandidates8_11 1478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_12

def innerTail8_10 : List Nat := 1279 :: innerTail8_11
def innerCandidates8_10 : List Nat := [5179, 6037, 6271]
set_option maxRecDepth 4096 in
theorem innerCandidates8_10_checked : candidates edge 4 31 1279 innerTail8_11 best.bound = innerCandidates8_10 := by
  unfold candidates
  have row : edge 1279 = PrimePairCache.smallRow1279.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild8_10 : search edge 3 [31, 1279] innerCandidates8_10 1310 best = best := by decide

theorem innerScan8_10 : scan edge (search edge 3) 4 [31] 31 innerTail8_10 best = best := by
  rw [innerTail8_10, scan, if_neg (by decide), innerCandidates8_10_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_11
    (if 3 ≤ innerCandidates8_10.length then search edge 3 [31, 1279] innerCandidates8_10 1310 best else best) = best
  rw [if_pos (by decide), innerChild8_10]
  exact innerScan8_11

def innerTail8_9 : List Nat := 1237 :: innerTail8_10
def innerCandidates8_9 : List Nat := [3307, 6271, 6571]
set_option maxRecDepth 4096 in
theorem innerCandidates8_9_checked : candidates edge 4 31 1237 innerTail8_10 best.bound = innerCandidates8_9 := by
  unfold candidates
  have row : edge 1237 = PrimePairCache.smallRow1237.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild8_9 : search edge 3 [31, 1237] innerCandidates8_9 1268 best = best := by decide

theorem innerScan8_9 : scan edge (search edge 3) 4 [31] 31 innerTail8_9 best = best := by
  rw [innerTail8_9, scan, if_neg (by decide), innerCandidates8_9_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_10
    (if 3 ≤ innerCandidates8_9.length then search edge 3 [31, 1237] innerCandidates8_9 1268 best else best) = best
  rw [if_pos (by decide), innerChild8_9]
  exact innerScan8_10

def innerTail8_8 : List Nat := 1153 :: innerTail8_9
def innerCandidates8_8 : List Nat := [2383, 3853, 5179, 6037, 6067, 7741]
set_option maxRecDepth 4096 in
theorem innerCandidates8_8_checked : candidates edge 4 31 1153 innerTail8_9 best.bound = innerCandidates8_8 := by
  unfold candidates
  have row : edge 1153 = PrimePairCache.smallRow1153.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild8_8 : search edge 3 [31, 1153] innerCandidates8_8 1184 best = best := by decide

theorem innerScan8_8 : scan edge (search edge 3) 4 [31] 31 innerTail8_8 best = best := by
  rw [innerTail8_8, scan, if_neg (by decide), innerCandidates8_8_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_9
    (if 3 ≤ innerCandidates8_8.length then search edge 3 [31, 1153] innerCandidates8_8 1184 best else best) = best
  rw [if_pos (by decide), innerChild8_8]
  exact innerScan8_9

def innerTail8_7 : List Nat := 1123 :: innerTail8_8
def innerCandidates8_7 : List Nat := [2029, 2281, 5281, 5527, 6037]
set_option maxRecDepth 4096 in
theorem innerCandidates8_7_checked : candidates edge 4 31 1123 innerTail8_8 best.bound = innerCandidates8_7 := by
  unfold candidates
  have row : edge 1123 = PrimePairCache.smallRow1123.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild8_7 : search edge 3 [31, 1123] innerCandidates8_7 1154 best = best := by decide

theorem innerScan8_7 : scan edge (search edge 3) 4 [31] 31 innerTail8_7 best = best := by
  rw [innerTail8_7, scan, if_neg (by decide), innerCandidates8_7_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_8
    (if 3 ≤ innerCandidates8_7.length then search edge 3 [31, 1123] innerCandidates8_7 1154 best else best) = best
  rw [if_pos (by decide), innerChild8_7]
  exact innerScan8_8

def innerTail8_6 : List Nat := 1009 :: innerTail8_7
def innerCandidates8_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates8_6_checked : candidates edge 4 31 1009 innerTail8_7 best.bound = innerCandidates8_6 := by
  unfold candidates
  have row : edge 1009 = PrimePairCache.smallRow1009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_6 : scan edge (search edge 3) 4 [31] 31 innerTail8_6 best = best := by
  rw [innerTail8_6, scan, if_neg (by decide), innerCandidates8_6_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_7
    (if 3 ≤ innerCandidates8_6.length then search edge 3 [31, 1009] innerCandidates8_6 1040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_7

def innerTail8_5 : List Nat := 991 :: innerTail8_6
def innerCandidates8_5 : List Nat := [1009, 2029, 3571, 5881]
set_option maxRecDepth 4096 in
theorem innerCandidates8_5_checked : candidates edge 4 31 991 innerTail8_6 best.bound = innerCandidates8_5 := by
  unfold candidates
  have row : edge 991 = PrimePairCache.smallRow991.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild8_5 : search edge 3 [31, 991] innerCandidates8_5 1022 best = best := by decide

theorem innerScan8_5 : scan edge (search edge 3) 4 [31] 31 innerTail8_5 best = best := by
  rw [innerTail8_5, scan, if_neg (by decide), innerCandidates8_5_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_6
    (if 3 ≤ innerCandidates8_5.length then search edge 3 [31, 991] innerCandidates8_5 1022 best else best) = best
  rw [if_pos (by decide), innerChild8_5]
  exact innerScan8_6

def innerTail8_4 : List Nat := 907 :: innerTail8_5
def innerCandidates8_4 : List Nat := [2383, 4357, 5851, 7591]
set_option maxRecDepth 4096 in
theorem innerCandidates8_4_checked : candidates edge 4 31 907 innerTail8_5 best.bound = innerCandidates8_4 := by
  unfold candidates
  have row : edge 907 = PrimePairCache.smallRow907.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild8_4 : search edge 3 [31, 907] innerCandidates8_4 938 best = best := by decide

theorem innerScan8_4 : scan edge (search edge 3) 4 [31] 31 innerTail8_4 best = best := by
  rw [innerTail8_4, scan, if_neg (by decide), innerCandidates8_4_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_5
    (if 3 ≤ innerCandidates8_4.length then search edge 3 [31, 907] innerCandidates8_4 938 best else best) = best
  rw [if_pos (by decide), innerChild8_4]
  exact innerScan8_5

def innerTail8_3 : List Nat := 859 :: innerTail8_4
def innerCandidates8_3 : List Nat := [3307, 4549]
set_option maxRecDepth 4096 in
theorem innerCandidates8_3_checked : candidates edge 4 31 859 innerTail8_4 best.bound = innerCandidates8_3 := by
  unfold candidates
  have row : edge 859 = PrimePairCache.smallRow859.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_3 : scan edge (search edge 3) 4 [31] 31 innerTail8_3 best = best := by
  rw [innerTail8_3, scan, if_neg (by decide), innerCandidates8_3_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_4
    (if 3 ≤ innerCandidates8_3.length then search edge 3 [31, 859] innerCandidates8_3 890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_4

def innerTail8_2 : List Nat := 181 :: innerTail8_3
def innerCandidates8_2 : List Nat := [2311]
set_option maxRecDepth 4096 in
theorem innerCandidates8_2_checked : candidates edge 4 31 181 innerTail8_3 best.bound = innerCandidates8_2 := by
  unfold candidates
  have row : edge 181 = PrimePairCache.smallRow181.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_2 : scan edge (search edge 3) 4 [31] 31 innerTail8_2 best = best := by
  rw [innerTail8_2, scan, if_neg (by decide), innerCandidates8_2_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_3
    (if 3 ≤ innerCandidates8_2.length then search edge 3 [31, 181] innerCandidates8_2 212 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_3

def innerTail8_1 : List Nat := 151 :: innerTail8_2
def innerCandidates8_1 : List Nat := [2281, 2311, 5011, 5281]
set_option maxRecDepth 4096 in
theorem innerCandidates8_1_checked : candidates edge 4 31 151 innerTail8_2 best.bound = innerCandidates8_1 := by
  unfold candidates
  have row : edge 151 = PrimePairCache.smallRow151.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild8_1 : search edge 3 [31, 151] innerCandidates8_1 182 best = best := by decide

theorem innerScan8_1 : scan edge (search edge 3) 4 [31] 31 innerTail8_1 best = best := by
  rw [innerTail8_1, scan, if_neg (by decide), innerCandidates8_1_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_2
    (if 3 ≤ innerCandidates8_1.length then search edge 3 [31, 151] innerCandidates8_1 182 best else best) = best
  rw [if_pos (by decide), innerChild8_1]
  exact innerScan8_2

def innerTail8_0 : List Nat := 139 :: innerTail8_1
def innerCandidates8_0 : List Nat := [907, 1447]
set_option maxRecDepth 4096 in
theorem innerCandidates8_0_checked : candidates edge 4 31 139 innerTail8_1 best.bound = innerCandidates8_0 := by
  unfold candidates
  have row : edge 139 = PrimePairCache.smallRow139.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan8_0 : scan edge (search edge 3) 4 [31] 31 innerTail8_0 best = best := by
  rw [innerTail8_0, scan, if_neg (by decide), innerCandidates8_0_checked]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_1
    (if 3 ≤ innerCandidates8_0.length then search edge 3 [31, 139] innerCandidates8_0 170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan8_1

theorem rootBranch8 : search edge 4 [31] rootCandidates8 31 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [31] 31 innerTail8_0 best = best
  exact innerScan8_0

end ProjectEulerSolutions.PrimePairUpper.Search
