import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates14 : List Nat := [167, 197, 419, 929, 971, 1091, 1163, 1193, 1259, 1289, 1319, 1601, 1901, 1973, 2087, 2099, 2237, 2309, 2351, 2531, 2609, 2663, 2897, 2903, 3083, 3449, 3767, 3851, 4157, 4211, 4397, 4721, 4751, 4889, 4931, 5039, 5081, 5261, 5303, 5351, 5381, 5519, 5717, 5741, 5807, 5843, 5927, 5981, 6047, 6569, 6653, 6779, 6857, 6863, 6899, 6917, 6977, 6983, 7253, 7349, 7523, 7559, 7673, 8093, 8123]
set_option maxRecDepth 4096 in
theorem rootCandidates14_checked : candidates edge 5 0 59 tail15 best.bound = rootCandidates14 := by
  unfold candidates
  have row : edge 59 = PrimePairCache.smallRow59.contains := by funext q; rfl
  rw [row]
  decide

def innerTail14_65 : List Nat := []
theorem innerScan14_65 : scan edge (search edge 3) 4 [59] 59 innerTail14_65 best = best := rfl

def innerTail14_64 : List Nat := 8123 :: innerTail14_65
def innerCandidates14_64 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_64_checked : candidates edge 4 59 8123 innerTail14_65 best.bound = innerCandidates14_64 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_64 : scan edge (search edge 3) 4 [59] 59 innerTail14_64 best = best := by
  rw [innerTail14_64, scan, if_neg (by decide), innerCandidates14_64_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_65
    (if 3 ≤ innerCandidates14_64.length then search edge 3 [59, 8123] innerCandidates14_64 8182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_65

def innerTail14_63 : List Nat := 8093 :: innerTail14_64
def innerCandidates14_63 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_63_checked : candidates edge 4 59 8093 innerTail14_64 best.bound = innerCandidates14_63 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_63 : scan edge (search edge 3) 4 [59] 59 innerTail14_63 best = best := by
  rw [innerTail14_63, scan, if_neg (by decide), innerCandidates14_63_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_64
    (if 3 ≤ innerCandidates14_63.length then search edge 3 [59, 8093] innerCandidates14_63 8152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_64

def innerTail14_62 : List Nat := 7673 :: innerTail14_63
def innerCandidates14_62 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_62_checked : candidates edge 4 59 7673 innerTail14_63 best.bound = innerCandidates14_62 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_62 : scan edge (search edge 3) 4 [59] 59 innerTail14_62 best = best := by
  rw [innerTail14_62, scan, if_neg (by decide), innerCandidates14_62_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_63
    (if 3 ≤ innerCandidates14_62.length then search edge 3 [59, 7673] innerCandidates14_62 7732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_63

def innerTail14_61 : List Nat := 7559 :: innerTail14_62
def innerCandidates14_61 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_61_checked : candidates edge 4 59 7559 innerTail14_62 best.bound = innerCandidates14_61 := by
  unfold candidates
  have row : edge 7559 = SearchData.row7559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_61 : scan edge (search edge 3) 4 [59] 59 innerTail14_61 best = best := by
  rw [innerTail14_61, scan, if_neg (by decide), innerCandidates14_61_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_62
    (if 3 ≤ innerCandidates14_61.length then search edge 3 [59, 7559] innerCandidates14_61 7618 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_62

def innerTail14_60 : List Nat := 7523 :: innerTail14_61
def innerCandidates14_60 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_60_checked : candidates edge 4 59 7523 innerTail14_61 best.bound = innerCandidates14_60 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_60 : scan edge (search edge 3) 4 [59] 59 innerTail14_60 best = best := by
  rw [innerTail14_60, scan, if_neg (by decide), innerCandidates14_60_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_61
    (if 3 ≤ innerCandidates14_60.length then search edge 3 [59, 7523] innerCandidates14_60 7582 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_61

def innerTail14_59 : List Nat := 7349 :: innerTail14_60
def innerCandidates14_59 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_59_checked : candidates edge 4 59 7349 innerTail14_60 best.bound = innerCandidates14_59 := by
  unfold candidates
  have row : edge 7349 = SearchData.row7349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_59 : scan edge (search edge 3) 4 [59] 59 innerTail14_59 best = best := by
  rw [innerTail14_59, scan, if_neg (by decide), innerCandidates14_59_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_60
    (if 3 ≤ innerCandidates14_59.length then search edge 3 [59, 7349] innerCandidates14_59 7408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_60

def innerTail14_58 : List Nat := 7253 :: innerTail14_59
def innerCandidates14_58 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_58_checked : candidates edge 4 59 7253 innerTail14_59 best.bound = innerCandidates14_58 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_58 : scan edge (search edge 3) 4 [59] 59 innerTail14_58 best = best := by
  rw [innerTail14_58, scan, if_neg (by decide), innerCandidates14_58_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_59
    (if 3 ≤ innerCandidates14_58.length then search edge 3 [59, 7253] innerCandidates14_58 7312 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_59

def innerTail14_57 : List Nat := 6983 :: innerTail14_58
def innerCandidates14_57 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_57_checked : candidates edge 4 59 6983 innerTail14_58 best.bound = innerCandidates14_57 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_57 : scan edge (search edge 3) 4 [59] 59 innerTail14_57 best = best := by
  rw [innerTail14_57, scan, if_neg (by decide), innerCandidates14_57_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_58
    (if 3 ≤ innerCandidates14_57.length then search edge 3 [59, 6983] innerCandidates14_57 7042 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_58

def innerTail14_56 : List Nat := 6977 :: innerTail14_57
def innerCandidates14_56 : List Nat := [7349]
set_option maxRecDepth 4096 in
theorem innerCandidates14_56_checked : candidates edge 4 59 6977 innerTail14_57 best.bound = innerCandidates14_56 := by
  unfold candidates
  have row : edge 6977 = SearchData.row6977.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_56 : scan edge (search edge 3) 4 [59] 59 innerTail14_56 best = best := by
  rw [innerTail14_56, scan, if_neg (by decide), innerCandidates14_56_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_57
    (if 3 ≤ innerCandidates14_56.length then search edge 3 [59, 6977] innerCandidates14_56 7036 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_57

def innerTail14_55 : List Nat := 6917 :: innerTail14_56
def innerCandidates14_55 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_55_checked : candidates edge 4 59 6917 innerTail14_56 best.bound = innerCandidates14_55 := by
  unfold candidates
  have row : edge 6917 = SearchData.row6917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_55 : scan edge (search edge 3) 4 [59] 59 innerTail14_55 best = best := by
  rw [innerTail14_55, scan, if_neg (by decide), innerCandidates14_55_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_56
    (if 3 ≤ innerCandidates14_55.length then search edge 3 [59, 6917] innerCandidates14_55 6976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_56

def innerTail14_54 : List Nat := 6899 :: innerTail14_55
def innerCandidates14_54 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_54_checked : candidates edge 4 59 6899 innerTail14_55 best.bound = innerCandidates14_54 := by
  unfold candidates
  have row : edge 6899 = SearchData.row6899.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_54 : scan edge (search edge 3) 4 [59] 59 innerTail14_54 best = best := by
  rw [innerTail14_54, scan, if_neg (by decide), innerCandidates14_54_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_55
    (if 3 ≤ innerCandidates14_54.length then search edge 3 [59, 6899] innerCandidates14_54 6958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_55

def innerTail14_53 : List Nat := 6863 :: innerTail14_54
def innerCandidates14_53 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_53_checked : candidates edge 4 59 6863 innerTail14_54 best.bound = innerCandidates14_53 := by
  unfold candidates
  have row : edge 6863 = SearchData.row6863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_53 : scan edge (search edge 3) 4 [59] 59 innerTail14_53 best = best := by
  rw [innerTail14_53, scan, if_neg (by decide), innerCandidates14_53_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_54
    (if 3 ≤ innerCandidates14_53.length then search edge 3 [59, 6863] innerCandidates14_53 6922 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_54

def innerTail14_52 : List Nat := 6857 :: innerTail14_53
def innerCandidates14_52 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_52_checked : candidates edge 4 59 6857 innerTail14_53 best.bound = innerCandidates14_52 := by
  unfold candidates
  have row : edge 6857 = SearchData.row6857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_52 : scan edge (search edge 3) 4 [59] 59 innerTail14_52 best = best := by
  rw [innerTail14_52, scan, if_neg (by decide), innerCandidates14_52_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_53
    (if 3 ≤ innerCandidates14_52.length then search edge 3 [59, 6857] innerCandidates14_52 6916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_53

def innerTail14_51 : List Nat := 6779 :: innerTail14_52
def innerCandidates14_51 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_51_checked : candidates edge 4 59 6779 innerTail14_52 best.bound = innerCandidates14_51 := by
  unfold candidates
  have row : edge 6779 = SearchData.row6779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_51 : scan edge (search edge 3) 4 [59] 59 innerTail14_51 best = best := by
  rw [innerTail14_51, scan, if_neg (by decide), innerCandidates14_51_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_52
    (if 3 ≤ innerCandidates14_51.length then search edge 3 [59, 6779] innerCandidates14_51 6838 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_52

def innerTail14_50 : List Nat := 6653 :: innerTail14_51
def innerCandidates14_50 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_50_checked : candidates edge 4 59 6653 innerTail14_51 best.bound = innerCandidates14_50 := by
  unfold candidates
  have row : edge 6653 = SearchData.row6653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_50 : scan edge (search edge 3) 4 [59] 59 innerTail14_50 best = best := by
  rw [innerTail14_50, scan, if_neg (by decide), innerCandidates14_50_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_51
    (if 3 ≤ innerCandidates14_50.length then search edge 3 [59, 6653] innerCandidates14_50 6712 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_51

def innerTail14_49 : List Nat := 6569 :: innerTail14_50
def innerCandidates14_49 : List Nat := [6779]
set_option maxRecDepth 4096 in
theorem innerCandidates14_49_checked : candidates edge 4 59 6569 innerTail14_50 best.bound = innerCandidates14_49 := by
  unfold candidates
  have row : edge 6569 = SearchData.row6569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_49 : scan edge (search edge 3) 4 [59] 59 innerTail14_49 best = best := by
  rw [innerTail14_49, scan, if_neg (by decide), innerCandidates14_49_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_50
    (if 3 ≤ innerCandidates14_49.length then search edge 3 [59, 6569] innerCandidates14_49 6628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_50

def innerTail14_48 : List Nat := 6047 :: innerTail14_49
def innerCandidates14_48 : List Nat := [6569]
set_option maxRecDepth 4096 in
theorem innerCandidates14_48_checked : candidates edge 4 59 6047 innerTail14_49 best.bound = innerCandidates14_48 := by
  unfold candidates
  have row : edge 6047 = SearchData.row6047.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_48 : scan edge (search edge 3) 4 [59] 59 innerTail14_48 best = best := by
  rw [innerTail14_48, scan, if_neg (by decide), innerCandidates14_48_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_49
    (if 3 ≤ innerCandidates14_48.length then search edge 3 [59, 6047] innerCandidates14_48 6106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_49

def innerTail14_47 : List Nat := 5981 :: innerTail14_48
def innerCandidates14_47 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_47_checked : candidates edge 4 59 5981 innerTail14_48 best.bound = innerCandidates14_47 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_47 : scan edge (search edge 3) 4 [59] 59 innerTail14_47 best = best := by
  rw [innerTail14_47, scan, if_neg (by decide), innerCandidates14_47_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_48
    (if 3 ≤ innerCandidates14_47.length then search edge 3 [59, 5981] innerCandidates14_47 6040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_48

def innerTail14_46 : List Nat := 5927 :: innerTail14_47
def innerCandidates14_46 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_46_checked : candidates edge 4 59 5927 innerTail14_47 best.bound = innerCandidates14_46 := by
  unfold candidates
  have row : edge 5927 = SearchData.row5927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_46 : scan edge (search edge 3) 4 [59] 59 innerTail14_46 best = best := by
  rw [innerTail14_46, scan, if_neg (by decide), innerCandidates14_46_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_47
    (if 3 ≤ innerCandidates14_46.length then search edge 3 [59, 5927] innerCandidates14_46 5986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_47

def innerTail14_45 : List Nat := 5843 :: innerTail14_46
def innerCandidates14_45 : List Nat := [6977]
set_option maxRecDepth 4096 in
theorem innerCandidates14_45_checked : candidates edge 4 59 5843 innerTail14_46 best.bound = innerCandidates14_45 := by
  unfold candidates
  have row : edge 5843 = SearchData.row5843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_45 : scan edge (search edge 3) 4 [59] 59 innerTail14_45 best = best := by
  rw [innerTail14_45, scan, if_neg (by decide), innerCandidates14_45_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_46
    (if 3 ≤ innerCandidates14_45.length then search edge 3 [59, 5843] innerCandidates14_45 5902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_46

def innerTail14_44 : List Nat := 5807 :: innerTail14_45
def innerCandidates14_44 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_44_checked : candidates edge 4 59 5807 innerTail14_45 best.bound = innerCandidates14_44 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_44 : scan edge (search edge 3) 4 [59] 59 innerTail14_44 best = best := by
  rw [innerTail14_44, scan, if_neg (by decide), innerCandidates14_44_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_45
    (if 3 ≤ innerCandidates14_44.length then search edge 3 [59, 5807] innerCandidates14_44 5866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_45

def innerTail14_43 : List Nat := 5741 :: innerTail14_44
def innerCandidates14_43 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates14_43_checked : candidates edge 4 59 5741 innerTail14_44 best.bound = innerCandidates14_43 := by
  unfold candidates
  have row : edge 5741 = SearchData.row5741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_43 : scan edge (search edge 3) 4 [59] 59 innerTail14_43 best = best := by
  rw [innerTail14_43, scan, if_neg (by decide), innerCandidates14_43_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_44
    (if 3 ≤ innerCandidates14_43.length then search edge 3 [59, 5741] innerCandidates14_43 5800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_44

def innerTail14_42 : List Nat := 5717 :: innerTail14_43
def innerCandidates14_42 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_42_checked : candidates edge 4 59 5717 innerTail14_43 best.bound = innerCandidates14_42 := by
  unfold candidates
  have row : edge 5717 = SearchData.row5717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_42 : scan edge (search edge 3) 4 [59] 59 innerTail14_42 best = best := by
  rw [innerTail14_42, scan, if_neg (by decide), innerCandidates14_42_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_43
    (if 3 ≤ innerCandidates14_42.length then search edge 3 [59, 5717] innerCandidates14_42 5776 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_43

def innerTail14_41 : List Nat := 5519 :: innerTail14_42
def innerCandidates14_41 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_41_checked : candidates edge 4 59 5519 innerTail14_42 best.bound = innerCandidates14_41 := by
  unfold candidates
  have row : edge 5519 = SearchData.row5519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_41 : scan edge (search edge 3) 4 [59] 59 innerTail14_41 best = best := by
  rw [innerTail14_41, scan, if_neg (by decide), innerCandidates14_41_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_42
    (if 3 ≤ innerCandidates14_41.length then search edge 3 [59, 5519] innerCandidates14_41 5578 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_42

def innerTail14_40 : List Nat := 5381 :: innerTail14_41
def innerCandidates14_40 : List Nat := [5981]
set_option maxRecDepth 4096 in
theorem innerCandidates14_40_checked : candidates edge 4 59 5381 innerTail14_41 best.bound = innerCandidates14_40 := by
  unfold candidates
  have row : edge 5381 = SearchData.row5381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_40 : scan edge (search edge 3) 4 [59] 59 innerTail14_40 best = best := by
  rw [innerTail14_40, scan, if_neg (by decide), innerCandidates14_40_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_41
    (if 3 ≤ innerCandidates14_40.length then search edge 3 [59, 5381] innerCandidates14_40 5440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_41

def innerTail14_39 : List Nat := 5351 :: innerTail14_40
def innerCandidates14_39 : List Nat := [5519, 5927, 6899]
set_option maxRecDepth 4096 in
theorem innerCandidates14_39_checked : candidates edge 4 59 5351 innerTail14_40 best.bound = innerCandidates14_39 := by
  unfold candidates
  have row : edge 5351 = SearchData.row5351.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild14_39 : search edge 3 [59, 5351] innerCandidates14_39 5410 best = best := by decide

theorem innerScan14_39 : scan edge (search edge 3) 4 [59] 59 innerTail14_39 best = best := by
  rw [innerTail14_39, scan, if_neg (by decide), innerCandidates14_39_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_40
    (if 3 ≤ innerCandidates14_39.length then search edge 3 [59, 5351] innerCandidates14_39 5410 best else best) = best
  rw [if_pos (by decide), innerChild14_39]
  exact innerScan14_40

def innerTail14_38 : List Nat := 5303 :: innerTail14_39
def innerCandidates14_38 : List Nat := [8123]
set_option maxRecDepth 4096 in
theorem innerCandidates14_38_checked : candidates edge 4 59 5303 innerTail14_39 best.bound = innerCandidates14_38 := by
  unfold candidates
  have row : edge 5303 = SearchData.row5303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_38 : scan edge (search edge 3) 4 [59] 59 innerTail14_38 best = best := by
  rw [innerTail14_38, scan, if_neg (by decide), innerCandidates14_38_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_39
    (if 3 ≤ innerCandidates14_38.length then search edge 3 [59, 5303] innerCandidates14_38 5362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_39

def innerTail14_37 : List Nat := 5261 :: innerTail14_38
def innerCandidates14_37 : List Nat := [6779, 6983]
set_option maxRecDepth 4096 in
theorem innerCandidates14_37_checked : candidates edge 4 59 5261 innerTail14_38 best.bound = innerCandidates14_37 := by
  unfold candidates
  have row : edge 5261 = SearchData.row5261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_37 : scan edge (search edge 3) 4 [59] 59 innerTail14_37 best = best := by
  rw [innerTail14_37, scan, if_neg (by decide), innerCandidates14_37_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_38
    (if 3 ≤ innerCandidates14_37.length then search edge 3 [59, 5261] innerCandidates14_37 5320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_38

def innerTail14_36 : List Nat := 5081 :: innerTail14_37
def innerCandidates14_36 : List Nat := [7253]
set_option maxRecDepth 4096 in
theorem innerCandidates14_36_checked : candidates edge 4 59 5081 innerTail14_37 best.bound = innerCandidates14_36 := by
  unfold candidates
  have row : edge 5081 = SearchData.row5081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_36 : scan edge (search edge 3) 4 [59] 59 innerTail14_36 best = best := by
  rw [innerTail14_36, scan, if_neg (by decide), innerCandidates14_36_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_37
    (if 3 ≤ innerCandidates14_36.length then search edge 3 [59, 5081] innerCandidates14_36 5140 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_37

def innerTail14_35 : List Nat := 5039 :: innerTail14_36
def innerCandidates14_35 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates14_35_checked : candidates edge 4 59 5039 innerTail14_36 best.bound = innerCandidates14_35 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_35 : scan edge (search edge 3) 4 [59] 59 innerTail14_35 best = best := by
  rw [innerTail14_35, scan, if_neg (by decide), innerCandidates14_35_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_36
    (if 3 ≤ innerCandidates14_35.length then search edge 3 [59, 5039] innerCandidates14_35 5098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_36

def innerTail14_34 : List Nat := 4931 :: innerTail14_35
def innerCandidates14_34 : List Nat := [5843, 7349]
set_option maxRecDepth 4096 in
theorem innerCandidates14_34_checked : candidates edge 4 59 4931 innerTail14_35 best.bound = innerCandidates14_34 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_34 : scan edge (search edge 3) 4 [59] 59 innerTail14_34 best = best := by
  rw [innerTail14_34, scan, if_neg (by decide), innerCandidates14_34_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_35
    (if 3 ≤ innerCandidates14_34.length then search edge 3 [59, 4931] innerCandidates14_34 4990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_35

def innerTail14_33 : List Nat := 4889 :: innerTail14_34
def innerCandidates14_33 : List Nat := [8123]
set_option maxRecDepth 4096 in
theorem innerCandidates14_33_checked : candidates edge 4 59 4889 innerTail14_34 best.bound = innerCandidates14_33 := by
  unfold candidates
  have row : edge 4889 = SearchData.row4889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_33 : scan edge (search edge 3) 4 [59] 59 innerTail14_33 best = best := by
  rw [innerTail14_33, scan, if_neg (by decide), innerCandidates14_33_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_34
    (if 3 ≤ innerCandidates14_33.length then search edge 3 [59, 4889] innerCandidates14_33 4948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_34

def innerTail14_32 : List Nat := 4751 :: innerTail14_33
def innerCandidates14_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_32_checked : candidates edge 4 59 4751 innerTail14_33 best.bound = innerCandidates14_32 := by
  unfold candidates
  have row : edge 4751 = SearchData.row4751.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_32 : scan edge (search edge 3) 4 [59] 59 innerTail14_32 best = best := by
  rw [innerTail14_32, scan, if_neg (by decide), innerCandidates14_32_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_33
    (if 3 ≤ innerCandidates14_32.length then search edge 3 [59, 4751] innerCandidates14_32 4810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_33

def innerTail14_31 : List Nat := 4721 :: innerTail14_32
def innerCandidates14_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_31_checked : candidates edge 4 59 4721 innerTail14_32 best.bound = innerCandidates14_31 := by
  unfold candidates
  have row : edge 4721 = SearchData.row4721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_31 : scan edge (search edge 3) 4 [59] 59 innerTail14_31 best = best := by
  rw [innerTail14_31, scan, if_neg (by decide), innerCandidates14_31_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_32
    (if 3 ≤ innerCandidates14_31.length then search edge 3 [59, 4721] innerCandidates14_31 4780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_32

def innerTail14_30 : List Nat := 4397 :: innerTail14_31
def innerCandidates14_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_30_checked : candidates edge 4 59 4397 innerTail14_31 best.bound = innerCandidates14_30 := by
  unfold candidates
  have row : edge 4397 = SearchData.row4397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_30 : scan edge (search edge 3) 4 [59] 59 innerTail14_30 best = best := by
  rw [innerTail14_30, scan, if_neg (by decide), innerCandidates14_30_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_31
    (if 3 ≤ innerCandidates14_30.length then search edge 3 [59, 4397] innerCandidates14_30 4456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_31

def innerTail14_29 : List Nat := 4211 :: innerTail14_30
def innerCandidates14_29 : List Nat := [5351]
set_option maxRecDepth 4096 in
theorem innerCandidates14_29_checked : candidates edge 4 59 4211 innerTail14_30 best.bound = innerCandidates14_29 := by
  unfold candidates
  have row : edge 4211 = SearchData.row4211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_29 : scan edge (search edge 3) 4 [59] 59 innerTail14_29 best = best := by
  rw [innerTail14_29, scan, if_neg (by decide), innerCandidates14_29_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_30
    (if 3 ≤ innerCandidates14_29.length then search edge 3 [59, 4211] innerCandidates14_29 4270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_30

def innerTail14_28 : List Nat := 4157 :: innerTail14_29
def innerCandidates14_28 : List Nat := [5717]
set_option maxRecDepth 4096 in
theorem innerCandidates14_28_checked : candidates edge 4 59 4157 innerTail14_29 best.bound = innerCandidates14_28 := by
  unfold candidates
  have row : edge 4157 = SearchData.row4157.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_28 : scan edge (search edge 3) 4 [59] 59 innerTail14_28 best = best := by
  rw [innerTail14_28, scan, if_neg (by decide), innerCandidates14_28_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_29
    (if 3 ≤ innerCandidates14_28.length then search edge 3 [59, 4157] innerCandidates14_28 4216 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_29

def innerTail14_27 : List Nat := 3851 :: innerTail14_28
def innerCandidates14_27 : List Nat := [6899, 6983, 7673]
set_option maxRecDepth 4096 in
theorem innerCandidates14_27_checked : candidates edge 4 59 3851 innerTail14_28 best.bound = innerCandidates14_27 := by
  unfold candidates
  have row : edge 3851 = SearchData.row3851.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild14_27 : search edge 3 [59, 3851] innerCandidates14_27 3910 best = best := by decide

theorem innerScan14_27 : scan edge (search edge 3) 4 [59] 59 innerTail14_27 best = best := by
  rw [innerTail14_27, scan, if_neg (by decide), innerCandidates14_27_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_28
    (if 3 ≤ innerCandidates14_27.length then search edge 3 [59, 3851] innerCandidates14_27 3910 best else best) = best
  rw [if_pos (by decide), innerChild14_27]
  exact innerScan14_28

def innerTail14_26 : List Nat := 3767 :: innerTail14_27
def innerCandidates14_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_26_checked : candidates edge 4 59 3767 innerTail14_27 best.bound = innerCandidates14_26 := by
  unfold candidates
  have row : edge 3767 = SearchData.row3767.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_26 : scan edge (search edge 3) 4 [59] 59 innerTail14_26 best = best := by
  rw [innerTail14_26, scan, if_neg (by decide), innerCandidates14_26_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_27
    (if 3 ≤ innerCandidates14_26.length then search edge 3 [59, 3767] innerCandidates14_26 3826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_27

def innerTail14_25 : List Nat := 3449 :: innerTail14_26
def innerCandidates14_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_25_checked : candidates edge 4 59 3449 innerTail14_26 best.bound = innerCandidates14_25 := by
  unfold candidates
  have row : edge 3449 = SearchData.row3449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_25 : scan edge (search edge 3) 4 [59] 59 innerTail14_25 best = best := by
  rw [innerTail14_25, scan, if_neg (by decide), innerCandidates14_25_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_26
    (if 3 ≤ innerCandidates14_25.length then search edge 3 [59, 3449] innerCandidates14_25 3508 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_26

def innerTail14_24 : List Nat := 3083 :: innerTail14_25
def innerCandidates14_24 : List Nat := [3449, 4931]
set_option maxRecDepth 4096 in
theorem innerCandidates14_24_checked : candidates edge 4 59 3083 innerTail14_25 best.bound = innerCandidates14_24 := by
  unfold candidates
  have row : edge 3083 = SearchData.row3083.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_24 : scan edge (search edge 3) 4 [59] 59 innerTail14_24 best = best := by
  rw [innerTail14_24, scan, if_neg (by decide), innerCandidates14_24_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_25
    (if 3 ≤ innerCandidates14_24.length then search edge 3 [59, 3083] innerCandidates14_24 3142 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_25

def innerTail14_23 : List Nat := 2903 :: innerTail14_24
def innerCandidates14_23 : List Nat := [4721, 5927, 6779, 8123]
set_option maxRecDepth 4096 in
theorem innerCandidates14_23_checked : candidates edge 4 59 2903 innerTail14_24 best.bound = innerCandidates14_23 := by
  unfold candidates
  have row : edge 2903 = SearchData.row2903.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild14_23 : search edge 3 [59, 2903] innerCandidates14_23 2962 best = best := by decide

theorem innerScan14_23 : scan edge (search edge 3) 4 [59] 59 innerTail14_23 best = best := by
  rw [innerTail14_23, scan, if_neg (by decide), innerCandidates14_23_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_24
    (if 3 ≤ innerCandidates14_23.length then search edge 3 [59, 2903] innerCandidates14_23 2962 best else best) = best
  rw [if_pos (by decide), innerChild14_23]
  exact innerScan14_24

def innerTail14_22 : List Nat := 2897 :: innerTail14_23
def innerCandidates14_22 : List Nat := [4721]
set_option maxRecDepth 4096 in
theorem innerCandidates14_22_checked : candidates edge 4 59 2897 innerTail14_23 best.bound = innerCandidates14_22 := by
  unfold candidates
  have row : edge 2897 = SearchData.row2897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_22 : scan edge (search edge 3) 4 [59] 59 innerTail14_22 best = best := by
  rw [innerTail14_22, scan, if_neg (by decide), innerCandidates14_22_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_23
    (if 3 ≤ innerCandidates14_22.length then search edge 3 [59, 2897] innerCandidates14_22 2956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_23

def innerTail14_21 : List Nat := 2663 :: innerTail14_22
def innerCandidates14_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_21_checked : candidates edge 4 59 2663 innerTail14_22 best.bound = innerCandidates14_21 := by
  unfold candidates
  have row : edge 2663 = PrimePairCache.smallRow2663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_21 : scan edge (search edge 3) 4 [59] 59 innerTail14_21 best = best := by
  rw [innerTail14_21, scan, if_neg (by decide), innerCandidates14_21_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_22
    (if 3 ≤ innerCandidates14_21.length then search edge 3 [59, 2663] innerCandidates14_21 2722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_22

def innerTail14_20 : List Nat := 2609 :: innerTail14_21
def innerCandidates14_20 : List Nat := [4931, 6569]
set_option maxRecDepth 4096 in
theorem innerCandidates14_20_checked : candidates edge 4 59 2609 innerTail14_21 best.bound = innerCandidates14_20 := by
  unfold candidates
  have row : edge 2609 = PrimePairCache.smallRow2609.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_20 : scan edge (search edge 3) 4 [59] 59 innerTail14_20 best = best := by
  rw [innerTail14_20, scan, if_neg (by decide), innerCandidates14_20_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_21
    (if 3 ≤ innerCandidates14_20.length then search edge 3 [59, 2609] innerCandidates14_20 2668 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_21

def innerTail14_19 : List Nat := 2531 :: innerTail14_20
def innerCandidates14_19 : List Nat := [4889, 8123]
set_option maxRecDepth 4096 in
theorem innerCandidates14_19_checked : candidates edge 4 59 2531 innerTail14_20 best.bound = innerCandidates14_19 := by
  unfold candidates
  have row : edge 2531 = PrimePairCache.smallRow2531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_19 : scan edge (search edge 3) 4 [59] 59 innerTail14_19 best = best := by
  rw [innerTail14_19, scan, if_neg (by decide), innerCandidates14_19_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_20
    (if 3 ≤ innerCandidates14_19.length then search edge 3 [59, 2531] innerCandidates14_19 2590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_20

def innerTail14_18 : List Nat := 2351 :: innerTail14_19
def innerCandidates14_18 : List Nat := [5519]
set_option maxRecDepth 4096 in
theorem innerCandidates14_18_checked : candidates edge 4 59 2351 innerTail14_19 best.bound = innerCandidates14_18 := by
  unfold candidates
  have row : edge 2351 = PrimePairCache.smallRow2351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_18 : scan edge (search edge 3) 4 [59] 59 innerTail14_18 best = best := by
  rw [innerTail14_18, scan, if_neg (by decide), innerCandidates14_18_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_19
    (if 3 ≤ innerCandidates14_18.length then search edge 3 [59, 2351] innerCandidates14_18 2410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_19

def innerTail14_17 : List Nat := 2309 :: innerTail14_18
def innerCandidates14_17 : List Nat := [5081]
set_option maxRecDepth 4096 in
theorem innerCandidates14_17_checked : candidates edge 4 59 2309 innerTail14_18 best.bound = innerCandidates14_17 := by
  unfold candidates
  have row : edge 2309 = PrimePairCache.smallRow2309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_17 : scan edge (search edge 3) 4 [59] 59 innerTail14_17 best = best := by
  rw [innerTail14_17, scan, if_neg (by decide), innerCandidates14_17_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_18
    (if 3 ≤ innerCandidates14_17.length then search edge 3 [59, 2309] innerCandidates14_17 2368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_18

def innerTail14_16 : List Nat := 2237 :: innerTail14_17
def innerCandidates14_16 : List Nat := [6779, 6917]
set_option maxRecDepth 4096 in
theorem innerCandidates14_16_checked : candidates edge 4 59 2237 innerTail14_17 best.bound = innerCandidates14_16 := by
  unfold candidates
  have row : edge 2237 = PrimePairCache.smallRow2237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_16 : scan edge (search edge 3) 4 [59] 59 innerTail14_16 best = best := by
  rw [innerTail14_16, scan, if_neg (by decide), innerCandidates14_16_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_17
    (if 3 ≤ innerCandidates14_16.length then search edge 3 [59, 2237] innerCandidates14_16 2296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_17

def innerTail14_15 : List Nat := 2099 :: innerTail14_16
def innerCandidates14_15 : List Nat := [3083, 5717, 6863, 8093]
set_option maxRecDepth 4096 in
theorem innerCandidates14_15_checked : candidates edge 4 59 2099 innerTail14_16 best.bound = innerCandidates14_15 := by
  unfold candidates
  have row : edge 2099 = PrimePairCache.smallRow2099.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild14_15 : search edge 3 [59, 2099] innerCandidates14_15 2158 best = best := by decide

theorem innerScan14_15 : scan edge (search edge 3) 4 [59] 59 innerTail14_15 best = best := by
  rw [innerTail14_15, scan, if_neg (by decide), innerCandidates14_15_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_16
    (if 3 ≤ innerCandidates14_15.length then search edge 3 [59, 2099] innerCandidates14_15 2158 best else best) = best
  rw [if_pos (by decide), innerChild14_15]
  exact innerScan14_16

def innerTail14_14 : List Nat := 2087 :: innerTail14_15
def innerCandidates14_14 : List Nat := [7559]
set_option maxRecDepth 4096 in
theorem innerCandidates14_14_checked : candidates edge 4 59 2087 innerTail14_15 best.bound = innerCandidates14_14 := by
  unfold candidates
  have row : edge 2087 = PrimePairCache.smallRow2087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_14 : scan edge (search edge 3) 4 [59] 59 innerTail14_14 best = best := by
  rw [innerTail14_14, scan, if_neg (by decide), innerCandidates14_14_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_15
    (if 3 ≤ innerCandidates14_14.length then search edge 3 [59, 2087] innerCandidates14_14 2146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_15

def innerTail14_13 : List Nat := 1973 :: innerTail14_14
def innerCandidates14_13 : List Nat := [4397, 4751, 5741]
set_option maxRecDepth 4096 in
theorem innerCandidates14_13_checked : candidates edge 4 59 1973 innerTail14_14 best.bound = innerCandidates14_13 := by
  unfold candidates
  have row : edge 1973 = PrimePairCache.smallRow1973.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild14_13 : search edge 3 [59, 1973] innerCandidates14_13 2032 best = best := by decide

theorem innerScan14_13 : scan edge (search edge 3) 4 [59] 59 innerTail14_13 best = best := by
  rw [innerTail14_13, scan, if_neg (by decide), innerCandidates14_13_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_14
    (if 3 ≤ innerCandidates14_13.length then search edge 3 [59, 1973] innerCandidates14_13 2032 best else best) = best
  rw [if_pos (by decide), innerChild14_13]
  exact innerScan14_14

def innerTail14_12 : List Nat := 1901 :: innerTail14_13
def innerCandidates14_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_12_checked : candidates edge 4 59 1901 innerTail14_13 best.bound = innerCandidates14_12 := by
  unfold candidates
  have row : edge 1901 = PrimePairCache.smallRow1901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_12 : scan edge (search edge 3) 4 [59] 59 innerTail14_12 best = best := by
  rw [innerTail14_12, scan, if_neg (by decide), innerCandidates14_12_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_13
    (if 3 ≤ innerCandidates14_12.length then search edge 3 [59, 1901] innerCandidates14_12 1960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_13

def innerTail14_11 : List Nat := 1601 :: innerTail14_12
def innerCandidates14_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_11_checked : candidates edge 4 59 1601 innerTail14_12 best.bound = innerCandidates14_11 := by
  unfold candidates
  have row : edge 1601 = PrimePairCache.smallRow1601.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_11 : scan edge (search edge 3) 4 [59] 59 innerTail14_11 best = best := by
  rw [innerTail14_11, scan, if_neg (by decide), innerCandidates14_11_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_12
    (if 3 ≤ innerCandidates14_11.length then search edge 3 [59, 1601] innerCandidates14_11 1660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_12

def innerTail14_10 : List Nat := 1319 :: innerTail14_11
def innerCandidates14_10 : List Nat := [1901, 5843, 5981]
set_option maxRecDepth 4096 in
theorem innerCandidates14_10_checked : candidates edge 4 59 1319 innerTail14_11 best.bound = innerCandidates14_10 := by
  unfold candidates
  have row : edge 1319 = PrimePairCache.smallRow1319.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild14_10 : search edge 3 [59, 1319] innerCandidates14_10 1378 best = best := by decide

theorem innerScan14_10 : scan edge (search edge 3) 4 [59] 59 innerTail14_10 best = best := by
  rw [innerTail14_10, scan, if_neg (by decide), innerCandidates14_10_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_11
    (if 3 ≤ innerCandidates14_10.length then search edge 3 [59, 1319] innerCandidates14_10 1378 best else best) = best
  rw [if_pos (by decide), innerChild14_10]
  exact innerScan14_11

def innerTail14_9 : List Nat := 1289 :: innerTail14_10
def innerCandidates14_9 : List Nat := [2237, 3767]
set_option maxRecDepth 4096 in
theorem innerCandidates14_9_checked : candidates edge 4 59 1289 innerTail14_10 best.bound = innerCandidates14_9 := by
  unfold candidates
  have row : edge 1289 = PrimePairCache.smallRow1289.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_9 : scan edge (search edge 3) 4 [59] 59 innerTail14_9 best = best := by
  rw [innerTail14_9, scan, if_neg (by decide), innerCandidates14_9_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_10
    (if 3 ≤ innerCandidates14_9.length then search edge 3 [59, 1289] innerCandidates14_9 1348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_10

def innerTail14_8 : List Nat := 1259 :: innerTail14_9
def innerCandidates14_8 : List Nat := [2609, 5039]
set_option maxRecDepth 4096 in
theorem innerCandidates14_8_checked : candidates edge 4 59 1259 innerTail14_9 best.bound = innerCandidates14_8 := by
  unfold candidates
  have row : edge 1259 = PrimePairCache.smallRow1259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_8 : scan edge (search edge 3) 4 [59] 59 innerTail14_8 best = best := by
  rw [innerTail14_8, scan, if_neg (by decide), innerCandidates14_8_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_9
    (if 3 ≤ innerCandidates14_8.length then search edge 3 [59, 1259] innerCandidates14_8 1318 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_9

def innerTail14_7 : List Nat := 1193 :: innerTail14_8
def innerCandidates14_7 : List Nat := [1901, 4931, 5519, 6779, 7559]
set_option maxRecDepth 4096 in
theorem innerCandidates14_7_checked : candidates edge 4 59 1193 innerTail14_8 best.bound = innerCandidates14_7 := by
  unfold candidates
  have row : edge 1193 = PrimePairCache.smallRow1193.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild14_7 : search edge 3 [59, 1193] innerCandidates14_7 1252 best = best := by decide

theorem innerScan14_7 : scan edge (search edge 3) 4 [59] 59 innerTail14_7 best = best := by
  rw [innerTail14_7, scan, if_neg (by decide), innerCandidates14_7_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_8
    (if 3 ≤ innerCandidates14_7.length then search edge 3 [59, 1193] innerCandidates14_7 1252 best else best) = best
  rw [if_pos (by decide), innerChild14_7]
  exact innerScan14_8

def innerTail14_6 : List Nat := 1163 :: innerTail14_7
def innerCandidates14_6 : List Nat := [6899, 7349]
set_option maxRecDepth 4096 in
theorem innerCandidates14_6_checked : candidates edge 4 59 1163 innerTail14_7 best.bound = innerCandidates14_6 := by
  unfold candidates
  have row : edge 1163 = PrimePairCache.smallRow1163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_6 : scan edge (search edge 3) 4 [59] 59 innerTail14_6 best = best := by
  rw [innerTail14_6, scan, if_neg (by decide), innerCandidates14_6_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_7
    (if 3 ≤ innerCandidates14_6.length then search edge 3 [59, 1163] innerCandidates14_6 1222 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_7

def innerTail14_5 : List Nat := 1091 :: innerTail14_6
def innerCandidates14_5 : List Nat := [1319, 4931]
set_option maxRecDepth 4096 in
theorem innerCandidates14_5_checked : candidates edge 4 59 1091 innerTail14_6 best.bound = innerCandidates14_5 := by
  unfold candidates
  have row : edge 1091 = PrimePairCache.smallRow1091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_5 : scan edge (search edge 3) 4 [59] 59 innerTail14_5 best = best := by
  rw [innerTail14_5, scan, if_neg (by decide), innerCandidates14_5_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_6
    (if 3 ≤ innerCandidates14_5.length then search edge 3 [59, 1091] innerCandidates14_5 1150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_6

def innerTail14_4 : List Nat := 971 :: innerTail14_5
def innerCandidates14_4 : List Nat := [2897, 6569, 8123]
set_option maxRecDepth 4096 in
theorem innerCandidates14_4_checked : candidates edge 4 59 971 innerTail14_5 best.bound = innerCandidates14_4 := by
  unfold candidates
  have row : edge 971 = PrimePairCache.smallRow971.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild14_4 : search edge 3 [59, 971] innerCandidates14_4 1030 best = best := by decide

theorem innerScan14_4 : scan edge (search edge 3) 4 [59] 59 innerTail14_4 best = best := by
  rw [innerTail14_4, scan, if_neg (by decide), innerCandidates14_4_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_5
    (if 3 ≤ innerCandidates14_4.length then search edge 3 [59, 971] innerCandidates14_4 1030 best else best) = best
  rw [if_pos (by decide), innerChild14_4]
  exact innerScan14_5

def innerTail14_3 : List Nat := 929 :: innerTail14_4
def innerCandidates14_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates14_3_checked : candidates edge 4 59 929 innerTail14_4 best.bound = innerCandidates14_3 := by
  unfold candidates
  have row : edge 929 = PrimePairCache.smallRow929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_3 : scan edge (search edge 3) 4 [59] 59 innerTail14_3 best = best := by
  rw [innerTail14_3, scan, if_neg (by decide), innerCandidates14_3_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_4
    (if 3 ≤ innerCandidates14_3.length then search edge 3 [59, 929] innerCandidates14_3 988 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_4

def innerTail14_2 : List Nat := 419 :: innerTail14_3
def innerCandidates14_2 : List Nat := [929, 2087, 7253]
set_option maxRecDepth 4096 in
theorem innerCandidates14_2_checked : candidates edge 4 59 419 innerTail14_3 best.bound = innerCandidates14_2 := by
  unfold candidates
  have row : edge 419 = PrimePairCache.smallRow419.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild14_2 : search edge 3 [59, 419] innerCandidates14_2 478 best = best := by decide

theorem innerScan14_2 : scan edge (search edge 3) 4 [59] 59 innerTail14_2 best = best := by
  rw [innerTail14_2, scan, if_neg (by decide), innerCandidates14_2_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_3
    (if 3 ≤ innerCandidates14_2.length then search edge 3 [59, 419] innerCandidates14_2 478 best else best) = best
  rw [if_pos (by decide), innerChild14_2]
  exact innerScan14_3

def innerTail14_1 : List Nat := 197 :: innerTail14_2
def innerCandidates14_1 : List Nat := [971, 2099]
set_option maxRecDepth 4096 in
theorem innerCandidates14_1_checked : candidates edge 4 59 197 innerTail14_2 best.bound = innerCandidates14_1 := by
  unfold candidates
  have row : edge 197 = PrimePairCache.smallRow197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan14_1 : scan edge (search edge 3) 4 [59] 59 innerTail14_1 best = best := by
  rw [innerTail14_1, scan, if_neg (by decide), innerCandidates14_1_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_2
    (if 3 ≤ innerCandidates14_1.length then search edge 3 [59, 197] innerCandidates14_1 256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan14_2

def innerTail14_0 : List Nat := 167 :: innerTail14_1
def innerCandidates14_0 : List Nat := [2087, 5981, 7253, 7673]
set_option maxRecDepth 4096 in
theorem innerCandidates14_0_checked : candidates edge 4 59 167 innerTail14_1 best.bound = innerCandidates14_0 := by
  unfold candidates
  have row : edge 167 = PrimePairCache.smallRow167.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild14_0 : search edge 3 [59, 167] innerCandidates14_0 226 best = best := by decide

theorem innerScan14_0 : scan edge (search edge 3) 4 [59] 59 innerTail14_0 best = best := by
  rw [innerTail14_0, scan, if_neg (by decide), innerCandidates14_0_checked]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_1
    (if 3 ≤ innerCandidates14_0.length then search edge 3 [59, 167] innerCandidates14_0 226 best else best) = best
  rw [if_pos (by decide), innerChild14_0]
  exact innerScan14_1

theorem rootBranch14 : search edge 4 [59] rootCandidates14 59 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [59] 59 innerTail14_0 best = best
  exact innerScan14_0

end ProjectEulerSolutions.PrimePairUpper.Search
