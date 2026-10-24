import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates1 : List Nat := [19, 61, 97, 109, 127, 229, 283, 433, 487, 523, 541, 547, 673, 691, 757, 823, 829, 853, 883, 937, 1171, 1237, 1249, 1399, 1453, 1471, 1549, 1693, 1741, 1861, 1879, 1993, 1999, 2089, 2161, 2269, 2287, 2341, 2467, 2503, 2617, 2671, 2707, 2719, 2953, 3019, 3181, 3331, 3361, 3433, 3583, 3613, 3709, 3727, 3823, 4027, 4159, 4177, 4201, 4219, 4357, 4441, 4507, 4567, 4729, 4903, 5437, 5521, 5641, 5659, 5689, 5821, 6091, 6163, 6421, 6481, 6679, 6949, 7351, 7417, 7489, 7723, 7933, 8017, 8101, 8167, 8311, 8317]
set_option maxRecDepth 4096 in
theorem rootCandidates1_checked : candidates edge 5 0 7 tail2 best.bound = rootCandidates1 := by
  unfold candidates
  have row : edge 7 = PrimePairCache.smallRow7.contains := by funext q; rfl
  rw [row]
  decide

def innerTail1_88 : List Nat := []
theorem innerScan1_88 : scan edge (search edge 3) 4 [7] 7 innerTail1_88 best = best := rfl

def innerTail1_87 : List Nat := 8317 :: innerTail1_88
def innerCandidates1_87 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates1_87_checked : candidates edge 4 7 8317 innerTail1_88 best.bound = innerCandidates1_87 := by
  unfold candidates
  have row : edge 8317 = SearchData.row8317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_87 : scan edge (search edge 3) 4 [7] 7 innerTail1_87 best = best := by
  rw [innerTail1_87, scan, if_neg (by decide), innerCandidates1_87_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_88
    (if 3 ≤ innerCandidates1_87.length then search edge 3 [7, 8317] innerCandidates1_87 8324 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_88

def innerTail1_86 : List Nat := 8311 :: innerTail1_87
def innerCandidates1_86 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates1_86_checked : candidates edge 4 7 8311 innerTail1_87 best.bound = innerCandidates1_86 := by
  unfold candidates
  have row : edge 8311 = SearchData.row8311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_86 : scan edge (search edge 3) 4 [7] 7 innerTail1_86 best = best := by
  rw [innerTail1_86, scan, if_neg (by decide), innerCandidates1_86_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_87
    (if 3 ≤ innerCandidates1_86.length then search edge 3 [7, 8311] innerCandidates1_86 8318 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_87

def innerTail1_85 : List Nat := 8167 :: innerTail1_86
def innerCandidates1_85 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates1_85_checked : candidates edge 4 7 8167 innerTail1_86 best.bound = innerCandidates1_85 := by
  unfold candidates
  have row : edge 8167 = SearchData.row8167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_85 : scan edge (search edge 3) 4 [7] 7 innerTail1_85 best = best := by
  rw [innerTail1_85, scan, if_neg (by decide), innerCandidates1_85_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_86
    (if 3 ≤ innerCandidates1_85.length then search edge 3 [7, 8167] innerCandidates1_85 8174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_86

def innerTail1_84 : List Nat := 8101 :: innerTail1_85
def innerCandidates1_84 : List Nat := [8317]
set_option maxRecDepth 4096 in
theorem innerCandidates1_84_checked : candidates edge 4 7 8101 innerTail1_85 best.bound = innerCandidates1_84 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_84 : scan edge (search edge 3) 4 [7] 7 innerTail1_84 best = best := by
  rw [innerTail1_84, scan, if_neg (by decide), innerCandidates1_84_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_85
    (if 3 ≤ innerCandidates1_84.length then search edge 3 [7, 8101] innerCandidates1_84 8108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_85

def innerTail1_83 : List Nat := 8017 :: innerTail1_84
def innerCandidates1_83 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates1_83_checked : candidates edge 4 7 8017 innerTail1_84 best.bound = innerCandidates1_83 := by
  unfold candidates
  have row : edge 8017 = SearchData.row8017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_83 : scan edge (search edge 3) 4 [7] 7 innerTail1_83 best = best := by
  rw [innerTail1_83, scan, if_neg (by decide), innerCandidates1_83_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_84
    (if 3 ≤ innerCandidates1_83.length then search edge 3 [7, 8017] innerCandidates1_83 8024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_84

def innerTail1_82 : List Nat := 7933 :: innerTail1_83
def innerCandidates1_82 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates1_82_checked : candidates edge 4 7 7933 innerTail1_83 best.bound = innerCandidates1_82 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_82 : scan edge (search edge 3) 4 [7] 7 innerTail1_82 best = best := by
  rw [innerTail1_82, scan, if_neg (by decide), innerCandidates1_82_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_83
    (if 3 ≤ innerCandidates1_82.length then search edge 3 [7, 7933] innerCandidates1_82 7940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_83

def innerTail1_81 : List Nat := 7723 :: innerTail1_82
def innerCandidates1_81 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates1_81_checked : candidates edge 4 7 7723 innerTail1_82 best.bound = innerCandidates1_81 := by
  unfold candidates
  have row : edge 7723 = SearchData.row7723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_81 : scan edge (search edge 3) 4 [7] 7 innerTail1_81 best = best := by
  rw [innerTail1_81, scan, if_neg (by decide), innerCandidates1_81_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_82
    (if 3 ≤ innerCandidates1_81.length then search edge 3 [7, 7723] innerCandidates1_81 7730 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_82

def innerTail1_80 : List Nat := 7489 :: innerTail1_81
def innerCandidates1_80 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates1_80_checked : candidates edge 4 7 7489 innerTail1_81 best.bound = innerCandidates1_80 := by
  unfold candidates
  have row : edge 7489 = SearchData.row7489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_80 : scan edge (search edge 3) 4 [7] 7 innerTail1_80 best = best := by
  rw [innerTail1_80, scan, if_neg (by decide), innerCandidates1_80_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_81
    (if 3 ≤ innerCandidates1_80.length then search edge 3 [7, 7489] innerCandidates1_80 7496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_81

def innerTail1_79 : List Nat := 7417 :: innerTail1_80
def innerCandidates1_79 : List Nat := [8017]
set_option maxRecDepth 4096 in
theorem innerCandidates1_79_checked : candidates edge 4 7 7417 innerTail1_80 best.bound = innerCandidates1_79 := by
  unfold candidates
  have row : edge 7417 = SearchData.row7417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_79 : scan edge (search edge 3) 4 [7] 7 innerTail1_79 best = best := by
  rw [innerTail1_79, scan, if_neg (by decide), innerCandidates1_79_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_80
    (if 3 ≤ innerCandidates1_79.length then search edge 3 [7, 7417] innerCandidates1_79 7424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_80

def innerTail1_78 : List Nat := 7351 :: innerTail1_79
def innerCandidates1_78 : List Nat := [7723]
set_option maxRecDepth 4096 in
theorem innerCandidates1_78_checked : candidates edge 4 7 7351 innerTail1_79 best.bound = innerCandidates1_78 := by
  unfold candidates
  have row : edge 7351 = SearchData.row7351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_78 : scan edge (search edge 3) 4 [7] 7 innerTail1_78 best = best := by
  rw [innerTail1_78, scan, if_neg (by decide), innerCandidates1_78_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_79
    (if 3 ≤ innerCandidates1_78.length then search edge 3 [7, 7351] innerCandidates1_78 7358 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_79

def innerTail1_77 : List Nat := 6949 :: innerTail1_78
def innerCandidates1_77 : List Nat := [7723]
set_option maxRecDepth 4096 in
theorem innerCandidates1_77_checked : candidates edge 4 7 6949 innerTail1_78 best.bound = innerCandidates1_77 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_77 : scan edge (search edge 3) 4 [7] 7 innerTail1_77 best = best := by
  rw [innerTail1_77, scan, if_neg (by decide), innerCandidates1_77_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_78
    (if 3 ≤ innerCandidates1_77.length then search edge 3 [7, 6949] innerCandidates1_77 6956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_78

def innerTail1_76 : List Nat := 6679 :: innerTail1_77
def innerCandidates1_76 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates1_76_checked : candidates edge 4 7 6679 innerTail1_77 best.bound = innerCandidates1_76 := by
  unfold candidates
  have row : edge 6679 = SearchData.row6679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_76 : scan edge (search edge 3) 4 [7] 7 innerTail1_76 best = best := by
  rw [innerTail1_76, scan, if_neg (by decide), innerCandidates1_76_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_77
    (if 3 ≤ innerCandidates1_76.length then search edge 3 [7, 6679] innerCandidates1_76 6686 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_77

def innerTail1_75 : List Nat := 6481 :: innerTail1_76
def innerCandidates1_75 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates1_75_checked : candidates edge 4 7 6481 innerTail1_76 best.bound = innerCandidates1_75 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_75 : scan edge (search edge 3) 4 [7] 7 innerTail1_75 best = best := by
  rw [innerTail1_75, scan, if_neg (by decide), innerCandidates1_75_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_76
    (if 3 ≤ innerCandidates1_75.length then search edge 3 [7, 6481] innerCandidates1_75 6488 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_76

def innerTail1_74 : List Nat := 6421 :: innerTail1_75
def innerCandidates1_74 : List Nat := [6949]
set_option maxRecDepth 4096 in
theorem innerCandidates1_74_checked : candidates edge 4 7 6421 innerTail1_75 best.bound = innerCandidates1_74 := by
  unfold candidates
  have row : edge 6421 = SearchData.row6421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_74 : scan edge (search edge 3) 4 [7] 7 innerTail1_74 best = best := by
  rw [innerTail1_74, scan, if_neg (by decide), innerCandidates1_74_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_75
    (if 3 ≤ innerCandidates1_74.length then search edge 3 [7, 6421] innerCandidates1_74 6428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_75

def innerTail1_73 : List Nat := 6163 :: innerTail1_74
def innerCandidates1_73 : List Nat := [8167]
set_option maxRecDepth 4096 in
theorem innerCandidates1_73_checked : candidates edge 4 7 6163 innerTail1_74 best.bound = innerCandidates1_73 := by
  unfold candidates
  have row : edge 6163 = SearchData.row6163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_73 : scan edge (search edge 3) 4 [7] 7 innerTail1_73 best = best := by
  rw [innerTail1_73, scan, if_neg (by decide), innerCandidates1_73_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_74
    (if 3 ≤ innerCandidates1_73.length then search edge 3 [7, 6163] innerCandidates1_73 6170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_74

def innerTail1_72 : List Nat := 6091 :: innerTail1_73
def innerCandidates1_72 : List Nat := [8311]
set_option maxRecDepth 4096 in
theorem innerCandidates1_72_checked : candidates edge 4 7 6091 innerTail1_73 best.bound = innerCandidates1_72 := by
  unfold candidates
  have row : edge 6091 = SearchData.row6091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_72 : scan edge (search edge 3) 4 [7] 7 innerTail1_72 best = best := by
  rw [innerTail1_72, scan, if_neg (by decide), innerCandidates1_72_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_73
    (if 3 ≤ innerCandidates1_72.length then search edge 3 [7, 6091] innerCandidates1_72 6098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_73

def innerTail1_71 : List Nat := 5821 :: innerTail1_72
def innerCandidates1_71 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates1_71_checked : candidates edge 4 7 5821 innerTail1_72 best.bound = innerCandidates1_71 := by
  unfold candidates
  have row : edge 5821 = SearchData.row5821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_71 : scan edge (search edge 3) 4 [7] 7 innerTail1_71 best = best := by
  rw [innerTail1_71, scan, if_neg (by decide), innerCandidates1_71_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_72
    (if 3 ≤ innerCandidates1_71.length then search edge 3 [7, 5821] innerCandidates1_71 5828 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_72

def innerTail1_70 : List Nat := 5689 :: innerTail1_71
def innerCandidates1_70 : List Nat := [6163]
set_option maxRecDepth 4096 in
theorem innerCandidates1_70_checked : candidates edge 4 7 5689 innerTail1_71 best.bound = innerCandidates1_70 := by
  unfold candidates
  have row : edge 5689 = SearchData.row5689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_70 : scan edge (search edge 3) 4 [7] 7 innerTail1_70 best = best := by
  rw [innerTail1_70, scan, if_neg (by decide), innerCandidates1_70_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_71
    (if 3 ≤ innerCandidates1_70.length then search edge 3 [7, 5689] innerCandidates1_70 5696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_71

def innerTail1_69 : List Nat := 5659 :: innerTail1_70
def innerCandidates1_69 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates1_69_checked : candidates edge 4 7 5659 innerTail1_70 best.bound = innerCandidates1_69 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_69 : scan edge (search edge 3) 4 [7] 7 innerTail1_69 best = best := by
  rw [innerTail1_69, scan, if_neg (by decide), innerCandidates1_69_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_70
    (if 3 ≤ innerCandidates1_69.length then search edge 3 [7, 5659] innerCandidates1_69 5666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_70

def innerTail1_68 : List Nat := 5641 :: innerTail1_69
def innerCandidates1_68 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates1_68_checked : candidates edge 4 7 5641 innerTail1_69 best.bound = innerCandidates1_68 := by
  unfold candidates
  have row : edge 5641 = SearchData.row5641.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_68 : scan edge (search edge 3) 4 [7] 7 innerTail1_68 best = best := by
  rw [innerTail1_68, scan, if_neg (by decide), innerCandidates1_68_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_69
    (if 3 ≤ innerCandidates1_68.length then search edge 3 [7, 5641] innerCandidates1_68 5648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_69

def innerTail1_67 : List Nat := 5521 :: innerTail1_68
def innerCandidates1_67 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates1_67_checked : candidates edge 4 7 5521 innerTail1_68 best.bound = innerCandidates1_67 := by
  unfold candidates
  have row : edge 5521 = SearchData.row5521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_67 : scan edge (search edge 3) 4 [7] 7 innerTail1_67 best = best := by
  rw [innerTail1_67, scan, if_neg (by decide), innerCandidates1_67_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_68
    (if 3 ≤ innerCandidates1_67.length then search edge 3 [7, 5521] innerCandidates1_67 5528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_68

def innerTail1_66 : List Nat := 5437 :: innerTail1_67
def innerCandidates1_66 : List Nat := [5689, 8017]
set_option maxRecDepth 4096 in
theorem innerCandidates1_66_checked : candidates edge 4 7 5437 innerTail1_67 best.bound = innerCandidates1_66 := by
  unfold candidates
  have row : edge 5437 = SearchData.row5437.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_66 : scan edge (search edge 3) 4 [7] 7 innerTail1_66 best = best := by
  rw [innerTail1_66, scan, if_neg (by decide), innerCandidates1_66_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_67
    (if 3 ≤ innerCandidates1_66.length then search edge 3 [7, 5437] innerCandidates1_66 5444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_67

def innerTail1_65 : List Nat := 4903 :: innerTail1_66
def innerCandidates1_65 : List Nat := [5659]
set_option maxRecDepth 4096 in
theorem innerCandidates1_65_checked : candidates edge 4 7 4903 innerTail1_66 best.bound = innerCandidates1_65 := by
  unfold candidates
  have row : edge 4903 = SearchData.row4903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_65 : scan edge (search edge 3) 4 [7] 7 innerTail1_65 best = best := by
  rw [innerTail1_65, scan, if_neg (by decide), innerCandidates1_65_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_66
    (if 3 ≤ innerCandidates1_65.length then search edge 3 [7, 4903] innerCandidates1_65 4910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_66

def innerTail1_64 : List Nat := 4729 :: innerTail1_65
def innerCandidates1_64 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates1_64_checked : candidates edge 4 7 4729 innerTail1_65 best.bound = innerCandidates1_64 := by
  unfold candidates
  have row : edge 4729 = SearchData.row4729.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_64 : scan edge (search edge 3) 4 [7] 7 innerTail1_64 best = best := by
  rw [innerTail1_64, scan, if_neg (by decide), innerCandidates1_64_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_65
    (if 3 ≤ innerCandidates1_64.length then search edge 3 [7, 4729] innerCandidates1_64 4736 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_65

def innerTail1_63 : List Nat := 4567 :: innerTail1_64
def innerCandidates1_63 : List Nat := [7417]
set_option maxRecDepth 4096 in
theorem innerCandidates1_63_checked : candidates edge 4 7 4567 innerTail1_64 best.bound = innerCandidates1_63 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_63 : scan edge (search edge 3) 4 [7] 7 innerTail1_63 best = best := by
  rw [innerTail1_63, scan, if_neg (by decide), innerCandidates1_63_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_64
    (if 3 ≤ innerCandidates1_63.length then search edge 3 [7, 4567] innerCandidates1_63 4574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_64

def innerTail1_62 : List Nat := 4507 :: innerTail1_63
def innerCandidates1_62 : List Nat := [6091]
set_option maxRecDepth 4096 in
theorem innerCandidates1_62_checked : candidates edge 4 7 4507 innerTail1_63 best.bound = innerCandidates1_62 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_62 : scan edge (search edge 3) 4 [7] 7 innerTail1_62 best = best := by
  rw [innerTail1_62, scan, if_neg (by decide), innerCandidates1_62_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_63
    (if 3 ≤ innerCandidates1_62.length then search edge 3 [7, 4507] innerCandidates1_62 4514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_63

def innerTail1_61 : List Nat := 4441 :: innerTail1_62
def innerCandidates1_61 : List Nat := [6481, 6949]
set_option maxRecDepth 4096 in
theorem innerCandidates1_61_checked : candidates edge 4 7 4441 innerTail1_62 best.bound = innerCandidates1_61 := by
  unfold candidates
  have row : edge 4441 = SearchData.row4441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_61 : scan edge (search edge 3) 4 [7] 7 innerTail1_61 best = best := by
  rw [innerTail1_61, scan, if_neg (by decide), innerCandidates1_61_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_62
    (if 3 ≤ innerCandidates1_61.length then search edge 3 [7, 4441] innerCandidates1_61 4448 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_62

def innerTail1_60 : List Nat := 4357 :: innerTail1_61
def innerCandidates1_60 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates1_60_checked : candidates edge 4 7 4357 innerTail1_61 best.bound = innerCandidates1_60 := by
  unfold candidates
  have row : edge 4357 = SearchData.row4357.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_60 : scan edge (search edge 3) 4 [7] 7 innerTail1_60 best = best := by
  rw [innerTail1_60, scan, if_neg (by decide), innerCandidates1_60_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_61
    (if 3 ≤ innerCandidates1_60.length then search edge 3 [7, 4357] innerCandidates1_60 4364 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_61

def innerTail1_59 : List Nat := 4219 :: innerTail1_60
def innerCandidates1_59 : List Nat := [5521]
set_option maxRecDepth 4096 in
theorem innerCandidates1_59_checked : candidates edge 4 7 4219 innerTail1_60 best.bound = innerCandidates1_59 := by
  unfold candidates
  have row : edge 4219 = SearchData.row4219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_59 : scan edge (search edge 3) 4 [7] 7 innerTail1_59 best = best := by
  rw [innerTail1_59, scan, if_neg (by decide), innerCandidates1_59_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_60
    (if 3 ≤ innerCandidates1_59.length then search edge 3 [7, 4219] innerCandidates1_59 4226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_60

def innerTail1_58 : List Nat := 4201 :: innerTail1_59
def innerCandidates1_58 : List Nat := [6421]
set_option maxRecDepth 4096 in
theorem innerCandidates1_58_checked : candidates edge 4 7 4201 innerTail1_59 best.bound = innerCandidates1_58 := by
  unfold candidates
  have row : edge 4201 = SearchData.row4201.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_58 : scan edge (search edge 3) 4 [7] 7 innerTail1_58 best = best := by
  rw [innerTail1_58, scan, if_neg (by decide), innerCandidates1_58_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_59
    (if 3 ≤ innerCandidates1_58.length then search edge 3 [7, 4201] innerCandidates1_58 4208 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_59

def innerTail1_57 : List Nat := 4177 :: innerTail1_58
def innerCandidates1_57 : List Nat := [4903, 8101, 8311]
set_option maxRecDepth 4096 in
theorem innerCandidates1_57_checked : candidates edge 4 7 4177 innerTail1_58 best.bound = innerCandidates1_57 := by
  unfold candidates
  have row : edge 4177 = SearchData.row4177.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_57 : search edge 3 [7, 4177] innerCandidates1_57 4184 best = best := by decide

theorem innerScan1_57 : scan edge (search edge 3) 4 [7] 7 innerTail1_57 best = best := by
  rw [innerTail1_57, scan, if_neg (by decide), innerCandidates1_57_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_58
    (if 3 ≤ innerCandidates1_57.length then search edge 3 [7, 4177] innerCandidates1_57 4184 best else best) = best
  rw [if_pos (by decide), innerChild1_57]
  exact innerScan1_58

def innerTail1_56 : List Nat := 4159 :: innerTail1_57
def innerCandidates1_56 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates1_56_checked : candidates edge 4 7 4159 innerTail1_57 best.bound = innerCandidates1_56 := by
  unfold candidates
  have row : edge 4159 = SearchData.row4159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_56 : scan edge (search edge 3) 4 [7] 7 innerTail1_56 best = best := by
  rw [innerTail1_56, scan, if_neg (by decide), innerCandidates1_56_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_57
    (if 3 ≤ innerCandidates1_56.length then search edge 3 [7, 4159] innerCandidates1_56 4166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_57

def innerTail1_55 : List Nat := 4027 :: innerTail1_56
def innerCandidates1_55 : List Nat := [6949, 7417]
set_option maxRecDepth 4096 in
theorem innerCandidates1_55_checked : candidates edge 4 7 4027 innerTail1_56 best.bound = innerCandidates1_55 := by
  unfold candidates
  have row : edge 4027 = SearchData.row4027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_55 : scan edge (search edge 3) 4 [7] 7 innerTail1_55 best = best := by
  rw [innerTail1_55, scan, if_neg (by decide), innerCandidates1_55_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_56
    (if 3 ≤ innerCandidates1_55.length then search edge 3 [7, 4027] innerCandidates1_55 4034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_56

def innerTail1_54 : List Nat := 3823 :: innerTail1_55
def innerCandidates1_54 : List Nat := [4159, 4201, 7417]
set_option maxRecDepth 4096 in
theorem innerCandidates1_54_checked : candidates edge 4 7 3823 innerTail1_55 best.bound = innerCandidates1_54 := by
  unfold candidates
  have row : edge 3823 = SearchData.row3823.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_54 : search edge 3 [7, 3823] innerCandidates1_54 3830 best = best := by decide

theorem innerScan1_54 : scan edge (search edge 3) 4 [7] 7 innerTail1_54 best = best := by
  rw [innerTail1_54, scan, if_neg (by decide), innerCandidates1_54_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_55
    (if 3 ≤ innerCandidates1_54.length then search edge 3 [7, 3823] innerCandidates1_54 3830 best else best) = best
  rw [if_pos (by decide), innerChild1_54]
  exact innerScan1_55

def innerTail1_53 : List Nat := 3727 :: innerTail1_54
def innerCandidates1_53 : List Nat := [5659, 6949]
set_option maxRecDepth 4096 in
theorem innerCandidates1_53_checked : candidates edge 4 7 3727 innerTail1_54 best.bound = innerCandidates1_53 := by
  unfold candidates
  have row : edge 3727 = SearchData.row3727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_53 : scan edge (search edge 3) 4 [7] 7 innerTail1_53 best = best := by
  rw [innerTail1_53, scan, if_neg (by decide), innerCandidates1_53_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_54
    (if 3 ≤ innerCandidates1_53.length then search edge 3 [7, 3727] innerCandidates1_53 3734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_54

def innerTail1_52 : List Nat := 3709 :: innerTail1_53
def innerCandidates1_52 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates1_52_checked : candidates edge 4 7 3709 innerTail1_53 best.bound = innerCandidates1_52 := by
  unfold candidates
  have row : edge 3709 = SearchData.row3709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_52 : scan edge (search edge 3) 4 [7] 7 innerTail1_52 best = best := by
  rw [innerTail1_52, scan, if_neg (by decide), innerCandidates1_52_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_53
    (if 3 ≤ innerCandidates1_52.length then search edge 3 [7, 3709] innerCandidates1_52 3716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_53

def innerTail1_51 : List Nat := 3613 :: innerTail1_52
def innerCandidates1_51 : List Nat := [5821]
set_option maxRecDepth 4096 in
theorem innerCandidates1_51_checked : candidates edge 4 7 3613 innerTail1_52 best.bound = innerCandidates1_51 := by
  unfold candidates
  have row : edge 3613 = SearchData.row3613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_51 : scan edge (search edge 3) 4 [7] 7 innerTail1_51 best = best := by
  rw [innerTail1_51, scan, if_neg (by decide), innerCandidates1_51_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_52
    (if 3 ≤ innerCandidates1_51.length then search edge 3 [7, 3613] innerCandidates1_51 3620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_52

def innerTail1_50 : List Nat := 3583 :: innerTail1_51
def innerCandidates1_50 : List Nat := [4177, 5689]
set_option maxRecDepth 4096 in
theorem innerCandidates1_50_checked : candidates edge 4 7 3583 innerTail1_51 best.bound = innerCandidates1_50 := by
  unfold candidates
  have row : edge 3583 = SearchData.row3583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_50 : scan edge (search edge 3) 4 [7] 7 innerTail1_50 best = best := by
  rw [innerTail1_50, scan, if_neg (by decide), innerCandidates1_50_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_51
    (if 3 ≤ innerCandidates1_50.length then search edge 3 [7, 3583] innerCandidates1_50 3590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_51

def innerTail1_49 : List Nat := 3433 :: innerTail1_50
def innerCandidates1_49 : List Nat := [8167]
set_option maxRecDepth 4096 in
theorem innerCandidates1_49_checked : candidates edge 4 7 3433 innerTail1_50 best.bound = innerCandidates1_49 := by
  unfold candidates
  have row : edge 3433 = SearchData.row3433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_49 : scan edge (search edge 3) 4 [7] 7 innerTail1_49 best = best := by
  rw [innerTail1_49, scan, if_neg (by decide), innerCandidates1_49_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_50
    (if 3 ≤ innerCandidates1_49.length then search edge 3 [7, 3433] innerCandidates1_49 3440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_50

def innerTail1_48 : List Nat := 3361 :: innerTail1_49
def innerCandidates1_48 : List Nat := [4507]
set_option maxRecDepth 4096 in
theorem innerCandidates1_48_checked : candidates edge 4 7 3361 innerTail1_49 best.bound = innerCandidates1_48 := by
  unfold candidates
  have row : edge 3361 = SearchData.row3361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_48 : scan edge (search edge 3) 4 [7] 7 innerTail1_48 best = best := by
  rw [innerTail1_48, scan, if_neg (by decide), innerCandidates1_48_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_49
    (if 3 ≤ innerCandidates1_48.length then search edge 3 [7, 3361] innerCandidates1_48 3368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_49

def innerTail1_47 : List Nat := 3331 :: innerTail1_48
def innerCandidates1_47 : List Nat := [4903]
set_option maxRecDepth 4096 in
theorem innerCandidates1_47_checked : candidates edge 4 7 3331 innerTail1_48 best.bound = innerCandidates1_47 := by
  unfold candidates
  have row : edge 3331 = SearchData.row3331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_47 : scan edge (search edge 3) 4 [7] 7 innerTail1_47 best = best := by
  rw [innerTail1_47, scan, if_neg (by decide), innerCandidates1_47_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_48
    (if 3 ≤ innerCandidates1_47.length then search edge 3 [7, 3331] innerCandidates1_47 3338 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_48

def innerTail1_46 : List Nat := 3181 :: innerTail1_47
def innerCandidates1_46 : List Nat := [4219, 4507, 4903, 6481]
set_option maxRecDepth 4096 in
theorem innerCandidates1_46_checked : candidates edge 4 7 3181 innerTail1_47 best.bound = innerCandidates1_46 := by
  unfold candidates
  have row : edge 3181 = SearchData.row3181.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_46 : search edge 3 [7, 3181] innerCandidates1_46 3188 best = best := by decide

theorem innerScan1_46 : scan edge (search edge 3) 4 [7] 7 innerTail1_46 best = best := by
  rw [innerTail1_46, scan, if_neg (by decide), innerCandidates1_46_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_47
    (if 3 ≤ innerCandidates1_46.length then search edge 3 [7, 3181] innerCandidates1_46 3188 best else best) = best
  rw [if_pos (by decide), innerChild1_46]
  exact innerScan1_47

def innerTail1_45 : List Nat := 3019 :: innerTail1_46
def innerCandidates1_45 : List Nat := [7351, 7417, 8317]
set_option maxRecDepth 4096 in
theorem innerCandidates1_45_checked : candidates edge 4 7 3019 innerTail1_46 best.bound = innerCandidates1_45 := by
  unfold candidates
  have row : edge 3019 = SearchData.row3019.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_45 : search edge 3 [7, 3019] innerCandidates1_45 3026 best = best := by decide

theorem innerScan1_45 : scan edge (search edge 3) 4 [7] 7 innerTail1_45 best = best := by
  rw [innerTail1_45, scan, if_neg (by decide), innerCandidates1_45_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_46
    (if 3 ≤ innerCandidates1_45.length then search edge 3 [7, 3019] innerCandidates1_45 3026 best else best) = best
  rw [if_pos (by decide), innerChild1_45]
  exact innerScan1_46

def innerTail1_44 : List Nat := 2953 :: innerTail1_45
def innerCandidates1_44 : List Nat := [3181, 3433, 4357, 6679]
set_option maxRecDepth 4096 in
theorem innerCandidates1_44_checked : candidates edge 4 7 2953 innerTail1_45 best.bound = innerCandidates1_44 := by
  unfold candidates
  have row : edge 2953 = SearchData.row2953.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_44 : search edge 3 [7, 2953] innerCandidates1_44 2960 best = best := by decide

theorem innerScan1_44 : scan edge (search edge 3) 4 [7] 7 innerTail1_44 best = best := by
  rw [innerTail1_44, scan, if_neg (by decide), innerCandidates1_44_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_45
    (if 3 ≤ innerCandidates1_44.length then search edge 3 [7, 2953] innerCandidates1_44 2960 best else best) = best
  rw [if_pos (by decide), innerChild1_44]
  exact innerScan1_45

def innerTail1_43 : List Nat := 2719 :: innerTail1_44
def innerCandidates1_43 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates1_43_checked : candidates edge 4 7 2719 innerTail1_44 best.bound = innerCandidates1_43 := by
  unfold candidates
  have row : edge 2719 = SearchData.row2719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_43 : scan edge (search edge 3) 4 [7] 7 innerTail1_43 best = best := by
  rw [innerTail1_43, scan, if_neg (by decide), innerCandidates1_43_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_44
    (if 3 ≤ innerCandidates1_43.length then search edge 3 [7, 2719] innerCandidates1_43 2726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_44

def innerTail1_42 : List Nat := 2707 :: innerTail1_43
def innerCandidates1_42 : List Nat := [4507]
set_option maxRecDepth 4096 in
theorem innerCandidates1_42_checked : candidates edge 4 7 2707 innerTail1_43 best.bound = innerCandidates1_42 := by
  unfold candidates
  have row : edge 2707 = SearchData.row2707.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_42 : scan edge (search edge 3) 4 [7] 7 innerTail1_42 best = best := by
  rw [innerTail1_42, scan, if_neg (by decide), innerCandidates1_42_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_43
    (if 3 ≤ innerCandidates1_42.length then search edge 3 [7, 2707] innerCandidates1_42 2714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_43

def innerTail1_41 : List Nat := 2671 :: innerTail1_42
def innerCandidates1_41 : List Nat := [3361, 7723]
set_option maxRecDepth 4096 in
theorem innerCandidates1_41_checked : candidates edge 4 7 2671 innerTail1_42 best.bound = innerCandidates1_41 := by
  unfold candidates
  have row : edge 2671 = SearchData.row2671.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_41 : scan edge (search edge 3) 4 [7] 7 innerTail1_41 best = best := by
  rw [innerTail1_41, scan, if_neg (by decide), innerCandidates1_41_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_42
    (if 3 ≤ innerCandidates1_41.length then search edge 3 [7, 2671] innerCandidates1_41 2678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_42

def innerTail1_40 : List Nat := 2617 :: innerTail1_41
def innerCandidates1_40 : List Nat := [3181, 4567, 5689]
set_option maxRecDepth 4096 in
theorem innerCandidates1_40_checked : candidates edge 4 7 2617 innerTail1_41 best.bound = innerCandidates1_40 := by
  unfold candidates
  have row : edge 2617 = PrimePairCache.smallRow2617.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_40 : search edge 3 [7, 2617] innerCandidates1_40 2624 best = best := by decide

theorem innerScan1_40 : scan edge (search edge 3) 4 [7] 7 innerTail1_40 best = best := by
  rw [innerTail1_40, scan, if_neg (by decide), innerCandidates1_40_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_41
    (if 3 ≤ innerCandidates1_40.length then search edge 3 [7, 2617] innerCandidates1_40 2624 best else best) = best
  rw [if_pos (by decide), innerChild1_40]
  exact innerScan1_41

def innerTail1_39 : List Nat := 2503 :: innerTail1_40
def innerCandidates1_39 : List Nat := [6163, 6421]
set_option maxRecDepth 4096 in
theorem innerCandidates1_39_checked : candidates edge 4 7 2503 innerTail1_40 best.bound = innerCandidates1_39 := by
  unfold candidates
  have row : edge 2503 = PrimePairCache.smallRow2503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_39 : scan edge (search edge 3) 4 [7] 7 innerTail1_39 best = best := by
  rw [innerTail1_39, scan, if_neg (by decide), innerCandidates1_39_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_40
    (if 3 ≤ innerCandidates1_39.length then search edge 3 [7, 2503] innerCandidates1_39 2510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_40

def innerTail1_38 : List Nat := 2467 :: innerTail1_39
def innerCandidates1_38 : List Nat := [5689, 8167]
set_option maxRecDepth 4096 in
theorem innerCandidates1_38_checked : candidates edge 4 7 2467 innerTail1_39 best.bound = innerCandidates1_38 := by
  unfold candidates
  have row : edge 2467 = PrimePairCache.smallRow2467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_38 : scan edge (search edge 3) 4 [7] 7 innerTail1_38 best = best := by
  rw [innerTail1_38, scan, if_neg (by decide), innerCandidates1_38_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_39
    (if 3 ≤ innerCandidates1_38.length then search edge 3 [7, 2467] innerCandidates1_38 2474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_39

def innerTail1_37 : List Nat := 2341 :: innerTail1_38
def innerCandidates1_37 : List Nat := [7489]
set_option maxRecDepth 4096 in
theorem innerCandidates1_37_checked : candidates edge 4 7 2341 innerTail1_38 best.bound = innerCandidates1_37 := by
  unfold candidates
  have row : edge 2341 = PrimePairCache.smallRow2341.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_37 : scan edge (search edge 3) 4 [7] 7 innerTail1_37 best = best := by
  rw [innerTail1_37, scan, if_neg (by decide), innerCandidates1_37_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_38
    (if 3 ≤ innerCandidates1_37.length then search edge 3 [7, 2341] innerCandidates1_37 2348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_38

def innerTail1_36 : List Nat := 2287 :: innerTail1_37
def innerCandidates1_36 : List Nat := [2341, 8101]
set_option maxRecDepth 4096 in
theorem innerCandidates1_36_checked : candidates edge 4 7 2287 innerTail1_37 best.bound = innerCandidates1_36 := by
  unfold candidates
  have row : edge 2287 = PrimePairCache.smallRow2287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_36 : scan edge (search edge 3) 4 [7] 7 innerTail1_36 best = best := by
  rw [innerTail1_36, scan, if_neg (by decide), innerCandidates1_36_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_37
    (if 3 ≤ innerCandidates1_36.length then search edge 3 [7, 2287] innerCandidates1_36 2294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_37

def innerTail1_35 : List Nat := 2269 :: innerTail1_36
def innerCandidates1_35 : List Nat := [3613, 5821, 7489]
set_option maxRecDepth 4096 in
theorem innerCandidates1_35_checked : candidates edge 4 7 2269 innerTail1_36 best.bound = innerCandidates1_35 := by
  unfold candidates
  have row : edge 2269 = PrimePairCache.smallRow2269.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_35 : search edge 3 [7, 2269] innerCandidates1_35 2276 best = best := by decide

theorem innerScan1_35 : scan edge (search edge 3) 4 [7] 7 innerTail1_35 best = best := by
  rw [innerTail1_35, scan, if_neg (by decide), innerCandidates1_35_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_36
    (if 3 ≤ innerCandidates1_35.length then search edge 3 [7, 2269] innerCandidates1_35 2276 best else best) = best
  rw [if_pos (by decide), innerChild1_35]
  exact innerScan1_36

def innerTail1_34 : List Nat := 2161 :: innerTail1_35
def innerCandidates1_34 : List Nat := [4357, 6949]
set_option maxRecDepth 4096 in
theorem innerCandidates1_34_checked : candidates edge 4 7 2161 innerTail1_35 best.bound = innerCandidates1_34 := by
  unfold candidates
  have row : edge 2161 = PrimePairCache.smallRow2161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_34 : scan edge (search edge 3) 4 [7] 7 innerTail1_34 best = best := by
  rw [innerTail1_34, scan, if_neg (by decide), innerCandidates1_34_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_35
    (if 3 ≤ innerCandidates1_34.length then search edge 3 [7, 2161] innerCandidates1_34 2168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_35

def innerTail1_33 : List Nat := 2089 :: innerTail1_34
def innerCandidates1_33 : List Nat := [2341, 2953, 3181, 4219]
set_option maxRecDepth 4096 in
theorem innerCandidates1_33_checked : candidates edge 4 7 2089 innerTail1_34 best.bound = innerCandidates1_33 := by
  unfold candidates
  have row : edge 2089 = PrimePairCache.smallRow2089.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_33 : search edge 3 [7, 2089] innerCandidates1_33 2096 best = best := by decide

theorem innerScan1_33 : scan edge (search edge 3) 4 [7] 7 innerTail1_33 best = best := by
  rw [innerTail1_33, scan, if_neg (by decide), innerCandidates1_33_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_34
    (if 3 ≤ innerCandidates1_33.length then search edge 3 [7, 2089] innerCandidates1_33 2096 best else best) = best
  rw [if_pos (by decide), innerChild1_33]
  exact innerScan1_34

def innerTail1_32 : List Nat := 1999 :: innerTail1_33
def innerCandidates1_32 : List Nat := [3361]
set_option maxRecDepth 4096 in
theorem innerCandidates1_32_checked : candidates edge 4 7 1999 innerTail1_33 best.bound = innerCandidates1_32 := by
  unfold candidates
  have row : edge 1999 = PrimePairCache.smallRow1999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_32 : scan edge (search edge 3) 4 [7] 7 innerTail1_32 best = best := by
  rw [innerTail1_32, scan, if_neg (by decide), innerCandidates1_32_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_33
    (if 3 ≤ innerCandidates1_32.length then search edge 3 [7, 1999] innerCandidates1_32 2006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_33

def innerTail1_31 : List Nat := 1993 :: innerTail1_32
def innerCandidates1_31 : List Nat := [6481]
set_option maxRecDepth 4096 in
theorem innerCandidates1_31_checked : candidates edge 4 7 1993 innerTail1_32 best.bound = innerCandidates1_31 := by
  unfold candidates
  have row : edge 1993 = PrimePairCache.smallRow1993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_31 : scan edge (search edge 3) 4 [7] 7 innerTail1_31 best = best := by
  rw [innerTail1_31, scan, if_neg (by decide), innerCandidates1_31_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_32
    (if 3 ≤ innerCandidates1_31.length then search edge 3 [7, 1993] innerCandidates1_31 2000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_32

def innerTail1_30 : List Nat := 1879 :: innerTail1_31
def innerCandidates1_30 : List Nat := [1999, 2341, 3331]
set_option maxRecDepth 4096 in
theorem innerCandidates1_30_checked : candidates edge 4 7 1879 innerTail1_31 best.bound = innerCandidates1_30 := by
  unfold candidates
  have row : edge 1879 = PrimePairCache.smallRow1879.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_30 : search edge 3 [7, 1879] innerCandidates1_30 1886 best = best := by decide

theorem innerScan1_30 : scan edge (search edge 3) 4 [7] 7 innerTail1_30 best = best := by
  rw [innerTail1_30, scan, if_neg (by decide), innerCandidates1_30_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_31
    (if 3 ≤ innerCandidates1_30.length then search edge 3 [7, 1879] innerCandidates1_30 1886 best else best) = best
  rw [if_pos (by decide), innerChild1_30]
  exact innerScan1_31

def innerTail1_29 : List Nat := 1861 :: innerTail1_30
def innerCandidates1_29 : List Nat := [3433, 3709, 5521]
set_option maxRecDepth 4096 in
theorem innerCandidates1_29_checked : candidates edge 4 7 1861 innerTail1_30 best.bound = innerCandidates1_29 := by
  unfold candidates
  have row : edge 1861 = PrimePairCache.smallRow1861.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_29 : search edge 3 [7, 1861] innerCandidates1_29 1868 best = best := by decide

theorem innerScan1_29 : scan edge (search edge 3) 4 [7] 7 innerTail1_29 best = best := by
  rw [innerTail1_29, scan, if_neg (by decide), innerCandidates1_29_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_30
    (if 3 ≤ innerCandidates1_29.length then search edge 3 [7, 1861] innerCandidates1_29 1868 best else best) = best
  rw [if_pos (by decide), innerChild1_29]
  exact innerScan1_30

def innerTail1_28 : List Nat := 1741 :: innerTail1_29
def innerCandidates1_28 : List Nat := [2287, 2953, 7489]
set_option maxRecDepth 4096 in
theorem innerCandidates1_28_checked : candidates edge 4 7 1741 innerTail1_29 best.bound = innerCandidates1_28 := by
  unfold candidates
  have row : edge 1741 = PrimePairCache.smallRow1741.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_28 : search edge 3 [7, 1741] innerCandidates1_28 1748 best = best := by decide

theorem innerScan1_28 : scan edge (search edge 3) 4 [7] 7 innerTail1_28 best = best := by
  rw [innerTail1_28, scan, if_neg (by decide), innerCandidates1_28_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_29
    (if 3 ≤ innerCandidates1_28.length then search edge 3 [7, 1741] innerCandidates1_28 1748 best else best) = best
  rw [if_pos (by decide), innerChild1_28]
  exact innerScan1_29

def innerTail1_27 : List Nat := 1693 :: innerTail1_28
def innerCandidates1_27 : List Nat := [3181, 4441, 7489]
set_option maxRecDepth 4096 in
theorem innerCandidates1_27_checked : candidates edge 4 7 1693 innerTail1_28 best.bound = innerCandidates1_27 := by
  unfold candidates
  have row : edge 1693 = PrimePairCache.smallRow1693.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_27 : search edge 3 [7, 1693] innerCandidates1_27 1700 best = best := by decide

theorem innerScan1_27 : scan edge (search edge 3) 4 [7] 7 innerTail1_27 best = best := by
  rw [innerTail1_27, scan, if_neg (by decide), innerCandidates1_27_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_28
    (if 3 ≤ innerCandidates1_27.length then search edge 3 [7, 1693] innerCandidates1_27 1700 best else best) = best
  rw [if_pos (by decide), innerChild1_27]
  exact innerScan1_28

def innerTail1_26 : List Nat := 1549 :: innerTail1_27
def innerCandidates1_26 : List Nat := [2707, 3019, 3823]
set_option maxRecDepth 4096 in
theorem innerCandidates1_26_checked : candidates edge 4 7 1549 innerTail1_27 best.bound = innerCandidates1_26 := by
  unfold candidates
  have row : edge 1549 = PrimePairCache.smallRow1549.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_26 : search edge 3 [7, 1549] innerCandidates1_26 1556 best = best := by decide

theorem innerScan1_26 : scan edge (search edge 3) 4 [7] 7 innerTail1_26 best = best := by
  rw [innerTail1_26, scan, if_neg (by decide), innerCandidates1_26_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_27
    (if 3 ≤ innerCandidates1_26.length then search edge 3 [7, 1549] innerCandidates1_26 1556 best else best) = best
  rw [if_pos (by decide), innerChild1_26]
  exact innerScan1_27

def innerTail1_25 : List Nat := 1471 :: innerTail1_26
def innerCandidates1_25 : List Nat := [3613, 4159, 7933]
set_option maxRecDepth 4096 in
theorem innerCandidates1_25_checked : candidates edge 4 7 1471 innerTail1_26 best.bound = innerCandidates1_25 := by
  unfold candidates
  have row : edge 1471 = PrimePairCache.smallRow1471.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_25 : search edge 3 [7, 1471] innerCandidates1_25 1478 best = best := by decide

theorem innerScan1_25 : scan edge (search edge 3) 4 [7] 7 innerTail1_25 best = best := by
  rw [innerTail1_25, scan, if_neg (by decide), innerCandidates1_25_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_26
    (if 3 ≤ innerCandidates1_25.length then search edge 3 [7, 1471] innerCandidates1_25 1478 best else best) = best
  rw [if_pos (by decide), innerChild1_25]
  exact innerScan1_26

def innerTail1_24 : List Nat := 1453 :: innerTail1_25
def innerCandidates1_24 : List Nat := [1879, 3361, 7417, 7933]
set_option maxRecDepth 4096 in
theorem innerCandidates1_24_checked : candidates edge 4 7 1453 innerTail1_25 best.bound = innerCandidates1_24 := by
  unfold candidates
  have row : edge 1453 = PrimePairCache.smallRow1453.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_24 : search edge 3 [7, 1453] innerCandidates1_24 1460 best = best := by decide

theorem innerScan1_24 : scan edge (search edge 3) 4 [7] 7 innerTail1_24 best = best := by
  rw [innerTail1_24, scan, if_neg (by decide), innerCandidates1_24_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_25
    (if 3 ≤ innerCandidates1_24.length then search edge 3 [7, 1453] innerCandidates1_24 1460 best else best) = best
  rw [if_pos (by decide), innerChild1_24]
  exact innerScan1_25

def innerTail1_23 : List Nat := 1399 :: innerTail1_24
def innerCandidates1_23 : List Nat := [1741, 3433, 6481]
set_option maxRecDepth 4096 in
theorem innerCandidates1_23_checked : candidates edge 4 7 1399 innerTail1_24 best.bound = innerCandidates1_23 := by
  unfold candidates
  have row : edge 1399 = PrimePairCache.smallRow1399.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_23 : search edge 3 [7, 1399] innerCandidates1_23 1406 best = best := by decide

theorem innerScan1_23 : scan edge (search edge 3) 4 [7] 7 innerTail1_23 best = best := by
  rw [innerTail1_23, scan, if_neg (by decide), innerCandidates1_23_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_24
    (if 3 ≤ innerCandidates1_23.length then search edge 3 [7, 1399] innerCandidates1_23 1406 best else best) = best
  rw [if_pos (by decide), innerChild1_23]
  exact innerScan1_24

def innerTail1_22 : List Nat := 1249 :: innerTail1_23
def innerCandidates1_22 : List Nat := [3727, 4441, 6949]
set_option maxRecDepth 4096 in
theorem innerCandidates1_22_checked : candidates edge 4 7 1249 innerTail1_23 best.bound = innerCandidates1_22 := by
  unfold candidates
  have row : edge 1249 = PrimePairCache.smallRow1249.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_22 : search edge 3 [7, 1249] innerCandidates1_22 1256 best = best := by decide

theorem innerScan1_22 : scan edge (search edge 3) 4 [7] 7 innerTail1_22 best = best := by
  rw [innerTail1_22, scan, if_neg (by decide), innerCandidates1_22_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_23
    (if 3 ≤ innerCandidates1_22.length then search edge 3 [7, 1249] innerCandidates1_22 1256 best else best) = best
  rw [if_pos (by decide), innerChild1_22]
  exact innerScan1_23

def innerTail1_21 : List Nat := 1237 :: innerTail1_22
def innerCandidates1_21 : List Nat := [1549, 1741, 2341, 3019, 4027]
set_option maxRecDepth 4096 in
theorem innerCandidates1_21_checked : candidates edge 4 7 1237 innerTail1_22 best.bound = innerCandidates1_21 := by
  unfold candidates
  have row : edge 1237 = PrimePairCache.smallRow1237.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_21 : search edge 3 [7, 1237] innerCandidates1_21 1244 best = best := by decide

theorem innerScan1_21 : scan edge (search edge 3) 4 [7] 7 innerTail1_21 best = best := by
  rw [innerTail1_21, scan, if_neg (by decide), innerCandidates1_21_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_22
    (if 3 ≤ innerCandidates1_21.length then search edge 3 [7, 1237] innerCandidates1_21 1244 best else best) = best
  rw [if_pos (by decide), innerChild1_21]
  exact innerScan1_22

def innerTail1_20 : List Nat := 1171 :: innerTail1_21
def innerCandidates1_20 : List Nat := [2161]
set_option maxRecDepth 4096 in
theorem innerCandidates1_20_checked : candidates edge 4 7 1171 innerTail1_21 best.bound = innerCandidates1_20 := by
  unfold candidates
  have row : edge 1171 = PrimePairCache.smallRow1171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_20 : scan edge (search edge 3) 4 [7] 7 innerTail1_20 best = best := by
  rw [innerTail1_20, scan, if_neg (by decide), innerCandidates1_20_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_21
    (if 3 ≤ innerCandidates1_20.length then search edge 3 [7, 1171] innerCandidates1_20 1178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_21

def innerTail1_19 : List Nat := 937 :: innerTail1_20
def innerCandidates1_19 : List Nat := [4159, 5437]
set_option maxRecDepth 4096 in
theorem innerCandidates1_19_checked : candidates edge 4 7 937 innerTail1_20 best.bound = innerCandidates1_19 := by
  unfold candidates
  have row : edge 937 = PrimePairCache.smallRow937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_19 : scan edge (search edge 3) 4 [7] 7 innerTail1_19 best = best := by
  rw [innerTail1_19, scan, if_neg (by decide), innerCandidates1_19_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_20
    (if 3 ≤ innerCandidates1_19.length then search edge 3 [7, 937] innerCandidates1_19 944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_20

def innerTail1_18 : List Nat := 883 :: innerTail1_19
def innerCandidates1_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates1_18_checked : candidates edge 4 7 883 innerTail1_19 best.bound = innerCandidates1_18 := by
  unfold candidates
  have row : edge 883 = PrimePairCache.smallRow883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_18 : scan edge (search edge 3) 4 [7] 7 innerTail1_18 best = best := by
  rw [innerTail1_18, scan, if_neg (by decide), innerCandidates1_18_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_19
    (if 3 ≤ innerCandidates1_18.length then search edge 3 [7, 883] innerCandidates1_18 890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_19

def innerTail1_17 : List Nat := 853 :: innerTail1_18
def innerCandidates1_17 : List Nat := [2269, 2707, 8017]
set_option maxRecDepth 4096 in
theorem innerCandidates1_17_checked : candidates edge 4 7 853 innerTail1_18 best.bound = innerCandidates1_17 := by
  unfold candidates
  have row : edge 853 = PrimePairCache.smallRow853.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_17 : search edge 3 [7, 853] innerCandidates1_17 860 best = best := by decide

theorem innerScan1_17 : scan edge (search edge 3) 4 [7] 7 innerTail1_17 best = best := by
  rw [innerTail1_17, scan, if_neg (by decide), innerCandidates1_17_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_18
    (if 3 ≤ innerCandidates1_17.length then search edge 3 [7, 853] innerCandidates1_17 860 best else best) = best
  rw [if_pos (by decide), innerChild1_17]
  exact innerScan1_18

def innerTail1_16 : List Nat := 829 :: innerTail1_17
def innerCandidates1_16 : List Nat := [1237, 2671, 3361, 7351]
set_option maxRecDepth 4096 in
theorem innerCandidates1_16_checked : candidates edge 4 7 829 innerTail1_17 best.bound = innerCandidates1_16 := by
  unfold candidates
  have row : edge 829 = PrimePairCache.smallRow829.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_16 : search edge 3 [7, 829] innerCandidates1_16 836 best = best := by decide

theorem innerScan1_16 : scan edge (search edge 3) 4 [7] 7 innerTail1_16 best = best := by
  rw [innerTail1_16, scan, if_neg (by decide), innerCandidates1_16_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_17
    (if 3 ≤ innerCandidates1_16.length then search edge 3 [7, 829] innerCandidates1_16 836 best else best) = best
  rw [if_pos (by decide), innerChild1_16]
  exact innerScan1_17

def innerTail1_15 : List Nat := 823 :: innerTail1_16
def innerCandidates1_15 : List Nat := [2953, 8101]
set_option maxRecDepth 4096 in
theorem innerCandidates1_15_checked : candidates edge 4 7 823 innerTail1_16 best.bound = innerCandidates1_15 := by
  unfold candidates
  have row : edge 823 = PrimePairCache.smallRow823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_15 : scan edge (search edge 3) 4 [7] 7 innerTail1_15 best = best := by
  rw [innerTail1_15, scan, if_neg (by decide), innerCandidates1_15_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_16
    (if 3 ≤ innerCandidates1_15.length then search edge 3 [7, 823] innerCandidates1_15 830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_16

def innerTail1_14 : List Nat := 757 :: innerTail1_15
def innerCandidates1_14 : List Nat := [829]
set_option maxRecDepth 4096 in
theorem innerCandidates1_14_checked : candidates edge 4 7 757 innerTail1_15 best.bound = innerCandidates1_14 := by
  unfold candidates
  have row : edge 757 = PrimePairCache.smallRow757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_14 : scan edge (search edge 3) 4 [7] 7 innerTail1_14 best = best := by
  rw [innerTail1_14, scan, if_neg (by decide), innerCandidates1_14_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_15
    (if 3 ≤ innerCandidates1_14.length then search edge 3 [7, 757] innerCandidates1_14 764 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_15

def innerTail1_13 : List Nat := 691 :: innerTail1_14
def innerCandidates1_13 : List Nat := [1249, 1693, 3433, 4729]
set_option maxRecDepth 4096 in
theorem innerCandidates1_13_checked : candidates edge 4 7 691 innerTail1_14 best.bound = innerCandidates1_13 := by
  unfold candidates
  have row : edge 691 = PrimePairCache.smallRow691.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_13 : search edge 3 [7, 691] innerCandidates1_13 698 best = best := by decide

theorem innerScan1_13 : scan edge (search edge 3) 4 [7] 7 innerTail1_13 best = best := by
  rw [innerTail1_13, scan, if_neg (by decide), innerCandidates1_13_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_14
    (if 3 ≤ innerCandidates1_13.length then search edge 3 [7, 691] innerCandidates1_13 698 best else best) = best
  rw [if_pos (by decide), innerChild1_13]
  exact innerScan1_14

def innerTail1_12 : List Nat := 673 :: innerTail1_13
def innerCandidates1_12 : List Nat := [7417]
set_option maxRecDepth 4096 in
theorem innerCandidates1_12_checked : candidates edge 4 7 673 innerTail1_13 best.bound = innerCandidates1_12 := by
  unfold candidates
  have row : edge 673 = PrimePairCache.smallRow673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_12 : scan edge (search edge 3) 4 [7] 7 innerTail1_12 best = best := by
  rw [innerTail1_12, scan, if_neg (by decide), innerCandidates1_12_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_13
    (if 3 ≤ innerCandidates1_12.length then search edge 3 [7, 673] innerCandidates1_12 680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_13

def innerTail1_11 : List Nat := 547 :: innerTail1_12
def innerCandidates1_11 : List Nat := [823, 853, 1237, 8311]
set_option maxRecDepth 4096 in
theorem innerCandidates1_11_checked : candidates edge 4 7 547 innerTail1_12 best.bound = innerCandidates1_11 := by
  unfold candidates
  have row : edge 547 = PrimePairCache.smallRow547.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_11 : search edge 3 [7, 547] innerCandidates1_11 554 best = best := by decide

theorem innerScan1_11 : scan edge (search edge 3) 4 [7] 7 innerTail1_11 best = best := by
  rw [innerTail1_11, scan, if_neg (by decide), innerCandidates1_11_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_12
    (if 3 ≤ innerCandidates1_11.length then search edge 3 [7, 547] innerCandidates1_11 554 best else best) = best
  rw [if_pos (by decide), innerChild1_11]
  exact innerScan1_12

def innerTail1_10 : List Nat := 541 :: innerTail1_11
def innerCandidates1_10 : List Nat := [2467, 4159, 8167]
set_option maxRecDepth 4096 in
theorem innerCandidates1_10_checked : candidates edge 4 7 541 innerTail1_11 best.bound = innerCandidates1_10 := by
  unfold candidates
  have row : edge 541 = PrimePairCache.smallRow541.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_10 : search edge 3 [7, 541] innerCandidates1_10 548 best = best := by decide

theorem innerScan1_10 : scan edge (search edge 3) 4 [7] 7 innerTail1_10 best = best := by
  rw [innerTail1_10, scan, if_neg (by decide), innerCandidates1_10_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_11
    (if 3 ≤ innerCandidates1_10.length then search edge 3 [7, 541] innerCandidates1_10 548 best else best) = best
  rw [if_pos (by decide), innerChild1_10]
  exact innerScan1_11

def innerTail1_9 : List Nat := 523 :: innerTail1_10
def innerCandidates1_9 : List Nat := [541, 2089, 2671]
set_option maxRecDepth 4096 in
theorem innerCandidates1_9_checked : candidates edge 4 7 523 innerTail1_10 best.bound = innerCandidates1_9 := by
  unfold candidates
  have row : edge 523 = PrimePairCache.smallRow523.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_9 : search edge 3 [7, 523] innerCandidates1_9 530 best = best := by decide

theorem innerScan1_9 : scan edge (search edge 3) 4 [7] 7 innerTail1_9 best = best := by
  rw [innerTail1_9, scan, if_neg (by decide), innerCandidates1_9_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_10
    (if 3 ≤ innerCandidates1_9.length then search edge 3 [7, 523] innerCandidates1_9 530 best else best) = best
  rw [if_pos (by decide), innerChild1_9]
  exact innerScan1_10

def innerTail1_8 : List Nat := 487 :: innerTail1_9
def innerCandidates1_8 : List Nat := [757, 3433]
set_option maxRecDepth 4096 in
theorem innerCandidates1_8_checked : candidates edge 4 7 487 innerTail1_9 best.bound = innerCandidates1_8 := by
  unfold candidates
  have row : edge 487 = PrimePairCache.smallRow487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan1_8 : scan edge (search edge 3) 4 [7] 7 innerTail1_8 best = best := by
  rw [innerTail1_8, scan, if_neg (by decide), innerCandidates1_8_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_9
    (if 3 ≤ innerCandidates1_8.length then search edge 3 [7, 487] innerCandidates1_8 494 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan1_9

def innerTail1_7 : List Nat := 433 :: innerTail1_8
def innerCandidates1_7 : List Nat := [883, 1471, 2719, 3613, 4441, 4903, 6091, 8167]
set_option maxRecDepth 4096 in
theorem innerCandidates1_7_checked : candidates edge 4 7 433 innerTail1_8 best.bound = innerCandidates1_7 := by
  unfold candidates
  have row : edge 433 = PrimePairCache.smallRow433.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_7 : search edge 3 [7, 433] innerCandidates1_7 440 best = best := by decide

theorem innerScan1_7 : scan edge (search edge 3) 4 [7] 7 innerTail1_7 best = best := by
  rw [innerTail1_7, scan, if_neg (by decide), innerCandidates1_7_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_8
    (if 3 ≤ innerCandidates1_7.length then search edge 3 [7, 433] innerCandidates1_7 440 best else best) = best
  rw [if_pos (by decide), innerChild1_7]
  exact innerScan1_8

def innerTail1_6 : List Nat := 283 :: innerTail1_7
def innerCandidates1_6 : List Nat := [487, 541, 1237, 2287, 2503]
set_option maxRecDepth 4096 in
theorem innerCandidates1_6_checked : candidates edge 4 7 283 innerTail1_7 best.bound = innerCandidates1_6 := by
  unfold candidates
  have row : edge 283 = PrimePairCache.smallRow283.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_6 : search edge 3 [7, 283] innerCandidates1_6 290 best = best := by decide

theorem innerScan1_6 : scan edge (search edge 3) 4 [7] 7 innerTail1_6 best = best := by
  rw [innerTail1_6, scan, if_neg (by decide), innerCandidates1_6_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_7
    (if 3 ≤ innerCandidates1_6.length then search edge 3 [7, 283] innerCandidates1_6 290 best else best) = best
  rw [if_pos (by decide), innerChild1_6]
  exact innerScan1_7

def innerTail1_5 : List Nat := 229 :: innerTail1_6
def innerCandidates1_5 : List Nat := [433, 547, 937, 5659]
set_option maxRecDepth 4096 in
theorem innerCandidates1_5_checked : candidates edge 4 7 229 innerTail1_6 best.bound = innerCandidates1_5 := by
  unfold candidates
  have row : edge 229 = PrimePairCache.smallRow229.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_5 : search edge 3 [7, 229] innerCandidates1_5 236 best = best := by decide

theorem innerScan1_5 : scan edge (search edge 3) 4 [7] 7 innerTail1_5 best = best := by
  rw [innerTail1_5, scan, if_neg (by decide), innerCandidates1_5_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_6
    (if 3 ≤ innerCandidates1_5.length then search edge 3 [7, 229] innerCandidates1_5 236 best else best) = best
  rw [if_pos (by decide), innerChild1_5]
  exact innerScan1_6

def innerTail1_4 : List Nat := 127 :: innerTail1_5
def innerCandidates1_4 : List Nat := [6481, 6949, 7723]
set_option maxRecDepth 4096 in
theorem innerCandidates1_4_checked : candidates edge 4 7 127 innerTail1_5 best.bound = innerCandidates1_4 := by
  unfold candidates
  have row : edge 127 = PrimePairCache.smallRow127.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_4 : search edge 3 [7, 127] innerCandidates1_4 134 best = best := by decide

theorem innerScan1_4 : scan edge (search edge 3) 4 [7] 7 innerTail1_4 best = best := by
  rw [innerTail1_4, scan, if_neg (by decide), innerCandidates1_4_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_5
    (if 3 ≤ innerCandidates1_4.length then search edge 3 [7, 127] innerCandidates1_4 134 best else best) = best
  rw [if_pos (by decide), innerChild1_4]
  exact innerScan1_5

def innerTail1_3 : List Nat := 109 :: innerTail1_4
def innerCandidates1_3 : List Nat := [673, 883, 1399, 3331, 4441, 4567, 7351, 7933]
set_option maxRecDepth 4096 in
theorem innerCandidates1_3_checked : candidates edge 4 7 109 innerTail1_4 best.bound = innerCandidates1_3 := by
  unfold candidates
  have row : edge 109 = PrimePairCache.smallRow109.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_3 : search edge 3 [7, 109] innerCandidates1_3 116 best = best := by decide

theorem innerScan1_3 : scan edge (search edge 3) 4 [7] 7 innerTail1_3 best = best := by
  rw [innerTail1_3, scan, if_neg (by decide), innerCandidates1_3_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_4
    (if 3 ≤ innerCandidates1_3.length then search edge 3 [7, 109] innerCandidates1_3 116 best else best) = best
  rw [if_pos (by decide), innerChild1_3]
  exact innerScan1_4

def innerTail1_2 : List Nat := 97 :: innerTail1_3
def innerCandidates1_2 : List Nat := [829, 883, 3331, 3727, 4507]
set_option maxRecDepth 4096 in
theorem innerCandidates1_2_checked : candidates edge 4 7 97 innerTail1_3 best.bound = innerCandidates1_2 := by
  unfold candidates
  have row : edge 97 = PrimePairCache.smallRow97.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_2 : search edge 3 [7, 97] innerCandidates1_2 104 best = best := by decide

theorem innerScan1_2 : scan edge (search edge 3) 4 [7] 7 innerTail1_2 best = best := by
  rw [innerTail1_2, scan, if_neg (by decide), innerCandidates1_2_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_3
    (if 3 ≤ innerCandidates1_2.length then search edge 3 [7, 97] innerCandidates1_2 104 best else best) = best
  rw [if_pos (by decide), innerChild1_2]
  exact innerScan1_3

def innerTail1_1 : List Nat := 61 :: innerTail1_2
def innerCandidates1_1 : List Nat := [487, 1453, 1693, 1999, 2341, 3181, 5821, 8311]
set_option maxRecDepth 4096 in
theorem innerCandidates1_1_checked : candidates edge 4 7 61 innerTail1_2 best.bound = innerCandidates1_1 := by
  unfold candidates
  have row : edge 61 = PrimePairCache.smallRow61.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_1 : search edge 3 [7, 61] innerCandidates1_1 68 best = best := by decide

theorem innerScan1_1 : scan edge (search edge 3) 4 [7] 7 innerTail1_1 best = best := by
  rw [innerTail1_1, scan, if_neg (by decide), innerCandidates1_1_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_2
    (if 3 ≤ innerCandidates1_1.length then search edge 3 [7, 61] innerCandidates1_1 68 best else best) = best
  rw [if_pos (by decide), innerChild1_1]
  exact innerScan1_2

def innerTail1_0 : List Nat := 19 :: innerTail1_1
def innerCandidates1_0 : List Nat := [97, 433, 937, 1237, 1249, 1693, 1861, 2161, 3727, 4507, 5659]
set_option maxRecDepth 4096 in
theorem innerCandidates1_0_checked : candidates edge 4 7 19 innerTail1_1 best.bound = innerCandidates1_0 := by
  unfold candidates
  have row : edge 19 = PrimePairCache.smallRow19.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild1_0 : search edge 3 [7, 19] innerCandidates1_0 26 best = best := by decide

theorem innerScan1_0 : scan edge (search edge 3) 4 [7] 7 innerTail1_0 best = best := by
  rw [innerTail1_0, scan, if_neg (by decide), innerCandidates1_0_checked]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_1
    (if 3 ≤ innerCandidates1_0.length then search edge 3 [7, 19] innerCandidates1_0 26 best else best) = best
  rw [if_pos (by decide), innerChild1_0]
  exact innerScan1_1

theorem rootBranch1 : search edge 4 [7] rootCandidates1 7 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [7] 7 innerTail1_0 best = best
  exact innerScan1_0

end ProjectEulerSolutions.PrimePairUpper.Search
