import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates681 : List Nat := [5431, 5449, 5659, 6793, 6823, 6991, 7069, 7789, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates681_checked : candidates edge 5 0 5113 tail682 best.bound = rootCandidates681 := by
  unfold candidates
  have row : edge 5113 = SearchData.row5113.contains := by funext q; rfl
  rw [row]
  decide

def innerTail681_9 : List Nat := []
theorem innerScan681_9 : scan edge (search edge 3) 4 [5113] 5113 innerTail681_9 best = best := rfl

def innerTail681_8 : List Nat := 8293 :: innerTail681_9
def innerCandidates681_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates681_8_checked : candidates edge 4 5113 8293 innerTail681_9 best.bound = innerCandidates681_8 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan681_8 : scan edge (search edge 3) 4 [5113] 5113 innerTail681_8 best = best := by
  rw [innerTail681_8, scan, if_neg (by decide), innerCandidates681_8_checked]
  change scan edge (search edge 3) 4 [5113] 5113 innerTail681_9
    (if 3 ≤ innerCandidates681_8.length then search edge 3 [5113, 8293] innerCandidates681_8 13406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan681_9

def innerTail681_7 : List Nat := 7789 :: innerTail681_8
def innerCandidates681_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates681_7_checked : candidates edge 4 5113 7789 innerTail681_8 best.bound = innerCandidates681_7 := by
  unfold candidates
  have row : edge 7789 = SearchData.row7789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan681_7 : scan edge (search edge 3) 4 [5113] 5113 innerTail681_7 best = best := by
  rw [innerTail681_7, scan, if_neg (by decide), innerCandidates681_7_checked]
  change scan edge (search edge 3) 4 [5113] 5113 innerTail681_8
    (if 3 ≤ innerCandidates681_7.length then search edge 3 [5113, 7789] innerCandidates681_7 12902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan681_8

def innerTail681_6 : List Nat := 7069 :: innerTail681_7
def innerCandidates681_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates681_6_checked : candidates edge 4 5113 7069 innerTail681_7 best.bound = innerCandidates681_6 := by
  unfold candidates
  have row : edge 7069 = SearchData.row7069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan681_6 : scan edge (search edge 3) 4 [5113] 5113 innerTail681_6 best = best := by
  rw [innerTail681_6, scan, if_neg (by decide), innerCandidates681_6_checked]
  change scan edge (search edge 3) 4 [5113] 5113 innerTail681_7
    (if 3 ≤ innerCandidates681_6.length then search edge 3 [5113, 7069] innerCandidates681_6 12182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan681_7

def innerTail681_5 : List Nat := 6991 :: innerTail681_6
def innerCandidates681_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates681_5_checked : candidates edge 4 5113 6991 innerTail681_6 best.bound = innerCandidates681_5 := by
  unfold candidates
  have row : edge 6991 = SearchData.row6991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan681_5 : scan edge (search edge 3) 4 [5113] 5113 innerTail681_5 best = best := by
  rw [innerTail681_5, scan, if_neg (by decide), innerCandidates681_5_checked]
  change scan edge (search edge 3) 4 [5113] 5113 innerTail681_6
    (if 3 ≤ innerCandidates681_5.length then search edge 3 [5113, 6991] innerCandidates681_5 12104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan681_6

def innerTail681_4 : List Nat := 6823 :: innerTail681_5
def innerCandidates681_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates681_4_checked : candidates edge 4 5113 6823 innerTail681_5 best.bound = innerCandidates681_4 := by
  unfold candidates
  have row : edge 6823 = SearchData.row6823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan681_4 : scan edge (search edge 3) 4 [5113] 5113 innerTail681_4 best = best := by
  rw [innerTail681_4, scan, if_neg (by decide), innerCandidates681_4_checked]
  change scan edge (search edge 3) 4 [5113] 5113 innerTail681_5
    (if 3 ≤ innerCandidates681_4.length then search edge 3 [5113, 6823] innerCandidates681_4 11936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan681_5

def innerTail681_3 : List Nat := 6793 :: innerTail681_4
def innerCandidates681_3 : List Nat := [8293]
set_option maxRecDepth 4096 in
theorem innerCandidates681_3_checked : candidates edge 4 5113 6793 innerTail681_4 best.bound = innerCandidates681_3 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan681_3 : scan edge (search edge 3) 4 [5113] 5113 innerTail681_3 best = best := by
  rw [innerTail681_3, scan, if_neg (by decide), innerCandidates681_3_checked]
  change scan edge (search edge 3) 4 [5113] 5113 innerTail681_4
    (if 3 ≤ innerCandidates681_3.length then search edge 3 [5113, 6793] innerCandidates681_3 11906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan681_4

def innerTail681_2 : List Nat := 5659 :: innerTail681_3
def innerCandidates681_2 : List Nat := [7069]
set_option maxRecDepth 4096 in
theorem innerCandidates681_2_checked : candidates edge 4 5113 5659 innerTail681_3 best.bound = innerCandidates681_2 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan681_2 : scan edge (search edge 3) 4 [5113] 5113 innerTail681_2 best = best := by
  rw [innerTail681_2, scan, if_neg (by decide), innerCandidates681_2_checked]
  change scan edge (search edge 3) 4 [5113] 5113 innerTail681_3
    (if 3 ≤ innerCandidates681_2.length then search edge 3 [5113, 5659] innerCandidates681_2 10772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan681_3

def innerTail681_1 : List Nat := 5449 :: innerTail681_2
def innerCandidates681_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates681_1_checked : candidates edge 4 5113 5449 innerTail681_2 best.bound = innerCandidates681_1 := by
  unfold candidates
  have row : edge 5449 = SearchData.row5449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan681_1 : scan edge (search edge 3) 4 [5113] 5113 innerTail681_1 best = best := by
  rw [innerTail681_1, scan, if_neg (by decide), innerCandidates681_1_checked]
  change scan edge (search edge 3) 4 [5113] 5113 innerTail681_2
    (if 3 ≤ innerCandidates681_1.length then search edge 3 [5113, 5449] innerCandidates681_1 10562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan681_2

def innerTail681_0 : List Nat := 5431 :: innerTail681_1
def innerCandidates681_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates681_0_checked : candidates edge 4 5113 5431 innerTail681_1 best.bound = innerCandidates681_0 := by
  unfold candidates
  have row : edge 5431 = SearchData.row5431.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan681_0 : scan edge (search edge 3) 4 [5113] 5113 innerTail681_0 best = best := by
  rw [innerTail681_0, scan, if_neg (by decide), innerCandidates681_0_checked]
  change scan edge (search edge 3) 4 [5113] 5113 innerTail681_1
    (if 3 ≤ innerCandidates681_0.length then search edge 3 [5113, 5431] innerCandidates681_0 10544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan681_1

theorem rootBranch681 : search edge 4 [5113] rootCandidates681 5113 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5113] 5113 innerTail681_0 best = best
  exact innerScan681_0

def rootCandidates680 : List Nat := [5779, 5839, 6763, 6793, 7717, 7879, 8017, 8311]
set_option maxRecDepth 4096 in
theorem rootCandidates680_checked : candidates edge 5 0 5107 tail681 best.bound = rootCandidates680 := by
  unfold candidates
  have row : edge 5107 = SearchData.row5107.contains := by funext q; rfl
  rw [row]
  decide

def innerTail680_8 : List Nat := []
theorem innerScan680_8 : scan edge (search edge 3) 4 [5107] 5107 innerTail680_8 best = best := rfl

def innerTail680_7 : List Nat := 8311 :: innerTail680_8
def innerCandidates680_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates680_7_checked : candidates edge 4 5107 8311 innerTail680_8 best.bound = innerCandidates680_7 := by
  unfold candidates
  have row : edge 8311 = SearchData.row8311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan680_7 : scan edge (search edge 3) 4 [5107] 5107 innerTail680_7 best = best := by
  rw [innerTail680_7, scan, if_neg (by decide), innerCandidates680_7_checked]
  change scan edge (search edge 3) 4 [5107] 5107 innerTail680_8
    (if 3 ≤ innerCandidates680_7.length then search edge 3 [5107, 8311] innerCandidates680_7 13418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan680_8

def innerTail680_6 : List Nat := 8017 :: innerTail680_7
def innerCandidates680_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates680_6_checked : candidates edge 4 5107 8017 innerTail680_7 best.bound = innerCandidates680_6 := by
  unfold candidates
  have row : edge 8017 = SearchData.row8017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan680_6 : scan edge (search edge 3) 4 [5107] 5107 innerTail680_6 best = best := by
  rw [innerTail680_6, scan, if_neg (by decide), innerCandidates680_6_checked]
  change scan edge (search edge 3) 4 [5107] 5107 innerTail680_7
    (if 3 ≤ innerCandidates680_6.length then search edge 3 [5107, 8017] innerCandidates680_6 13124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan680_7

def innerTail680_5 : List Nat := 7879 :: innerTail680_6
def innerCandidates680_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates680_5_checked : candidates edge 4 5107 7879 innerTail680_6 best.bound = innerCandidates680_5 := by
  unfold candidates
  have row : edge 7879 = SearchData.row7879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan680_5 : scan edge (search edge 3) 4 [5107] 5107 innerTail680_5 best = best := by
  rw [innerTail680_5, scan, if_neg (by decide), innerCandidates680_5_checked]
  change scan edge (search edge 3) 4 [5107] 5107 innerTail680_6
    (if 3 ≤ innerCandidates680_5.length then search edge 3 [5107, 7879] innerCandidates680_5 12986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan680_6

def innerTail680_4 : List Nat := 7717 :: innerTail680_5
def innerCandidates680_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates680_4_checked : candidates edge 4 5107 7717 innerTail680_5 best.bound = innerCandidates680_4 := by
  unfold candidates
  have row : edge 7717 = SearchData.row7717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan680_4 : scan edge (search edge 3) 4 [5107] 5107 innerTail680_4 best = best := by
  rw [innerTail680_4, scan, if_neg (by decide), innerCandidates680_4_checked]
  change scan edge (search edge 3) 4 [5107] 5107 innerTail680_5
    (if 3 ≤ innerCandidates680_4.length then search edge 3 [5107, 7717] innerCandidates680_4 12824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan680_5

def innerTail680_3 : List Nat := 6793 :: innerTail680_4
def innerCandidates680_3 : List Nat := [7717]
set_option maxRecDepth 4096 in
theorem innerCandidates680_3_checked : candidates edge 4 5107 6793 innerTail680_4 best.bound = innerCandidates680_3 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan680_3 : scan edge (search edge 3) 4 [5107] 5107 innerTail680_3 best = best := by
  rw [innerTail680_3, scan, if_neg (by decide), innerCandidates680_3_checked]
  change scan edge (search edge 3) 4 [5107] 5107 innerTail680_4
    (if 3 ≤ innerCandidates680_3.length then search edge 3 [5107, 6793] innerCandidates680_3 11900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan680_4

def innerTail680_2 : List Nat := 6763 :: innerTail680_3
def innerCandidates680_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates680_2_checked : candidates edge 4 5107 6763 innerTail680_3 best.bound = innerCandidates680_2 := by
  unfold candidates
  have row : edge 6763 = SearchData.row6763.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan680_2 : scan edge (search edge 3) 4 [5107] 5107 innerTail680_2 best = best := by
  rw [innerTail680_2, scan, if_neg (by decide), innerCandidates680_2_checked]
  change scan edge (search edge 3) 4 [5107] 5107 innerTail680_3
    (if 3 ≤ innerCandidates680_2.length then search edge 3 [5107, 6763] innerCandidates680_2 11870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan680_3

def innerTail680_1 : List Nat := 5839 :: innerTail680_2
def innerCandidates680_1 : List Nat := [7879]
set_option maxRecDepth 4096 in
theorem innerCandidates680_1_checked : candidates edge 4 5107 5839 innerTail680_2 best.bound = innerCandidates680_1 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan680_1 : scan edge (search edge 3) 4 [5107] 5107 innerTail680_1 best = best := by
  rw [innerTail680_1, scan, if_neg (by decide), innerCandidates680_1_checked]
  change scan edge (search edge 3) 4 [5107] 5107 innerTail680_2
    (if 3 ≤ innerCandidates680_1.length then search edge 3 [5107, 5839] innerCandidates680_1 10946 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan680_2

def innerTail680_0 : List Nat := 5779 :: innerTail680_1
def innerCandidates680_0 : List Nat := [7717]
set_option maxRecDepth 4096 in
theorem innerCandidates680_0_checked : candidates edge 4 5107 5779 innerTail680_1 best.bound = innerCandidates680_0 := by
  unfold candidates
  have row : edge 5779 = SearchData.row5779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan680_0 : scan edge (search edge 3) 4 [5107] 5107 innerTail680_0 best = best := by
  rw [innerTail680_0, scan, if_neg (by decide), innerCandidates680_0_checked]
  change scan edge (search edge 3) 4 [5107] 5107 innerTail680_1
    (if 3 ≤ innerCandidates680_0.length then search edge 3 [5107, 5779] innerCandidates680_0 10886 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan680_1

theorem rootBranch680 : search edge 4 [5107] rootCandidates680 5107 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5107] 5107 innerTail680_0 best = best
  exact innerScan680_0

def rootCandidates679 : List Nat := [5953, 6151, 6397, 6571, 6883, 7069, 7237, 8263, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates679_checked : candidates edge 5 0 5101 tail680 best.bound = rootCandidates679 := by
  unfold candidates
  have row : edge 5101 = SearchData.row5101.contains := by funext q; rfl
  rw [row]
  decide

def innerTail679_9 : List Nat := []
theorem innerScan679_9 : scan edge (search edge 3) 4 [5101] 5101 innerTail679_9 best = best := rfl

def innerTail679_8 : List Nat := 8293 :: innerTail679_9
def innerCandidates679_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates679_8_checked : candidates edge 4 5101 8293 innerTail679_9 best.bound = innerCandidates679_8 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan679_8 : scan edge (search edge 3) 4 [5101] 5101 innerTail679_8 best = best := by
  rw [innerTail679_8, scan, if_neg (by decide), innerCandidates679_8_checked]
  change scan edge (search edge 3) 4 [5101] 5101 innerTail679_9
    (if 3 ≤ innerCandidates679_8.length then search edge 3 [5101, 8293] innerCandidates679_8 13394 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan679_9

def innerTail679_7 : List Nat := 8263 :: innerTail679_8
def innerCandidates679_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates679_7_checked : candidates edge 4 5101 8263 innerTail679_8 best.bound = innerCandidates679_7 := by
  unfold candidates
  have row : edge 8263 = SearchData.row8263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan679_7 : scan edge (search edge 3) 4 [5101] 5101 innerTail679_7 best = best := by
  rw [innerTail679_7, scan, if_neg (by decide), innerCandidates679_7_checked]
  change scan edge (search edge 3) 4 [5101] 5101 innerTail679_8
    (if 3 ≤ innerCandidates679_7.length then search edge 3 [5101, 8263] innerCandidates679_7 13364 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan679_8

def innerTail679_6 : List Nat := 7237 :: innerTail679_7
def innerCandidates679_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates679_6_checked : candidates edge 4 5101 7237 innerTail679_7 best.bound = innerCandidates679_6 := by
  unfold candidates
  have row : edge 7237 = SearchData.row7237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan679_6 : scan edge (search edge 3) 4 [5101] 5101 innerTail679_6 best = best := by
  rw [innerTail679_6, scan, if_neg (by decide), innerCandidates679_6_checked]
  change scan edge (search edge 3) 4 [5101] 5101 innerTail679_7
    (if 3 ≤ innerCandidates679_6.length then search edge 3 [5101, 7237] innerCandidates679_6 12338 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan679_7

def innerTail679_5 : List Nat := 7069 :: innerTail679_6
def innerCandidates679_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates679_5_checked : candidates edge 4 5101 7069 innerTail679_6 best.bound = innerCandidates679_5 := by
  unfold candidates
  have row : edge 7069 = SearchData.row7069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan679_5 : scan edge (search edge 3) 4 [5101] 5101 innerTail679_5 best = best := by
  rw [innerTail679_5, scan, if_neg (by decide), innerCandidates679_5_checked]
  change scan edge (search edge 3) 4 [5101] 5101 innerTail679_6
    (if 3 ≤ innerCandidates679_5.length then search edge 3 [5101, 7069] innerCandidates679_5 12170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan679_6

def innerTail679_4 : List Nat := 6883 :: innerTail679_5
def innerCandidates679_4 : List Nat := [8293]
set_option maxRecDepth 4096 in
theorem innerCandidates679_4_checked : candidates edge 4 5101 6883 innerTail679_5 best.bound = innerCandidates679_4 := by
  unfold candidates
  have row : edge 6883 = SearchData.row6883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan679_4 : scan edge (search edge 3) 4 [5101] 5101 innerTail679_4 best = best := by
  rw [innerTail679_4, scan, if_neg (by decide), innerCandidates679_4_checked]
  change scan edge (search edge 3) 4 [5101] 5101 innerTail679_5
    (if 3 ≤ innerCandidates679_4.length then search edge 3 [5101, 6883] innerCandidates679_4 11984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan679_5

def innerTail679_3 : List Nat := 6571 :: innerTail679_4
def innerCandidates679_3 : List Nat := [8293]
set_option maxRecDepth 4096 in
theorem innerCandidates679_3_checked : candidates edge 4 5101 6571 innerTail679_4 best.bound = innerCandidates679_3 := by
  unfold candidates
  have row : edge 6571 = SearchData.row6571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan679_3 : scan edge (search edge 3) 4 [5101] 5101 innerTail679_3 best = best := by
  rw [innerTail679_3, scan, if_neg (by decide), innerCandidates679_3_checked]
  change scan edge (search edge 3) 4 [5101] 5101 innerTail679_4
    (if 3 ≤ innerCandidates679_3.length then search edge 3 [5101, 6571] innerCandidates679_3 11672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan679_4

def innerTail679_2 : List Nat := 6397 :: innerTail679_3
def innerCandidates679_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates679_2_checked : candidates edge 4 5101 6397 innerTail679_3 best.bound = innerCandidates679_2 := by
  unfold candidates
  have row : edge 6397 = SearchData.row6397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan679_2 : scan edge (search edge 3) 4 [5101] 5101 innerTail679_2 best = best := by
  rw [innerTail679_2, scan, if_neg (by decide), innerCandidates679_2_checked]
  change scan edge (search edge 3) 4 [5101] 5101 innerTail679_3
    (if 3 ≤ innerCandidates679_2.length then search edge 3 [5101, 6397] innerCandidates679_2 11498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan679_3

def innerTail679_1 : List Nat := 6151 :: innerTail679_2
def innerCandidates679_1 : List Nat := [6571]
set_option maxRecDepth 4096 in
theorem innerCandidates679_1_checked : candidates edge 4 5101 6151 innerTail679_2 best.bound = innerCandidates679_1 := by
  unfold candidates
  have row : edge 6151 = SearchData.row6151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan679_1 : scan edge (search edge 3) 4 [5101] 5101 innerTail679_1 best = best := by
  rw [innerTail679_1, scan, if_neg (by decide), innerCandidates679_1_checked]
  change scan edge (search edge 3) 4 [5101] 5101 innerTail679_2
    (if 3 ≤ innerCandidates679_1.length then search edge 3 [5101, 6151] innerCandidates679_1 11252 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan679_2

def innerTail679_0 : List Nat := 5953 :: innerTail679_1
def innerCandidates679_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates679_0_checked : candidates edge 4 5101 5953 innerTail679_1 best.bound = innerCandidates679_0 := by
  unfold candidates
  have row : edge 5953 = SearchData.row5953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan679_0 : scan edge (search edge 3) 4 [5101] 5101 innerTail679_0 best = best := by
  rw [innerTail679_0, scan, if_neg (by decide), innerCandidates679_0_checked]
  change scan edge (search edge 3) 4 [5101] 5101 innerTail679_1
    (if 3 ≤ innerCandidates679_0.length then search edge 3 [5101, 5953] innerCandidates679_0 11054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan679_1

theorem rootBranch679 : search edge 4 [5101] rootCandidates679 5101 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5101] 5101 innerTail679_0 best = best
  exact innerScan679_0

def rootCandidates678 : List Nat := [5471, 5501, 5507, 6857, 7151, 7523, 7649, 7853, 7949, 8231, 8273]
set_option maxRecDepth 4096 in
theorem rootCandidates678_checked : candidates edge 5 0 5099 tail679 best.bound = rootCandidates678 := by
  unfold candidates
  have row : edge 5099 = SearchData.row5099.contains := by funext q; rfl
  rw [row]
  decide

def innerTail678_11 : List Nat := []
theorem innerScan678_11 : scan edge (search edge 3) 4 [5099] 5099 innerTail678_11 best = best := rfl

def innerTail678_10 : List Nat := 8273 :: innerTail678_11
def innerCandidates678_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates678_10_checked : candidates edge 4 5099 8273 innerTail678_11 best.bound = innerCandidates678_10 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan678_10 : scan edge (search edge 3) 4 [5099] 5099 innerTail678_10 best = best := by
  rw [innerTail678_10, scan, if_neg (by decide), innerCandidates678_10_checked]
  change scan edge (search edge 3) 4 [5099] 5099 innerTail678_11
    (if 3 ≤ innerCandidates678_10.length then search edge 3 [5099, 8273] innerCandidates678_10 13372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan678_11

def innerTail678_9 : List Nat := 8231 :: innerTail678_10
def innerCandidates678_9 : List Nat := [8273]
set_option maxRecDepth 4096 in
theorem innerCandidates678_9_checked : candidates edge 4 5099 8231 innerTail678_10 best.bound = innerCandidates678_9 := by
  unfold candidates
  have row : edge 8231 = SearchData.row8231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan678_9 : scan edge (search edge 3) 4 [5099] 5099 innerTail678_9 best = best := by
  rw [innerTail678_9, scan, if_neg (by decide), innerCandidates678_9_checked]
  change scan edge (search edge 3) 4 [5099] 5099 innerTail678_10
    (if 3 ≤ innerCandidates678_9.length then search edge 3 [5099, 8231] innerCandidates678_9 13330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan678_10

def innerTail678_8 : List Nat := 7949 :: innerTail678_9
def innerCandidates678_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates678_8_checked : candidates edge 4 5099 7949 innerTail678_9 best.bound = innerCandidates678_8 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan678_8 : scan edge (search edge 3) 4 [5099] 5099 innerTail678_8 best = best := by
  rw [innerTail678_8, scan, if_neg (by decide), innerCandidates678_8_checked]
  change scan edge (search edge 3) 4 [5099] 5099 innerTail678_9
    (if 3 ≤ innerCandidates678_8.length then search edge 3 [5099, 7949] innerCandidates678_8 13048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan678_9

def innerTail678_7 : List Nat := 7853 :: innerTail678_8
def innerCandidates678_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates678_7_checked : candidates edge 4 5099 7853 innerTail678_8 best.bound = innerCandidates678_7 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan678_7 : scan edge (search edge 3) 4 [5099] 5099 innerTail678_7 best = best := by
  rw [innerTail678_7, scan, if_neg (by decide), innerCandidates678_7_checked]
  change scan edge (search edge 3) 4 [5099] 5099 innerTail678_8
    (if 3 ≤ innerCandidates678_7.length then search edge 3 [5099, 7853] innerCandidates678_7 12952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan678_8

def innerTail678_6 : List Nat := 7649 :: innerTail678_7
def innerCandidates678_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates678_6_checked : candidates edge 4 5099 7649 innerTail678_7 best.bound = innerCandidates678_6 := by
  unfold candidates
  have row : edge 7649 = SearchData.row7649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan678_6 : scan edge (search edge 3) 4 [5099] 5099 innerTail678_6 best = best := by
  rw [innerTail678_6, scan, if_neg (by decide), innerCandidates678_6_checked]
  change scan edge (search edge 3) 4 [5099] 5099 innerTail678_7
    (if 3 ≤ innerCandidates678_6.length then search edge 3 [5099, 7649] innerCandidates678_6 12748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan678_7

def innerTail678_5 : List Nat := 7523 :: innerTail678_6
def innerCandidates678_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates678_5_checked : candidates edge 4 5099 7523 innerTail678_6 best.bound = innerCandidates678_5 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan678_5 : scan edge (search edge 3) 4 [5099] 5099 innerTail678_5 best = best := by
  rw [innerTail678_5, scan, if_neg (by decide), innerCandidates678_5_checked]
  change scan edge (search edge 3) 4 [5099] 5099 innerTail678_6
    (if 3 ≤ innerCandidates678_5.length then search edge 3 [5099, 7523] innerCandidates678_5 12622 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan678_6

def innerTail678_4 : List Nat := 7151 :: innerTail678_5
def innerCandidates678_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates678_4_checked : candidates edge 4 5099 7151 innerTail678_5 best.bound = innerCandidates678_4 := by
  unfold candidates
  have row : edge 7151 = SearchData.row7151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan678_4 : scan edge (search edge 3) 4 [5099] 5099 innerTail678_4 best = best := by
  rw [innerTail678_4, scan, if_neg (by decide), innerCandidates678_4_checked]
  change scan edge (search edge 3) 4 [5099] 5099 innerTail678_5
    (if 3 ≤ innerCandidates678_4.length then search edge 3 [5099, 7151] innerCandidates678_4 12250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan678_5

def innerTail678_3 : List Nat := 6857 :: innerTail678_4
def innerCandidates678_3 : List Nat := [7151]
set_option maxRecDepth 4096 in
theorem innerCandidates678_3_checked : candidates edge 4 5099 6857 innerTail678_4 best.bound = innerCandidates678_3 := by
  unfold candidates
  have row : edge 6857 = SearchData.row6857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan678_3 : scan edge (search edge 3) 4 [5099] 5099 innerTail678_3 best = best := by
  rw [innerTail678_3, scan, if_neg (by decide), innerCandidates678_3_checked]
  change scan edge (search edge 3) 4 [5099] 5099 innerTail678_4
    (if 3 ≤ innerCandidates678_3.length then search edge 3 [5099, 6857] innerCandidates678_3 11956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan678_4

def innerTail678_2 : List Nat := 5507 :: innerTail678_3
def innerCandidates678_2 : List Nat := [7853]
set_option maxRecDepth 4096 in
theorem innerCandidates678_2_checked : candidates edge 4 5099 5507 innerTail678_3 best.bound = innerCandidates678_2 := by
  unfold candidates
  have row : edge 5507 = SearchData.row5507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan678_2 : scan edge (search edge 3) 4 [5099] 5099 innerTail678_2 best = best := by
  rw [innerTail678_2, scan, if_neg (by decide), innerCandidates678_2_checked]
  change scan edge (search edge 3) 4 [5099] 5099 innerTail678_3
    (if 3 ≤ innerCandidates678_2.length then search edge 3 [5099, 5507] innerCandidates678_2 10606 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan678_3

def innerTail678_1 : List Nat := 5501 :: innerTail678_2
def innerCandidates678_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates678_1_checked : candidates edge 4 5099 5501 innerTail678_2 best.bound = innerCandidates678_1 := by
  unfold candidates
  have row : edge 5501 = SearchData.row5501.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan678_1 : scan edge (search edge 3) 4 [5099] 5099 innerTail678_1 best = best := by
  rw [innerTail678_1, scan, if_neg (by decide), innerCandidates678_1_checked]
  change scan edge (search edge 3) 4 [5099] 5099 innerTail678_2
    (if 3 ≤ innerCandidates678_1.length then search edge 3 [5099, 5501] innerCandidates678_1 10600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan678_2

def innerTail678_0 : List Nat := 5471 :: innerTail678_1
def innerCandidates678_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates678_0_checked : candidates edge 4 5099 5471 innerTail678_1 best.bound = innerCandidates678_0 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan678_0 : scan edge (search edge 3) 4 [5099] 5099 innerTail678_0 best = best := by
  rw [innerTail678_0, scan, if_neg (by decide), innerCandidates678_0_checked]
  change scan edge (search edge 3) 4 [5099] 5099 innerTail678_1
    (if 3 ≤ innerCandidates678_0.length then search edge 3 [5099, 5471] innerCandidates678_0 10570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan678_1

theorem rootBranch678 : search edge 4 [5099] rootCandidates678 5099 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5099] 5099 innerTail678_0 best = best
  exact innerScan678_0

def rootCandidates677 : List Nat := [5273, 5519, 5861, 5897, 5927, 6569, 6719, 6779, 6803, 7247, 8147]
set_option maxRecDepth 4096 in
theorem rootCandidates677_checked : candidates edge 5 0 5087 tail678 best.bound = rootCandidates677 := by
  unfold candidates
  have row : edge 5087 = SearchData.row5087.contains := by funext q; rfl
  rw [row]
  decide

def innerTail677_11 : List Nat := []
theorem innerScan677_11 : scan edge (search edge 3) 4 [5087] 5087 innerTail677_11 best = best := rfl

def innerTail677_10 : List Nat := 8147 :: innerTail677_11
def innerCandidates677_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates677_10_checked : candidates edge 4 5087 8147 innerTail677_11 best.bound = innerCandidates677_10 := by
  unfold candidates
  have row : edge 8147 = SearchData.row8147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan677_10 : scan edge (search edge 3) 4 [5087] 5087 innerTail677_10 best = best := by
  rw [innerTail677_10, scan, if_neg (by decide), innerCandidates677_10_checked]
  change scan edge (search edge 3) 4 [5087] 5087 innerTail677_11
    (if 3 ≤ innerCandidates677_10.length then search edge 3 [5087, 8147] innerCandidates677_10 13234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan677_11

def innerTail677_9 : List Nat := 7247 :: innerTail677_10
def innerCandidates677_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates677_9_checked : candidates edge 4 5087 7247 innerTail677_10 best.bound = innerCandidates677_9 := by
  unfold candidates
  have row : edge 7247 = SearchData.row7247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan677_9 : scan edge (search edge 3) 4 [5087] 5087 innerTail677_9 best = best := by
  rw [innerTail677_9, scan, if_neg (by decide), innerCandidates677_9_checked]
  change scan edge (search edge 3) 4 [5087] 5087 innerTail677_10
    (if 3 ≤ innerCandidates677_9.length then search edge 3 [5087, 7247] innerCandidates677_9 12334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan677_10

def innerTail677_8 : List Nat := 6803 :: innerTail677_9
def innerCandidates677_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates677_8_checked : candidates edge 4 5087 6803 innerTail677_9 best.bound = innerCandidates677_8 := by
  unfold candidates
  have row : edge 6803 = SearchData.row6803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan677_8 : scan edge (search edge 3) 4 [5087] 5087 innerTail677_8 best = best := by
  rw [innerTail677_8, scan, if_neg (by decide), innerCandidates677_8_checked]
  change scan edge (search edge 3) 4 [5087] 5087 innerTail677_9
    (if 3 ≤ innerCandidates677_8.length then search edge 3 [5087, 6803] innerCandidates677_8 11890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan677_9

def innerTail677_7 : List Nat := 6779 :: innerTail677_8
def innerCandidates677_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates677_7_checked : candidates edge 4 5087 6779 innerTail677_8 best.bound = innerCandidates677_7 := by
  unfold candidates
  have row : edge 6779 = SearchData.row6779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan677_7 : scan edge (search edge 3) 4 [5087] 5087 innerTail677_7 best = best := by
  rw [innerTail677_7, scan, if_neg (by decide), innerCandidates677_7_checked]
  change scan edge (search edge 3) 4 [5087] 5087 innerTail677_8
    (if 3 ≤ innerCandidates677_7.length then search edge 3 [5087, 6779] innerCandidates677_7 11866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan677_8

def innerTail677_6 : List Nat := 6719 :: innerTail677_7
def innerCandidates677_6 : List Nat := [8147]
set_option maxRecDepth 4096 in
theorem innerCandidates677_6_checked : candidates edge 4 5087 6719 innerTail677_7 best.bound = innerCandidates677_6 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan677_6 : scan edge (search edge 3) 4 [5087] 5087 innerTail677_6 best = best := by
  rw [innerTail677_6, scan, if_neg (by decide), innerCandidates677_6_checked]
  change scan edge (search edge 3) 4 [5087] 5087 innerTail677_7
    (if 3 ≤ innerCandidates677_6.length then search edge 3 [5087, 6719] innerCandidates677_6 11806 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan677_7

def innerTail677_5 : List Nat := 6569 :: innerTail677_6
def innerCandidates677_5 : List Nat := [6779]
set_option maxRecDepth 4096 in
theorem innerCandidates677_5_checked : candidates edge 4 5087 6569 innerTail677_6 best.bound = innerCandidates677_5 := by
  unfold candidates
  have row : edge 6569 = SearchData.row6569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan677_5 : scan edge (search edge 3) 4 [5087] 5087 innerTail677_5 best = best := by
  rw [innerTail677_5, scan, if_neg (by decide), innerCandidates677_5_checked]
  change scan edge (search edge 3) 4 [5087] 5087 innerTail677_6
    (if 3 ≤ innerCandidates677_5.length then search edge 3 [5087, 6569] innerCandidates677_5 11656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan677_6

def innerTail677_4 : List Nat := 5927 :: innerTail677_5
def innerCandidates677_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates677_4_checked : candidates edge 4 5087 5927 innerTail677_5 best.bound = innerCandidates677_4 := by
  unfold candidates
  have row : edge 5927 = SearchData.row5927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan677_4 : scan edge (search edge 3) 4 [5087] 5087 innerTail677_4 best = best := by
  rw [innerTail677_4, scan, if_neg (by decide), innerCandidates677_4_checked]
  change scan edge (search edge 3) 4 [5087] 5087 innerTail677_5
    (if 3 ≤ innerCandidates677_4.length then search edge 3 [5087, 5927] innerCandidates677_4 11014 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan677_5

def innerTail677_3 : List Nat := 5897 :: innerTail677_4
def innerCandidates677_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates677_3_checked : candidates edge 4 5087 5897 innerTail677_4 best.bound = innerCandidates677_3 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan677_3 : scan edge (search edge 3) 4 [5087] 5087 innerTail677_3 best = best := by
  rw [innerTail677_3, scan, if_neg (by decide), innerCandidates677_3_checked]
  change scan edge (search edge 3) 4 [5087] 5087 innerTail677_4
    (if 3 ≤ innerCandidates677_3.length then search edge 3 [5087, 5897] innerCandidates677_3 10984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan677_4

def innerTail677_2 : List Nat := 5861 :: innerTail677_3
def innerCandidates677_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates677_2_checked : candidates edge 4 5087 5861 innerTail677_3 best.bound = innerCandidates677_2 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan677_2 : scan edge (search edge 3) 4 [5087] 5087 innerTail677_2 best = best := by
  rw [innerTail677_2, scan, if_neg (by decide), innerCandidates677_2_checked]
  change scan edge (search edge 3) 4 [5087] 5087 innerTail677_3
    (if 3 ≤ innerCandidates677_2.length then search edge 3 [5087, 5861] innerCandidates677_2 10948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan677_3

def innerTail677_1 : List Nat := 5519 :: innerTail677_2
def innerCandidates677_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates677_1_checked : candidates edge 4 5087 5519 innerTail677_2 best.bound = innerCandidates677_1 := by
  unfold candidates
  have row : edge 5519 = SearchData.row5519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan677_1 : scan edge (search edge 3) 4 [5087] 5087 innerTail677_1 best = best := by
  rw [innerTail677_1, scan, if_neg (by decide), innerCandidates677_1_checked]
  change scan edge (search edge 3) 4 [5087] 5087 innerTail677_2
    (if 3 ≤ innerCandidates677_1.length then search edge 3 [5087, 5519] innerCandidates677_1 10606 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan677_2

def innerTail677_0 : List Nat := 5273 :: innerTail677_1
def innerCandidates677_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates677_0_checked : candidates edge 4 5087 5273 innerTail677_1 best.bound = innerCandidates677_0 := by
  unfold candidates
  have row : edge 5273 = SearchData.row5273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan677_0 : scan edge (search edge 3) 4 [5087] 5087 innerTail677_0 best = best := by
  rw [innerTail677_0, scan, if_neg (by decide), innerCandidates677_0_checked]
  change scan edge (search edge 3) 4 [5087] 5087 innerTail677_1
    (if 3 ≤ innerCandidates677_0.length then search edge 3 [5087, 5273] innerCandidates677_0 10360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan677_1

theorem rootBranch677 : search edge 4 [5087] rootCandidates677 5087 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5087] 5087 innerTail677_0 best = best
  exact innerScan677_0

def rootCandidates676 : List Nat := [5783, 6971, 7211, 7253, 7307, 7817, 8291]
set_option maxRecDepth 4096 in
theorem rootCandidates676_checked : candidates edge 5 0 5081 tail677 best.bound = rootCandidates676 := by
  unfold candidates
  have row : edge 5081 = SearchData.row5081.contains := by funext q; rfl
  rw [row]
  decide

def innerTail676_7 : List Nat := []
theorem innerScan676_7 : scan edge (search edge 3) 4 [5081] 5081 innerTail676_7 best = best := rfl

def innerTail676_6 : List Nat := 8291 :: innerTail676_7
def innerCandidates676_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates676_6_checked : candidates edge 4 5081 8291 innerTail676_7 best.bound = innerCandidates676_6 := by
  unfold candidates
  have row : edge 8291 = SearchData.row8291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan676_6 : scan edge (search edge 3) 4 [5081] 5081 innerTail676_6 best = best := by
  rw [innerTail676_6, scan, if_neg (by decide), innerCandidates676_6_checked]
  change scan edge (search edge 3) 4 [5081] 5081 innerTail676_7
    (if 3 ≤ innerCandidates676_6.length then search edge 3 [5081, 8291] innerCandidates676_6 13372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan676_7

def innerTail676_5 : List Nat := 7817 :: innerTail676_6
def innerCandidates676_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates676_5_checked : candidates edge 4 5081 7817 innerTail676_6 best.bound = innerCandidates676_5 := by
  unfold candidates
  have row : edge 7817 = SearchData.row7817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan676_5 : scan edge (search edge 3) 4 [5081] 5081 innerTail676_5 best = best := by
  rw [innerTail676_5, scan, if_neg (by decide), innerCandidates676_5_checked]
  change scan edge (search edge 3) 4 [5081] 5081 innerTail676_6
    (if 3 ≤ innerCandidates676_5.length then search edge 3 [5081, 7817] innerCandidates676_5 12898 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan676_6

def innerTail676_4 : List Nat := 7307 :: innerTail676_5
def innerCandidates676_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates676_4_checked : candidates edge 4 5081 7307 innerTail676_5 best.bound = innerCandidates676_4 := by
  unfold candidates
  have row : edge 7307 = SearchData.row7307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan676_4 : scan edge (search edge 3) 4 [5081] 5081 innerTail676_4 best = best := by
  rw [innerTail676_4, scan, if_neg (by decide), innerCandidates676_4_checked]
  change scan edge (search edge 3) 4 [5081] 5081 innerTail676_5
    (if 3 ≤ innerCandidates676_4.length then search edge 3 [5081, 7307] innerCandidates676_4 12388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan676_5

def innerTail676_3 : List Nat := 7253 :: innerTail676_4
def innerCandidates676_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates676_3_checked : candidates edge 4 5081 7253 innerTail676_4 best.bound = innerCandidates676_3 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan676_3 : scan edge (search edge 3) 4 [5081] 5081 innerTail676_3 best = best := by
  rw [innerTail676_3, scan, if_neg (by decide), innerCandidates676_3_checked]
  change scan edge (search edge 3) 4 [5081] 5081 innerTail676_4
    (if 3 ≤ innerCandidates676_3.length then search edge 3 [5081, 7253] innerCandidates676_3 12334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan676_4

def innerTail676_2 : List Nat := 7211 :: innerTail676_3
def innerCandidates676_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates676_2_checked : candidates edge 4 5081 7211 innerTail676_3 best.bound = innerCandidates676_2 := by
  unfold candidates
  have row : edge 7211 = SearchData.row7211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan676_2 : scan edge (search edge 3) 4 [5081] 5081 innerTail676_2 best = best := by
  rw [innerTail676_2, scan, if_neg (by decide), innerCandidates676_2_checked]
  change scan edge (search edge 3) 4 [5081] 5081 innerTail676_3
    (if 3 ≤ innerCandidates676_2.length then search edge 3 [5081, 7211] innerCandidates676_2 12292 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan676_3

def innerTail676_1 : List Nat := 6971 :: innerTail676_2
def innerCandidates676_1 : List Nat := [7211]
set_option maxRecDepth 4096 in
theorem innerCandidates676_1_checked : candidates edge 4 5081 6971 innerTail676_2 best.bound = innerCandidates676_1 := by
  unfold candidates
  have row : edge 6971 = SearchData.row6971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan676_1 : scan edge (search edge 3) 4 [5081] 5081 innerTail676_1 best = best := by
  rw [innerTail676_1, scan, if_neg (by decide), innerCandidates676_1_checked]
  change scan edge (search edge 3) 4 [5081] 5081 innerTail676_2
    (if 3 ≤ innerCandidates676_1.length then search edge 3 [5081, 6971] innerCandidates676_1 12052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan676_2

def innerTail676_0 : List Nat := 5783 :: innerTail676_1
def innerCandidates676_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates676_0_checked : candidates edge 4 5081 5783 innerTail676_1 best.bound = innerCandidates676_0 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan676_0 : scan edge (search edge 3) 4 [5081] 5081 innerTail676_0 best = best := by
  rw [innerTail676_0, scan, if_neg (by decide), innerCandidates676_0_checked]
  change scan edge (search edge 3) 4 [5081] 5081 innerTail676_1
    (if 3 ≤ innerCandidates676_0.length then search edge 3 [5081, 5783] innerCandidates676_0 10864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan676_1

theorem rootBranch676 : search edge 4 [5081] rootCandidates676 5081 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5081] 5081 innerTail676_0 best = best
  exact innerScan676_0

def rootCandidates675 : List Nat := [5779, 6043, 6091, 6367, 6763, 7459, 7687, 7873]
set_option maxRecDepth 4096 in
theorem rootCandidates675_checked : candidates edge 5 0 5077 tail676 best.bound = rootCandidates675 := by
  unfold candidates
  have row : edge 5077 = SearchData.row5077.contains := by funext q; rfl
  rw [row]
  decide

def innerTail675_8 : List Nat := []
theorem innerScan675_8 : scan edge (search edge 3) 4 [5077] 5077 innerTail675_8 best = best := rfl

def innerTail675_7 : List Nat := 7873 :: innerTail675_8
def innerCandidates675_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates675_7_checked : candidates edge 4 5077 7873 innerTail675_8 best.bound = innerCandidates675_7 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan675_7 : scan edge (search edge 3) 4 [5077] 5077 innerTail675_7 best = best := by
  rw [innerTail675_7, scan, if_neg (by decide), innerCandidates675_7_checked]
  change scan edge (search edge 3) 4 [5077] 5077 innerTail675_8
    (if 3 ≤ innerCandidates675_7.length then search edge 3 [5077, 7873] innerCandidates675_7 12950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan675_8

def innerTail675_6 : List Nat := 7687 :: innerTail675_7
def innerCandidates675_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates675_6_checked : candidates edge 4 5077 7687 innerTail675_7 best.bound = innerCandidates675_6 := by
  unfold candidates
  have row : edge 7687 = SearchData.row7687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan675_6 : scan edge (search edge 3) 4 [5077] 5077 innerTail675_6 best = best := by
  rw [innerTail675_6, scan, if_neg (by decide), innerCandidates675_6_checked]
  change scan edge (search edge 3) 4 [5077] 5077 innerTail675_7
    (if 3 ≤ innerCandidates675_6.length then search edge 3 [5077, 7687] innerCandidates675_6 12764 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan675_7

def innerTail675_5 : List Nat := 7459 :: innerTail675_6
def innerCandidates675_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates675_5_checked : candidates edge 4 5077 7459 innerTail675_6 best.bound = innerCandidates675_5 := by
  unfold candidates
  have row : edge 7459 = SearchData.row7459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan675_5 : scan edge (search edge 3) 4 [5077] 5077 innerTail675_5 best = best := by
  rw [innerTail675_5, scan, if_neg (by decide), innerCandidates675_5_checked]
  change scan edge (search edge 3) 4 [5077] 5077 innerTail675_6
    (if 3 ≤ innerCandidates675_5.length then search edge 3 [5077, 7459] innerCandidates675_5 12536 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan675_6

def innerTail675_4 : List Nat := 6763 :: innerTail675_5
def innerCandidates675_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates675_4_checked : candidates edge 4 5077 6763 innerTail675_5 best.bound = innerCandidates675_4 := by
  unfold candidates
  have row : edge 6763 = SearchData.row6763.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan675_4 : scan edge (search edge 3) 4 [5077] 5077 innerTail675_4 best = best := by
  rw [innerTail675_4, scan, if_neg (by decide), innerCandidates675_4_checked]
  change scan edge (search edge 3) 4 [5077] 5077 innerTail675_5
    (if 3 ≤ innerCandidates675_4.length then search edge 3 [5077, 6763] innerCandidates675_4 11840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan675_5

def innerTail675_3 : List Nat := 6367 :: innerTail675_4
def innerCandidates675_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates675_3_checked : candidates edge 4 5077 6367 innerTail675_4 best.bound = innerCandidates675_3 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan675_3 : scan edge (search edge 3) 4 [5077] 5077 innerTail675_3 best = best := by
  rw [innerTail675_3, scan, if_neg (by decide), innerCandidates675_3_checked]
  change scan edge (search edge 3) 4 [5077] 5077 innerTail675_4
    (if 3 ≤ innerCandidates675_3.length then search edge 3 [5077, 6367] innerCandidates675_3 11444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan675_4

def innerTail675_2 : List Nat := 6091 :: innerTail675_3
def innerCandidates675_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates675_2_checked : candidates edge 4 5077 6091 innerTail675_3 best.bound = innerCandidates675_2 := by
  unfold candidates
  have row : edge 6091 = SearchData.row6091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan675_2 : scan edge (search edge 3) 4 [5077] 5077 innerTail675_2 best = best := by
  rw [innerTail675_2, scan, if_neg (by decide), innerCandidates675_2_checked]
  change scan edge (search edge 3) 4 [5077] 5077 innerTail675_3
    (if 3 ≤ innerCandidates675_2.length then search edge 3 [5077, 6091] innerCandidates675_2 11168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan675_3

def innerTail675_1 : List Nat := 6043 :: innerTail675_2
def innerCandidates675_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates675_1_checked : candidates edge 4 5077 6043 innerTail675_2 best.bound = innerCandidates675_1 := by
  unfold candidates
  have row : edge 6043 = SearchData.row6043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan675_1 : scan edge (search edge 3) 4 [5077] 5077 innerTail675_1 best = best := by
  rw [innerTail675_1, scan, if_neg (by decide), innerCandidates675_1_checked]
  change scan edge (search edge 3) 4 [5077] 5077 innerTail675_2
    (if 3 ≤ innerCandidates675_1.length then search edge 3 [5077, 6043] innerCandidates675_1 11120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan675_2

def innerTail675_0 : List Nat := 5779 :: innerTail675_1
def innerCandidates675_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates675_0_checked : candidates edge 4 5077 5779 innerTail675_1 best.bound = innerCandidates675_0 := by
  unfold candidates
  have row : edge 5779 = SearchData.row5779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan675_0 : scan edge (search edge 3) 4 [5077] 5077 innerTail675_0 best = best := by
  rw [innerTail675_0, scan, if_neg (by decide), innerCandidates675_0_checked]
  change scan edge (search edge 3) 4 [5077] 5077 innerTail675_1
    (if 3 ≤ innerCandidates675_0.length then search edge 3 [5077, 5779] innerCandidates675_0 10856 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan675_1

theorem rootBranch675 : search edge 4 [5077] rootCandidates675 5077 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5077] 5077 innerTail675_0 best = best
  exact innerScan675_0

end ProjectEulerSolutions.PrimePairUpper.Search
