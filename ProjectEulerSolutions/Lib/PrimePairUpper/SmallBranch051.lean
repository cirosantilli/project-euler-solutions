import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates646 : List Nat := [5021, 5741, 5783, 5903, 6173, 6197, 7127]
set_option maxRecDepth 4096 in
theorem rootCandidates646_checked : candidates edge 5 0 4817 tail647 best.bound = rootCandidates646 := by
  unfold candidates
  have row : edge 4817 = SearchData.row4817.contains := by funext q; rfl
  rw [row]
  decide

def innerTail646_7 : List Nat := []
theorem innerScan646_7 : scan edge (search edge 3) 4 [4817] 4817 innerTail646_7 best = best := rfl

def innerTail646_6 : List Nat := 7127 :: innerTail646_7
def innerCandidates646_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates646_6_checked : candidates edge 4 4817 7127 innerTail646_7 best.bound = innerCandidates646_6 := by
  unfold candidates
  have row : edge 7127 = SearchData.row7127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan646_6 : scan edge (search edge 3) 4 [4817] 4817 innerTail646_6 best = best := by
  rw [innerTail646_6, scan, if_neg (by decide), innerCandidates646_6_checked]
  change scan edge (search edge 3) 4 [4817] 4817 innerTail646_7
    (if 3 ≤ innerCandidates646_6.length then search edge 3 [4817, 7127] innerCandidates646_6 11944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan646_7

def innerTail646_5 : List Nat := 6197 :: innerTail646_6
def innerCandidates646_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates646_5_checked : candidates edge 4 4817 6197 innerTail646_6 best.bound = innerCandidates646_5 := by
  unfold candidates
  have row : edge 6197 = SearchData.row6197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan646_5 : scan edge (search edge 3) 4 [4817] 4817 innerTail646_5 best = best := by
  rw [innerTail646_5, scan, if_neg (by decide), innerCandidates646_5_checked]
  change scan edge (search edge 3) 4 [4817] 4817 innerTail646_6
    (if 3 ≤ innerCandidates646_5.length then search edge 3 [4817, 6197] innerCandidates646_5 11014 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan646_6

def innerTail646_4 : List Nat := 6173 :: innerTail646_5
def innerCandidates646_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates646_4_checked : candidates edge 4 4817 6173 innerTail646_5 best.bound = innerCandidates646_4 := by
  unfold candidates
  have row : edge 6173 = SearchData.row6173.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan646_4 : scan edge (search edge 3) 4 [4817] 4817 innerTail646_4 best = best := by
  rw [innerTail646_4, scan, if_neg (by decide), innerCandidates646_4_checked]
  change scan edge (search edge 3) 4 [4817] 4817 innerTail646_5
    (if 3 ≤ innerCandidates646_4.length then search edge 3 [4817, 6173] innerCandidates646_4 10990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan646_5

def innerTail646_3 : List Nat := 5903 :: innerTail646_4
def innerCandidates646_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates646_3_checked : candidates edge 4 4817 5903 innerTail646_4 best.bound = innerCandidates646_3 := by
  unfold candidates
  have row : edge 5903 = SearchData.row5903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan646_3 : scan edge (search edge 3) 4 [4817] 4817 innerTail646_3 best = best := by
  rw [innerTail646_3, scan, if_neg (by decide), innerCandidates646_3_checked]
  change scan edge (search edge 3) 4 [4817] 4817 innerTail646_4
    (if 3 ≤ innerCandidates646_3.length then search edge 3 [4817, 5903] innerCandidates646_3 10720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan646_4

def innerTail646_2 : List Nat := 5783 :: innerTail646_3
def innerCandidates646_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates646_2_checked : candidates edge 4 4817 5783 innerTail646_3 best.bound = innerCandidates646_2 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan646_2 : scan edge (search edge 3) 4 [4817] 4817 innerTail646_2 best = best := by
  rw [innerTail646_2, scan, if_neg (by decide), innerCandidates646_2_checked]
  change scan edge (search edge 3) 4 [4817] 4817 innerTail646_3
    (if 3 ≤ innerCandidates646_2.length then search edge 3 [4817, 5783] innerCandidates646_2 10600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan646_3

def innerTail646_1 : List Nat := 5741 :: innerTail646_2
def innerCandidates646_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates646_1_checked : candidates edge 4 4817 5741 innerTail646_2 best.bound = innerCandidates646_1 := by
  unfold candidates
  have row : edge 5741 = SearchData.row5741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan646_1 : scan edge (search edge 3) 4 [4817] 4817 innerTail646_1 best = best := by
  rw [innerTail646_1, scan, if_neg (by decide), innerCandidates646_1_checked]
  change scan edge (search edge 3) 4 [4817] 4817 innerTail646_2
    (if 3 ≤ innerCandidates646_1.length then search edge 3 [4817, 5741] innerCandidates646_1 10558 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan646_2

def innerTail646_0 : List Nat := 5021 :: innerTail646_1
def innerCandidates646_0 : List Nat := [5741, 7127]
set_option maxRecDepth 4096 in
theorem innerCandidates646_0_checked : candidates edge 4 4817 5021 innerTail646_1 best.bound = innerCandidates646_0 := by
  unfold candidates
  have row : edge 5021 = SearchData.row5021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan646_0 : scan edge (search edge 3) 4 [4817] 4817 innerTail646_0 best = best := by
  rw [innerTail646_0, scan, if_neg (by decide), innerCandidates646_0_checked]
  change scan edge (search edge 3) 4 [4817] 4817 innerTail646_1
    (if 3 ≤ innerCandidates646_0.length then search edge 3 [4817, 5021] innerCandidates646_0 9838 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan646_1

theorem rootBranch646 : search edge 4 [4817] rootCandidates646 4817 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4817] 4817 innerTail646_0 best = best
  exact innerScan646_0

def rootCandidates645 : List Nat := [5323, 5701, 5827, 5869, 6427, 6661, 7057, 7351, 8179]
set_option maxRecDepth 4096 in
theorem rootCandidates645_checked : candidates edge 5 0 4813 tail646 best.bound = rootCandidates645 := by
  unfold candidates
  have row : edge 4813 = SearchData.row4813.contains := by funext q; rfl
  rw [row]
  decide

def innerTail645_9 : List Nat := []
theorem innerScan645_9 : scan edge (search edge 3) 4 [4813] 4813 innerTail645_9 best = best := rfl

def innerTail645_8 : List Nat := 8179 :: innerTail645_9
def innerCandidates645_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates645_8_checked : candidates edge 4 4813 8179 innerTail645_9 best.bound = innerCandidates645_8 := by
  unfold candidates
  have row : edge 8179 = SearchData.row8179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan645_8 : scan edge (search edge 3) 4 [4813] 4813 innerTail645_8 best = best := by
  rw [innerTail645_8, scan, if_neg (by decide), innerCandidates645_8_checked]
  change scan edge (search edge 3) 4 [4813] 4813 innerTail645_9
    (if 3 ≤ innerCandidates645_8.length then search edge 3 [4813, 8179] innerCandidates645_8 12992 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan645_9

def innerTail645_7 : List Nat := 7351 :: innerTail645_8
def innerCandidates645_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates645_7_checked : candidates edge 4 4813 7351 innerTail645_8 best.bound = innerCandidates645_7 := by
  unfold candidates
  have row : edge 7351 = SearchData.row7351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan645_7 : scan edge (search edge 3) 4 [4813] 4813 innerTail645_7 best = best := by
  rw [innerTail645_7, scan, if_neg (by decide), innerCandidates645_7_checked]
  change scan edge (search edge 3) 4 [4813] 4813 innerTail645_8
    (if 3 ≤ innerCandidates645_7.length then search edge 3 [4813, 7351] innerCandidates645_7 12164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan645_8

def innerTail645_6 : List Nat := 7057 :: innerTail645_7
def innerCandidates645_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates645_6_checked : candidates edge 4 4813 7057 innerTail645_7 best.bound = innerCandidates645_6 := by
  unfold candidates
  have row : edge 7057 = SearchData.row7057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan645_6 : scan edge (search edge 3) 4 [4813] 4813 innerTail645_6 best = best := by
  rw [innerTail645_6, scan, if_neg (by decide), innerCandidates645_6_checked]
  change scan edge (search edge 3) 4 [4813] 4813 innerTail645_7
    (if 3 ≤ innerCandidates645_6.length then search edge 3 [4813, 7057] innerCandidates645_6 11870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan645_7

def innerTail645_5 : List Nat := 6661 :: innerTail645_6
def innerCandidates645_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates645_5_checked : candidates edge 4 4813 6661 innerTail645_6 best.bound = innerCandidates645_5 := by
  unfold candidates
  have row : edge 6661 = SearchData.row6661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan645_5 : scan edge (search edge 3) 4 [4813] 4813 innerTail645_5 best = best := by
  rw [innerTail645_5, scan, if_neg (by decide), innerCandidates645_5_checked]
  change scan edge (search edge 3) 4 [4813] 4813 innerTail645_6
    (if 3 ≤ innerCandidates645_5.length then search edge 3 [4813, 6661] innerCandidates645_5 11474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan645_6

def innerTail645_4 : List Nat := 6427 :: innerTail645_5
def innerCandidates645_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates645_4_checked : candidates edge 4 4813 6427 innerTail645_5 best.bound = innerCandidates645_4 := by
  unfold candidates
  have row : edge 6427 = SearchData.row6427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan645_4 : scan edge (search edge 3) 4 [4813] 4813 innerTail645_4 best = best := by
  rw [innerTail645_4, scan, if_neg (by decide), innerCandidates645_4_checked]
  change scan edge (search edge 3) 4 [4813] 4813 innerTail645_5
    (if 3 ≤ innerCandidates645_4.length then search edge 3 [4813, 6427] innerCandidates645_4 11240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan645_5

def innerTail645_3 : List Nat := 5869 :: innerTail645_4
def innerCandidates645_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates645_3_checked : candidates edge 4 4813 5869 innerTail645_4 best.bound = innerCandidates645_3 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan645_3 : scan edge (search edge 3) 4 [4813] 4813 innerTail645_3 best = best := by
  rw [innerTail645_3, scan, if_neg (by decide), innerCandidates645_3_checked]
  change scan edge (search edge 3) 4 [4813] 4813 innerTail645_4
    (if 3 ≤ innerCandidates645_3.length then search edge 3 [4813, 5869] innerCandidates645_3 10682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan645_4

def innerTail645_2 : List Nat := 5827 :: innerTail645_3
def innerCandidates645_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates645_2_checked : candidates edge 4 4813 5827 innerTail645_3 best.bound = innerCandidates645_2 := by
  unfold candidates
  have row : edge 5827 = SearchData.row5827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan645_2 : scan edge (search edge 3) 4 [4813] 4813 innerTail645_2 best = best := by
  rw [innerTail645_2, scan, if_neg (by decide), innerCandidates645_2_checked]
  change scan edge (search edge 3) 4 [4813] 4813 innerTail645_3
    (if 3 ≤ innerCandidates645_2.length then search edge 3 [4813, 5827] innerCandidates645_2 10640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan645_3

def innerTail645_1 : List Nat := 5701 :: innerTail645_2
def innerCandidates645_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates645_1_checked : candidates edge 4 4813 5701 innerTail645_2 best.bound = innerCandidates645_1 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan645_1 : scan edge (search edge 3) 4 [4813] 4813 innerTail645_1 best = best := by
  rw [innerTail645_1, scan, if_neg (by decide), innerCandidates645_1_checked]
  change scan edge (search edge 3) 4 [4813] 4813 innerTail645_2
    (if 3 ≤ innerCandidates645_1.length then search edge 3 [4813, 5701] innerCandidates645_1 10514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan645_2

def innerTail645_0 : List Nat := 5323 :: innerTail645_1
def innerCandidates645_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates645_0_checked : candidates edge 4 4813 5323 innerTail645_1 best.bound = innerCandidates645_0 := by
  unfold candidates
  have row : edge 5323 = SearchData.row5323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan645_0 : scan edge (search edge 3) 4 [4813] 4813 innerTail645_0 best = best := by
  rw [innerTail645_0, scan, if_neg (by decide), innerCandidates645_0_checked]
  change scan edge (search edge 3) 4 [4813] 4813 innerTail645_1
    (if 3 ≤ innerCandidates645_0.length then search edge 3 [4813, 5323] innerCandidates645_0 10136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan645_1

theorem rootBranch645 : search edge 4 [4813] rootCandidates645 4813 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4813] 4813 innerTail645_0 best = best
  exact innerScan645_0

def rootCandidates644 : List Nat := [5197, 5419, 5839, 6229, 6361, 6733, 6961, 7159, 7927, 7993, 8317]
set_option maxRecDepth 4096 in
theorem rootCandidates644_checked : candidates edge 5 0 4801 tail645 best.bound = rootCandidates644 := by
  unfold candidates
  have row : edge 4801 = SearchData.row4801.contains := by funext q; rfl
  rw [row]
  decide

def innerTail644_11 : List Nat := []
theorem innerScan644_11 : scan edge (search edge 3) 4 [4801] 4801 innerTail644_11 best = best := rfl

def innerTail644_10 : List Nat := 8317 :: innerTail644_11
def innerCandidates644_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates644_10_checked : candidates edge 4 4801 8317 innerTail644_11 best.bound = innerCandidates644_10 := by
  unfold candidates
  have row : edge 8317 = SearchData.row8317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan644_10 : scan edge (search edge 3) 4 [4801] 4801 innerTail644_10 best = best := by
  rw [innerTail644_10, scan, if_neg (by decide), innerCandidates644_10_checked]
  change scan edge (search edge 3) 4 [4801] 4801 innerTail644_11
    (if 3 ≤ innerCandidates644_10.length then search edge 3 [4801, 8317] innerCandidates644_10 13118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan644_11

def innerTail644_9 : List Nat := 7993 :: innerTail644_10
def innerCandidates644_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates644_9_checked : candidates edge 4 4801 7993 innerTail644_10 best.bound = innerCandidates644_9 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan644_9 : scan edge (search edge 3) 4 [4801] 4801 innerTail644_9 best = best := by
  rw [innerTail644_9, scan, if_neg (by decide), innerCandidates644_9_checked]
  change scan edge (search edge 3) 4 [4801] 4801 innerTail644_10
    (if 3 ≤ innerCandidates644_9.length then search edge 3 [4801, 7993] innerCandidates644_9 12794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan644_10

def innerTail644_8 : List Nat := 7927 :: innerTail644_9
def innerCandidates644_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates644_8_checked : candidates edge 4 4801 7927 innerTail644_9 best.bound = innerCandidates644_8 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan644_8 : scan edge (search edge 3) 4 [4801] 4801 innerTail644_8 best = best := by
  rw [innerTail644_8, scan, if_neg (by decide), innerCandidates644_8_checked]
  change scan edge (search edge 3) 4 [4801] 4801 innerTail644_9
    (if 3 ≤ innerCandidates644_8.length then search edge 3 [4801, 7927] innerCandidates644_8 12728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan644_9

def innerTail644_7 : List Nat := 7159 :: innerTail644_8
def innerCandidates644_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates644_7_checked : candidates edge 4 4801 7159 innerTail644_8 best.bound = innerCandidates644_7 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan644_7 : scan edge (search edge 3) 4 [4801] 4801 innerTail644_7 best = best := by
  rw [innerTail644_7, scan, if_neg (by decide), innerCandidates644_7_checked]
  change scan edge (search edge 3) 4 [4801] 4801 innerTail644_8
    (if 3 ≤ innerCandidates644_7.length then search edge 3 [4801, 7159] innerCandidates644_7 11960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan644_8

def innerTail644_6 : List Nat := 6961 :: innerTail644_7
def innerCandidates644_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates644_6_checked : candidates edge 4 4801 6961 innerTail644_7 best.bound = innerCandidates644_6 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan644_6 : scan edge (search edge 3) 4 [4801] 4801 innerTail644_6 best = best := by
  rw [innerTail644_6, scan, if_neg (by decide), innerCandidates644_6_checked]
  change scan edge (search edge 3) 4 [4801] 4801 innerTail644_7
    (if 3 ≤ innerCandidates644_6.length then search edge 3 [4801, 6961] innerCandidates644_6 11762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan644_7

def innerTail644_5 : List Nat := 6733 :: innerTail644_6
def innerCandidates644_5 : List Nat := [7993]
set_option maxRecDepth 4096 in
theorem innerCandidates644_5_checked : candidates edge 4 4801 6733 innerTail644_6 best.bound = innerCandidates644_5 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan644_5 : scan edge (search edge 3) 4 [4801] 4801 innerTail644_5 best = best := by
  rw [innerTail644_5, scan, if_neg (by decide), innerCandidates644_5_checked]
  change scan edge (search edge 3) 4 [4801] 4801 innerTail644_6
    (if 3 ≤ innerCandidates644_5.length then search edge 3 [4801, 6733] innerCandidates644_5 11534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan644_6

def innerTail644_4 : List Nat := 6361 :: innerTail644_5
def innerCandidates644_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates644_4_checked : candidates edge 4 4801 6361 innerTail644_5 best.bound = innerCandidates644_4 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan644_4 : scan edge (search edge 3) 4 [4801] 4801 innerTail644_4 best = best := by
  rw [innerTail644_4, scan, if_neg (by decide), innerCandidates644_4_checked]
  change scan edge (search edge 3) 4 [4801] 4801 innerTail644_5
    (if 3 ≤ innerCandidates644_4.length then search edge 3 [4801, 6361] innerCandidates644_4 11162 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan644_5

def innerTail644_3 : List Nat := 6229 :: innerTail644_4
def innerCandidates644_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates644_3_checked : candidates edge 4 4801 6229 innerTail644_4 best.bound = innerCandidates644_3 := by
  unfold candidates
  have row : edge 6229 = SearchData.row6229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan644_3 : scan edge (search edge 3) 4 [4801] 4801 innerTail644_3 best = best := by
  rw [innerTail644_3, scan, if_neg (by decide), innerCandidates644_3_checked]
  change scan edge (search edge 3) 4 [4801] 4801 innerTail644_4
    (if 3 ≤ innerCandidates644_3.length then search edge 3 [4801, 6229] innerCandidates644_3 11030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan644_4

def innerTail644_2 : List Nat := 5839 :: innerTail644_3
def innerCandidates644_2 : List Nat := [6733]
set_option maxRecDepth 4096 in
theorem innerCandidates644_2_checked : candidates edge 4 4801 5839 innerTail644_3 best.bound = innerCandidates644_2 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan644_2 : scan edge (search edge 3) 4 [4801] 4801 innerTail644_2 best = best := by
  rw [innerTail644_2, scan, if_neg (by decide), innerCandidates644_2_checked]
  change scan edge (search edge 3) 4 [4801] 4801 innerTail644_3
    (if 3 ≤ innerCandidates644_2.length then search edge 3 [4801, 5839] innerCandidates644_2 10640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan644_3

def innerTail644_1 : List Nat := 5419 :: innerTail644_2
def innerCandidates644_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates644_1_checked : candidates edge 4 4801 5419 innerTail644_2 best.bound = innerCandidates644_1 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan644_1 : scan edge (search edge 3) 4 [4801] 4801 innerTail644_1 best = best := by
  rw [innerTail644_1, scan, if_neg (by decide), innerCandidates644_1_checked]
  change scan edge (search edge 3) 4 [4801] 4801 innerTail644_2
    (if 3 ≤ innerCandidates644_1.length then search edge 3 [4801, 5419] innerCandidates644_1 10220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan644_2

def innerTail644_0 : List Nat := 5197 :: innerTail644_1
def innerCandidates644_0 : List Nat := [6733]
set_option maxRecDepth 4096 in
theorem innerCandidates644_0_checked : candidates edge 4 4801 5197 innerTail644_1 best.bound = innerCandidates644_0 := by
  unfold candidates
  have row : edge 5197 = SearchData.row5197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan644_0 : scan edge (search edge 3) 4 [4801] 4801 innerTail644_0 best = best := by
  rw [innerTail644_0, scan, if_neg (by decide), innerCandidates644_0_checked]
  change scan edge (search edge 3) 4 [4801] 4801 innerTail644_1
    (if 3 ≤ innerCandidates644_0.length then search edge 3 [4801, 5197] innerCandidates644_0 9998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan644_1

theorem rootBranch644 : search edge 4 [4801] rootCandidates644 4801 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4801] 4801 innerTail644_0 best = best
  exact innerScan644_0

def rootCandidates643 : List Nat := [4889, 5231, 5531, 5639, 5867, 6329, 7487, 7583]
set_option maxRecDepth 4096 in
theorem rootCandidates643_checked : candidates edge 5 0 4799 tail644 best.bound = rootCandidates643 := by
  unfold candidates
  have row : edge 4799 = SearchData.row4799.contains := by funext q; rfl
  rw [row]
  decide

def innerTail643_8 : List Nat := []
theorem innerScan643_8 : scan edge (search edge 3) 4 [4799] 4799 innerTail643_8 best = best := rfl

def innerTail643_7 : List Nat := 7583 :: innerTail643_8
def innerCandidates643_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates643_7_checked : candidates edge 4 4799 7583 innerTail643_8 best.bound = innerCandidates643_7 := by
  unfold candidates
  have row : edge 7583 = SearchData.row7583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan643_7 : scan edge (search edge 3) 4 [4799] 4799 innerTail643_7 best = best := by
  rw [innerTail643_7, scan, if_neg (by decide), innerCandidates643_7_checked]
  change scan edge (search edge 3) 4 [4799] 4799 innerTail643_8
    (if 3 ≤ innerCandidates643_7.length then search edge 3 [4799, 7583] innerCandidates643_7 12382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan643_8

def innerTail643_6 : List Nat := 7487 :: innerTail643_7
def innerCandidates643_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates643_6_checked : candidates edge 4 4799 7487 innerTail643_7 best.bound = innerCandidates643_6 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan643_6 : scan edge (search edge 3) 4 [4799] 4799 innerTail643_6 best = best := by
  rw [innerTail643_6, scan, if_neg (by decide), innerCandidates643_6_checked]
  change scan edge (search edge 3) 4 [4799] 4799 innerTail643_7
    (if 3 ≤ innerCandidates643_6.length then search edge 3 [4799, 7487] innerCandidates643_6 12286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan643_7

def innerTail643_5 : List Nat := 6329 :: innerTail643_6
def innerCandidates643_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates643_5_checked : candidates edge 4 4799 6329 innerTail643_6 best.bound = innerCandidates643_5 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan643_5 : scan edge (search edge 3) 4 [4799] 4799 innerTail643_5 best = best := by
  rw [innerTail643_5, scan, if_neg (by decide), innerCandidates643_5_checked]
  change scan edge (search edge 3) 4 [4799] 4799 innerTail643_6
    (if 3 ≤ innerCandidates643_5.length then search edge 3 [4799, 6329] innerCandidates643_5 11128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan643_6

def innerTail643_4 : List Nat := 5867 :: innerTail643_5
def innerCandidates643_4 : List Nat := [6329]
set_option maxRecDepth 4096 in
theorem innerCandidates643_4_checked : candidates edge 4 4799 5867 innerTail643_5 best.bound = innerCandidates643_4 := by
  unfold candidates
  have row : edge 5867 = SearchData.row5867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan643_4 : scan edge (search edge 3) 4 [4799] 4799 innerTail643_4 best = best := by
  rw [innerTail643_4, scan, if_neg (by decide), innerCandidates643_4_checked]
  change scan edge (search edge 3) 4 [4799] 4799 innerTail643_5
    (if 3 ≤ innerCandidates643_4.length then search edge 3 [4799, 5867] innerCandidates643_4 10666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan643_5

def innerTail643_3 : List Nat := 5639 :: innerTail643_4
def innerCandidates643_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates643_3_checked : candidates edge 4 4799 5639 innerTail643_4 best.bound = innerCandidates643_3 := by
  unfold candidates
  have row : edge 5639 = SearchData.row5639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan643_3 : scan edge (search edge 3) 4 [4799] 4799 innerTail643_3 best = best := by
  rw [innerTail643_3, scan, if_neg (by decide), innerCandidates643_3_checked]
  change scan edge (search edge 3) 4 [4799] 4799 innerTail643_4
    (if 3 ≤ innerCandidates643_3.length then search edge 3 [4799, 5639] innerCandidates643_3 10438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan643_4

def innerTail643_2 : List Nat := 5531 :: innerTail643_3
def innerCandidates643_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates643_2_checked : candidates edge 4 4799 5531 innerTail643_3 best.bound = innerCandidates643_2 := by
  unfold candidates
  have row : edge 5531 = SearchData.row5531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan643_2 : scan edge (search edge 3) 4 [4799] 4799 innerTail643_2 best = best := by
  rw [innerTail643_2, scan, if_neg (by decide), innerCandidates643_2_checked]
  change scan edge (search edge 3) 4 [4799] 4799 innerTail643_3
    (if 3 ≤ innerCandidates643_2.length then search edge 3 [4799, 5531] innerCandidates643_2 10330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan643_3

def innerTail643_1 : List Nat := 5231 :: innerTail643_2
def innerCandidates643_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates643_1_checked : candidates edge 4 4799 5231 innerTail643_2 best.bound = innerCandidates643_1 := by
  unfold candidates
  have row : edge 5231 = SearchData.row5231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan643_1 : scan edge (search edge 3) 4 [4799] 4799 innerTail643_1 best = best := by
  rw [innerTail643_1, scan, if_neg (by decide), innerCandidates643_1_checked]
  change scan edge (search edge 3) 4 [4799] 4799 innerTail643_2
    (if 3 ≤ innerCandidates643_1.length then search edge 3 [4799, 5231] innerCandidates643_1 10030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan643_2

def innerTail643_0 : List Nat := 4889 :: innerTail643_1
def innerCandidates643_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates643_0_checked : candidates edge 4 4799 4889 innerTail643_1 best.bound = innerCandidates643_0 := by
  unfold candidates
  have row : edge 4889 = SearchData.row4889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan643_0 : scan edge (search edge 3) 4 [4799] 4799 innerTail643_0 best = best := by
  rw [innerTail643_0, scan, if_neg (by decide), innerCandidates643_0_checked]
  change scan edge (search edge 3) 4 [4799] 4799 innerTail643_1
    (if 3 ≤ innerCandidates643_0.length then search edge 3 [4799, 4889] innerCandidates643_0 9688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan643_1

theorem rootBranch643 : search edge 4 [4799] rootCandidates643 4799 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4799] 4799 innerTail643_0 best = best
  exact innerScan643_0

def rootCandidates642 : List Nat := [4871, 5003, 5099, 5399, 5501, 6389, 6869, 7013, 7481, 8123]
set_option maxRecDepth 4096 in
theorem rootCandidates642_checked : candidates edge 5 0 4793 tail643 best.bound = rootCandidates642 := by
  unfold candidates
  have row : edge 4793 = SearchData.row4793.contains := by funext q; rfl
  rw [row]
  decide

def innerTail642_10 : List Nat := []
theorem innerScan642_10 : scan edge (search edge 3) 4 [4793] 4793 innerTail642_10 best = best := rfl

def innerTail642_9 : List Nat := 8123 :: innerTail642_10
def innerCandidates642_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates642_9_checked : candidates edge 4 4793 8123 innerTail642_10 best.bound = innerCandidates642_9 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan642_9 : scan edge (search edge 3) 4 [4793] 4793 innerTail642_9 best = best := by
  rw [innerTail642_9, scan, if_neg (by decide), innerCandidates642_9_checked]
  change scan edge (search edge 3) 4 [4793] 4793 innerTail642_10
    (if 3 ≤ innerCandidates642_9.length then search edge 3 [4793, 8123] innerCandidates642_9 12916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan642_10

def innerTail642_8 : List Nat := 7481 :: innerTail642_9
def innerCandidates642_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates642_8_checked : candidates edge 4 4793 7481 innerTail642_9 best.bound = innerCandidates642_8 := by
  unfold candidates
  have row : edge 7481 = SearchData.row7481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan642_8 : scan edge (search edge 3) 4 [4793] 4793 innerTail642_8 best = best := by
  rw [innerTail642_8, scan, if_neg (by decide), innerCandidates642_8_checked]
  change scan edge (search edge 3) 4 [4793] 4793 innerTail642_9
    (if 3 ≤ innerCandidates642_8.length then search edge 3 [4793, 7481] innerCandidates642_8 12274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan642_9

def innerTail642_7 : List Nat := 7013 :: innerTail642_8
def innerCandidates642_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates642_7_checked : candidates edge 4 4793 7013 innerTail642_8 best.bound = innerCandidates642_7 := by
  unfold candidates
  have row : edge 7013 = SearchData.row7013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan642_7 : scan edge (search edge 3) 4 [4793] 4793 innerTail642_7 best = best := by
  rw [innerTail642_7, scan, if_neg (by decide), innerCandidates642_7_checked]
  change scan edge (search edge 3) 4 [4793] 4793 innerTail642_8
    (if 3 ≤ innerCandidates642_7.length then search edge 3 [4793, 7013] innerCandidates642_7 11806 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan642_8

def innerTail642_6 : List Nat := 6869 :: innerTail642_7
def innerCandidates642_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates642_6_checked : candidates edge 4 4793 6869 innerTail642_7 best.bound = innerCandidates642_6 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan642_6 : scan edge (search edge 3) 4 [4793] 4793 innerTail642_6 best = best := by
  rw [innerTail642_6, scan, if_neg (by decide), innerCandidates642_6_checked]
  change scan edge (search edge 3) 4 [4793] 4793 innerTail642_7
    (if 3 ≤ innerCandidates642_6.length then search edge 3 [4793, 6869] innerCandidates642_6 11662 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan642_7

def innerTail642_5 : List Nat := 6389 :: innerTail642_6
def innerCandidates642_5 : List Nat := [7481]
set_option maxRecDepth 4096 in
theorem innerCandidates642_5_checked : candidates edge 4 4793 6389 innerTail642_6 best.bound = innerCandidates642_5 := by
  unfold candidates
  have row : edge 6389 = SearchData.row6389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan642_5 : scan edge (search edge 3) 4 [4793] 4793 innerTail642_5 best = best := by
  rw [innerTail642_5, scan, if_neg (by decide), innerCandidates642_5_checked]
  change scan edge (search edge 3) 4 [4793] 4793 innerTail642_6
    (if 3 ≤ innerCandidates642_5.length then search edge 3 [4793, 6389] innerCandidates642_5 11182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan642_6

def innerTail642_4 : List Nat := 5501 :: innerTail642_5
def innerCandidates642_4 : List Nat := [7481]
set_option maxRecDepth 4096 in
theorem innerCandidates642_4_checked : candidates edge 4 4793 5501 innerTail642_5 best.bound = innerCandidates642_4 := by
  unfold candidates
  have row : edge 5501 = SearchData.row5501.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan642_4 : scan edge (search edge 3) 4 [4793] 4793 innerTail642_4 best = best := by
  rw [innerTail642_4, scan, if_neg (by decide), innerCandidates642_4_checked]
  change scan edge (search edge 3) 4 [4793] 4793 innerTail642_5
    (if 3 ≤ innerCandidates642_4.length then search edge 3 [4793, 5501] innerCandidates642_4 10294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan642_5

def innerTail642_3 : List Nat := 5399 :: innerTail642_4
def innerCandidates642_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates642_3_checked : candidates edge 4 4793 5399 innerTail642_4 best.bound = innerCandidates642_3 := by
  unfold candidates
  have row : edge 5399 = SearchData.row5399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan642_3 : scan edge (search edge 3) 4 [4793] 4793 innerTail642_3 best = best := by
  rw [innerTail642_3, scan, if_neg (by decide), innerCandidates642_3_checked]
  change scan edge (search edge 3) 4 [4793] 4793 innerTail642_4
    (if 3 ≤ innerCandidates642_3.length then search edge 3 [4793, 5399] innerCandidates642_3 10192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan642_4

def innerTail642_2 : List Nat := 5099 :: innerTail642_3
def innerCandidates642_2 : List Nat := [5501]
set_option maxRecDepth 4096 in
theorem innerCandidates642_2_checked : candidates edge 4 4793 5099 innerTail642_3 best.bound = innerCandidates642_2 := by
  unfold candidates
  have row : edge 5099 = SearchData.row5099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan642_2 : scan edge (search edge 3) 4 [4793] 4793 innerTail642_2 best = best := by
  rw [innerTail642_2, scan, if_neg (by decide), innerCandidates642_2_checked]
  change scan edge (search edge 3) 4 [4793] 4793 innerTail642_3
    (if 3 ≤ innerCandidates642_2.length then search edge 3 [4793, 5099] innerCandidates642_2 9892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan642_3

def innerTail642_1 : List Nat := 5003 :: innerTail642_2
def innerCandidates642_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates642_1_checked : candidates edge 4 4793 5003 innerTail642_2 best.bound = innerCandidates642_1 := by
  unfold candidates
  have row : edge 5003 = SearchData.row5003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan642_1 : scan edge (search edge 3) 4 [4793] 4793 innerTail642_1 best = best := by
  rw [innerTail642_1, scan, if_neg (by decide), innerCandidates642_1_checked]
  change scan edge (search edge 3) 4 [4793] 4793 innerTail642_2
    (if 3 ≤ innerCandidates642_1.length then search edge 3 [4793, 5003] innerCandidates642_1 9796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan642_2

def innerTail642_0 : List Nat := 4871 :: innerTail642_1
def innerCandidates642_0 : List Nat := [6389, 8123]
set_option maxRecDepth 4096 in
theorem innerCandidates642_0_checked : candidates edge 4 4793 4871 innerTail642_1 best.bound = innerCandidates642_0 := by
  unfold candidates
  have row : edge 4871 = SearchData.row4871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan642_0 : scan edge (search edge 3) 4 [4793] 4793 innerTail642_0 best = best := by
  rw [innerTail642_0, scan, if_neg (by decide), innerCandidates642_0_checked]
  change scan edge (search edge 3) 4 [4793] 4793 innerTail642_1
    (if 3 ≤ innerCandidates642_0.length then search edge 3 [4793, 4871] innerCandidates642_0 9664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan642_1

theorem rootBranch642 : search edge 4 [4793] rootCandidates642 4793 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4793] 4793 innerTail642_0 best = best
  exact innerScan642_0

def rootCandidates641 : List Nat := [5119, 5179, 6547, 6553, 7561, 8269]
set_option maxRecDepth 4096 in
theorem rootCandidates641_checked : candidates edge 5 0 4789 tail642 best.bound = rootCandidates641 := by
  unfold candidates
  have row : edge 4789 = SearchData.row4789.contains := by funext q; rfl
  rw [row]
  decide

def innerTail641_6 : List Nat := []
theorem innerScan641_6 : scan edge (search edge 3) 4 [4789] 4789 innerTail641_6 best = best := rfl

def innerTail641_5 : List Nat := 8269 :: innerTail641_6
def innerCandidates641_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates641_5_checked : candidates edge 4 4789 8269 innerTail641_6 best.bound = innerCandidates641_5 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan641_5 : scan edge (search edge 3) 4 [4789] 4789 innerTail641_5 best = best := by
  rw [innerTail641_5, scan, if_neg (by decide), innerCandidates641_5_checked]
  change scan edge (search edge 3) 4 [4789] 4789 innerTail641_6
    (if 3 ≤ innerCandidates641_5.length then search edge 3 [4789, 8269] innerCandidates641_5 13058 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan641_6

def innerTail641_4 : List Nat := 7561 :: innerTail641_5
def innerCandidates641_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates641_4_checked : candidates edge 4 4789 7561 innerTail641_5 best.bound = innerCandidates641_4 := by
  unfold candidates
  have row : edge 7561 = SearchData.row7561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan641_4 : scan edge (search edge 3) 4 [4789] 4789 innerTail641_4 best = best := by
  rw [innerTail641_4, scan, if_neg (by decide), innerCandidates641_4_checked]
  change scan edge (search edge 3) 4 [4789] 4789 innerTail641_5
    (if 3 ≤ innerCandidates641_4.length then search edge 3 [4789, 7561] innerCandidates641_4 12350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan641_5

def innerTail641_3 : List Nat := 6553 :: innerTail641_4
def innerCandidates641_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates641_3_checked : candidates edge 4 4789 6553 innerTail641_4 best.bound = innerCandidates641_3 := by
  unfold candidates
  have row : edge 6553 = SearchData.row6553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan641_3 : scan edge (search edge 3) 4 [4789] 4789 innerTail641_3 best = best := by
  rw [innerTail641_3, scan, if_neg (by decide), innerCandidates641_3_checked]
  change scan edge (search edge 3) 4 [4789] 4789 innerTail641_4
    (if 3 ≤ innerCandidates641_3.length then search edge 3 [4789, 6553] innerCandidates641_3 11342 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan641_4

def innerTail641_2 : List Nat := 6547 :: innerTail641_3
def innerCandidates641_2 : List Nat := [7561]
set_option maxRecDepth 4096 in
theorem innerCandidates641_2_checked : candidates edge 4 4789 6547 innerTail641_3 best.bound = innerCandidates641_2 := by
  unfold candidates
  have row : edge 6547 = SearchData.row6547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan641_2 : scan edge (search edge 3) 4 [4789] 4789 innerTail641_2 best = best := by
  rw [innerTail641_2, scan, if_neg (by decide), innerCandidates641_2_checked]
  change scan edge (search edge 3) 4 [4789] 4789 innerTail641_3
    (if 3 ≤ innerCandidates641_2.length then search edge 3 [4789, 6547] innerCandidates641_2 11336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan641_3

def innerTail641_1 : List Nat := 5179 :: innerTail641_2
def innerCandidates641_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates641_1_checked : candidates edge 4 4789 5179 innerTail641_2 best.bound = innerCandidates641_1 := by
  unfold candidates
  have row : edge 5179 = SearchData.row5179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan641_1 : scan edge (search edge 3) 4 [4789] 4789 innerTail641_1 best = best := by
  rw [innerTail641_1, scan, if_neg (by decide), innerCandidates641_1_checked]
  change scan edge (search edge 3) 4 [4789] 4789 innerTail641_2
    (if 3 ≤ innerCandidates641_1.length then search edge 3 [4789, 5179] innerCandidates641_1 9968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan641_2

def innerTail641_0 : List Nat := 5119 :: innerTail641_1
def innerCandidates641_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates641_0_checked : candidates edge 4 4789 5119 innerTail641_1 best.bound = innerCandidates641_0 := by
  unfold candidates
  have row : edge 5119 = SearchData.row5119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan641_0 : scan edge (search edge 3) 4 [4789] 4789 innerTail641_0 best = best := by
  rw [innerTail641_0, scan, if_neg (by decide), innerCandidates641_0_checked]
  change scan edge (search edge 3) 4 [4789] 4789 innerTail641_1
    (if 3 ≤ innerCandidates641_0.length then search edge 3 [4789, 5119] innerCandidates641_0 9908 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan641_1

theorem rootBranch641 : search edge 4 [4789] rootCandidates641 4789 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4789] 4789 innerTail641_0 best = best
  exact innerScan641_0

def rootCandidates640 : List Nat := [5099, 5237, 5981, 6047, 7013, 7187, 7229, 7433, 7649, 8009]
set_option maxRecDepth 4096 in
theorem rootCandidates640_checked : candidates edge 5 0 4787 tail641 best.bound = rootCandidates640 := by
  unfold candidates
  have row : edge 4787 = SearchData.row4787.contains := by funext q; rfl
  rw [row]
  decide

def innerTail640_10 : List Nat := []
theorem innerScan640_10 : scan edge (search edge 3) 4 [4787] 4787 innerTail640_10 best = best := rfl

def innerTail640_9 : List Nat := 8009 :: innerTail640_10
def innerCandidates640_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates640_9_checked : candidates edge 4 4787 8009 innerTail640_10 best.bound = innerCandidates640_9 := by
  unfold candidates
  have row : edge 8009 = SearchData.row8009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan640_9 : scan edge (search edge 3) 4 [4787] 4787 innerTail640_9 best = best := by
  rw [innerTail640_9, scan, if_neg (by decide), innerCandidates640_9_checked]
  change scan edge (search edge 3) 4 [4787] 4787 innerTail640_10
    (if 3 ≤ innerCandidates640_9.length then search edge 3 [4787, 8009] innerCandidates640_9 12796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan640_10

def innerTail640_8 : List Nat := 7649 :: innerTail640_9
def innerCandidates640_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates640_8_checked : candidates edge 4 4787 7649 innerTail640_9 best.bound = innerCandidates640_8 := by
  unfold candidates
  have row : edge 7649 = SearchData.row7649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan640_8 : scan edge (search edge 3) 4 [4787] 4787 innerTail640_8 best = best := by
  rw [innerTail640_8, scan, if_neg (by decide), innerCandidates640_8_checked]
  change scan edge (search edge 3) 4 [4787] 4787 innerTail640_9
    (if 3 ≤ innerCandidates640_8.length then search edge 3 [4787, 7649] innerCandidates640_8 12436 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan640_9

def innerTail640_7 : List Nat := 7433 :: innerTail640_8
def innerCandidates640_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates640_7_checked : candidates edge 4 4787 7433 innerTail640_8 best.bound = innerCandidates640_7 := by
  unfold candidates
  have row : edge 7433 = SearchData.row7433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan640_7 : scan edge (search edge 3) 4 [4787] 4787 innerTail640_7 best = best := by
  rw [innerTail640_7, scan, if_neg (by decide), innerCandidates640_7_checked]
  change scan edge (search edge 3) 4 [4787] 4787 innerTail640_8
    (if 3 ≤ innerCandidates640_7.length then search edge 3 [4787, 7433] innerCandidates640_7 12220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan640_8

def innerTail640_6 : List Nat := 7229 :: innerTail640_7
def innerCandidates640_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates640_6_checked : candidates edge 4 4787 7229 innerTail640_7 best.bound = innerCandidates640_6 := by
  unfold candidates
  have row : edge 7229 = SearchData.row7229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan640_6 : scan edge (search edge 3) 4 [4787] 4787 innerTail640_6 best = best := by
  rw [innerTail640_6, scan, if_neg (by decide), innerCandidates640_6_checked]
  change scan edge (search edge 3) 4 [4787] 4787 innerTail640_7
    (if 3 ≤ innerCandidates640_6.length then search edge 3 [4787, 7229] innerCandidates640_6 12016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan640_7

def innerTail640_5 : List Nat := 7187 :: innerTail640_6
def innerCandidates640_5 : List Nat := [7229]
set_option maxRecDepth 4096 in
theorem innerCandidates640_5_checked : candidates edge 4 4787 7187 innerTail640_6 best.bound = innerCandidates640_5 := by
  unfold candidates
  have row : edge 7187 = SearchData.row7187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan640_5 : scan edge (search edge 3) 4 [4787] 4787 innerTail640_5 best = best := by
  rw [innerTail640_5, scan, if_neg (by decide), innerCandidates640_5_checked]
  change scan edge (search edge 3) 4 [4787] 4787 innerTail640_6
    (if 3 ≤ innerCandidates640_5.length then search edge 3 [4787, 7187] innerCandidates640_5 11974 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan640_6

def innerTail640_4 : List Nat := 7013 :: innerTail640_5
def innerCandidates640_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates640_4_checked : candidates edge 4 4787 7013 innerTail640_5 best.bound = innerCandidates640_4 := by
  unfold candidates
  have row : edge 7013 = SearchData.row7013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan640_4 : scan edge (search edge 3) 4 [4787] 4787 innerTail640_4 best = best := by
  rw [innerTail640_4, scan, if_neg (by decide), innerCandidates640_4_checked]
  change scan edge (search edge 3) 4 [4787] 4787 innerTail640_5
    (if 3 ≤ innerCandidates640_4.length then search edge 3 [4787, 7013] innerCandidates640_4 11800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan640_5

def innerTail640_3 : List Nat := 6047 :: innerTail640_4
def innerCandidates640_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates640_3_checked : candidates edge 4 4787 6047 innerTail640_4 best.bound = innerCandidates640_3 := by
  unfold candidates
  have row : edge 6047 = SearchData.row6047.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan640_3 : scan edge (search edge 3) 4 [4787] 4787 innerTail640_3 best = best := by
  rw [innerTail640_3, scan, if_neg (by decide), innerCandidates640_3_checked]
  change scan edge (search edge 3) 4 [4787] 4787 innerTail640_4
    (if 3 ≤ innerCandidates640_3.length then search edge 3 [4787, 6047] innerCandidates640_3 10834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan640_4

def innerTail640_2 : List Nat := 5981 :: innerTail640_3
def innerCandidates640_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates640_2_checked : candidates edge 4 4787 5981 innerTail640_3 best.bound = innerCandidates640_2 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan640_2 : scan edge (search edge 3) 4 [4787] 4787 innerTail640_2 best = best := by
  rw [innerTail640_2, scan, if_neg (by decide), innerCandidates640_2_checked]
  change scan edge (search edge 3) 4 [4787] 4787 innerTail640_3
    (if 3 ≤ innerCandidates640_2.length then search edge 3 [4787, 5981] innerCandidates640_2 10768 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan640_3

def innerTail640_1 : List Nat := 5237 :: innerTail640_2
def innerCandidates640_1 : List Nat := [8009]
set_option maxRecDepth 4096 in
theorem innerCandidates640_1_checked : candidates edge 4 4787 5237 innerTail640_2 best.bound = innerCandidates640_1 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan640_1 : scan edge (search edge 3) 4 [4787] 4787 innerTail640_1 best = best := by
  rw [innerTail640_1, scan, if_neg (by decide), innerCandidates640_1_checked]
  change scan edge (search edge 3) 4 [4787] 4787 innerTail640_2
    (if 3 ≤ innerCandidates640_1.length then search edge 3 [4787, 5237] innerCandidates640_1 10024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan640_2

def innerTail640_0 : List Nat := 5099 :: innerTail640_1
def innerCandidates640_0 : List Nat := [7649]
set_option maxRecDepth 4096 in
theorem innerCandidates640_0_checked : candidates edge 4 4787 5099 innerTail640_1 best.bound = innerCandidates640_0 := by
  unfold candidates
  have row : edge 5099 = SearchData.row5099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan640_0 : scan edge (search edge 3) 4 [4787] 4787 innerTail640_0 best = best := by
  rw [innerTail640_0, scan, if_neg (by decide), innerCandidates640_0_checked]
  change scan edge (search edge 3) 4 [4787] 4787 innerTail640_1
    (if 3 ≤ innerCandidates640_0.length then search edge 3 [4787, 5099] innerCandidates640_0 9886 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan640_1

theorem rootBranch640 : search edge 4 [4787] rootCandidates640 4787 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4787] 4787 innerTail640_0 best = best
  exact innerScan640_0

end ProjectEulerSolutions.PrimePairUpper.Search
