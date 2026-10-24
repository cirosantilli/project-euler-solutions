import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates887 : List Nat := [7211, 7331, 8147]
set_option maxRecDepth 4096 in
theorem rootCandidates887_checked : candidates edge 5 0 6917 tail888 best.bound = rootCandidates887 := by
  unfold candidates
  have row : edge 6917 = SearchData.row6917.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates886 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates886_checked : candidates edge 5 0 6911 tail887 best.bound = rootCandidates886 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates885 : List Nat := [7549, 7759]
set_option maxRecDepth 4096 in
theorem rootCandidates885_checked : candidates edge 5 0 6907 tail886 best.bound = rootCandidates885 := by
  unfold candidates
  have row : edge 6907 = SearchData.row6907.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates884 : List Nat := [6911, 7127, 7529]
set_option maxRecDepth 4096 in
theorem rootCandidates884_checked : candidates edge 5 0 6899 tail885 best.bound = rootCandidates884 := by
  unfold candidates
  have row : edge 6899 = SearchData.row6899.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates883 : List Nat := [7477, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates883_checked : candidates edge 5 0 6883 tail884 best.bound = rootCandidates883 := by
  unfold candidates
  have row : edge 6883 = SearchData.row6883.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates882 : List Nat := [7039, 8191]
set_option maxRecDepth 4096 in
theorem rootCandidates882_checked : candidates edge 5 0 6871 tail883 best.bound = rootCandidates882 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates881 : List Nat := [6959, 7121, 7583, 7907, 8087, 8111]
set_option maxRecDepth 4096 in
theorem rootCandidates881_checked : candidates edge 5 0 6869 tail882 best.bound = rootCandidates881 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext q; rfl
  rw [row]
  decide

def innerTail881_6 : List Nat := []
theorem innerScan881_6 : scan edge (search edge 3) 4 [6869] 6869 innerTail881_6 best = best := rfl

def innerTail881_5 : List Nat := 8111 :: innerTail881_6
def innerCandidates881_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates881_5_checked : candidates edge 4 6869 8111 innerTail881_6 best.bound = innerCandidates881_5 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan881_5 : scan edge (search edge 3) 4 [6869] 6869 innerTail881_5 best = best := by
  rw [innerTail881_5, scan, if_neg (by decide), innerCandidates881_5_checked]
  change scan edge (search edge 3) 4 [6869] 6869 innerTail881_6
    (if 3 ≤ innerCandidates881_5.length then search edge 3 [6869, 8111] innerCandidates881_5 14980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan881_6

def innerTail881_4 : List Nat := 8087 :: innerTail881_5
def innerCandidates881_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates881_4_checked : candidates edge 4 6869 8087 innerTail881_5 best.bound = innerCandidates881_4 := by
  unfold candidates
  have row : edge 8087 = SearchData.row8087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan881_4 : scan edge (search edge 3) 4 [6869] 6869 innerTail881_4 best = best := by
  rw [innerTail881_4, scan, if_neg (by decide), innerCandidates881_4_checked]
  change scan edge (search edge 3) 4 [6869] 6869 innerTail881_5
    (if 3 ≤ innerCandidates881_4.length then search edge 3 [6869, 8087] innerCandidates881_4 14956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan881_5

def innerTail881_3 : List Nat := 7907 :: innerTail881_4
def innerCandidates881_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates881_3_checked : candidates edge 4 6869 7907 innerTail881_4 best.bound = innerCandidates881_3 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan881_3 : scan edge (search edge 3) 4 [6869] 6869 innerTail881_3 best = best := by
  rw [innerTail881_3, scan, if_neg (by decide), innerCandidates881_3_checked]
  change scan edge (search edge 3) 4 [6869] 6869 innerTail881_4
    (if 3 ≤ innerCandidates881_3.length then search edge 3 [6869, 7907] innerCandidates881_3 14776 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan881_4

def innerTail881_2 : List Nat := 7583 :: innerTail881_3
def innerCandidates881_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates881_2_checked : candidates edge 4 6869 7583 innerTail881_3 best.bound = innerCandidates881_2 := by
  unfold candidates
  have row : edge 7583 = SearchData.row7583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan881_2 : scan edge (search edge 3) 4 [6869] 6869 innerTail881_2 best = best := by
  rw [innerTail881_2, scan, if_neg (by decide), innerCandidates881_2_checked]
  change scan edge (search edge 3) 4 [6869] 6869 innerTail881_3
    (if 3 ≤ innerCandidates881_2.length then search edge 3 [6869, 7583] innerCandidates881_2 14452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan881_3

def innerTail881_1 : List Nat := 7121 :: innerTail881_2
def innerCandidates881_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates881_1_checked : candidates edge 4 6869 7121 innerTail881_2 best.bound = innerCandidates881_1 := by
  unfold candidates
  have row : edge 7121 = SearchData.row7121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan881_1 : scan edge (search edge 3) 4 [6869] 6869 innerTail881_1 best = best := by
  rw [innerTail881_1, scan, if_neg (by decide), innerCandidates881_1_checked]
  change scan edge (search edge 3) 4 [6869] 6869 innerTail881_2
    (if 3 ≤ innerCandidates881_1.length then search edge 3 [6869, 7121] innerCandidates881_1 13990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan881_2

def innerTail881_0 : List Nat := 6959 :: innerTail881_1
def innerCandidates881_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates881_0_checked : candidates edge 4 6869 6959 innerTail881_1 best.bound = innerCandidates881_0 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan881_0 : scan edge (search edge 3) 4 [6869] 6869 innerTail881_0 best = best := by
  rw [innerTail881_0, scan, if_neg (by decide), innerCandidates881_0_checked]
  change scan edge (search edge 3) 4 [6869] 6869 innerTail881_1
    (if 3 ≤ innerCandidates881_0.length then search edge 3 [6869, 6959] innerCandidates881_0 13828 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan881_1

theorem rootBranch881 : search edge 4 [6869] rootCandidates881 6869 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6869] 6869 innerTail881_0 best = best
  exact innerScan881_0

def rootCandidates880 : List Nat := [8243]
set_option maxRecDepth 4096 in
theorem rootCandidates880_checked : candidates edge 5 0 6863 tail881 best.bound = rootCandidates880 := by
  unfold candidates
  have row : edge 6863 = SearchData.row6863.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
