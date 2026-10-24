import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root658Candidates : List Nat := Data.row4943
set_option maxRecDepth 8192 in
theorem root658Candidates_checked : candidates edge 5 0 4943 tail659 winner.bound = root658Candidates := by
  apply candidates_eq_row edge 5 0 4943 tail659 root658Candidates winner.bound tail659_ordered
  · decide
  · decide
  · funext q
    rfl
def root658Tail31 : List Nat := []
def root658Tail30 : List Nat := 21011 :: root658Tail31
def root658Tail29 : List Nat := 20297 :: root658Tail30
def root658Tail28 : List Nat := 20261 :: root658Tail29
def root658Tail27 : List Nat := 20219 :: root658Tail28
def root658Tail26 : List Nat := 19919 :: root658Tail27
def root658Tail25 : List Nat := 19559 :: root658Tail26
def root658Tail24 : List Nat := 19391 :: root658Tail25
def root658Tail23 : List Nat := 19379 :: root658Tail24
def root658Tail22 : List Nat := 18749 :: root658Tail23
def root658Tail21 : List Nat := 17573 :: root658Tail22
def root658Tail20 : List Nat := 15329 :: root658Tail21
def root658Tail19 : List Nat := 13877 :: root658Tail20
def root658Tail18 : List Nat := 13763 :: root658Tail19
def root658Tail17 : List Nat := 11801 :: root658Tail18
def root658Tail16 : List Nat := 10781 :: root658Tail17
def root658Tail15 : List Nat := 10709 :: root658Tail16
def root658Tail14 : List Nat := 10163 :: root658Tail15
def root658Tail13 : List Nat := 10151 :: root658Tail14
def root658Tail12 : List Nat := 8819 :: root658Tail13
def root658Tail11 : List Nat := 7919 :: root658Tail12
def root658Tail10 : List Nat := 7757 :: root658Tail11
def root658Tail9 : List Nat := 7517 :: root658Tail10
def root658Tail8 : List Nat := 7151 :: root658Tail9
def root658Tail7 : List Nat := 6701 :: root658Tail8
def root658Tail6 : List Nat := 6449 :: root658Tail7
def root658Tail5 : List Nat := 5741 :: root658Tail6
def root658Tail4 : List Nat := 5417 :: root658Tail5
def root658Tail3 : List Nat := 5261 :: root658Tail4
def root658Tail2 : List Nat := 5189 :: root658Tail3
def root658Tail1 : List Nat := 5039 :: root658Tail2
def root658Tail0 : List Nat := 5021 :: root658Tail1
def root658Candidates0 : List Nat := [5741, 10163]
set_option maxRecDepth 8192 in
theorem root658Candidates0_checked : candidates edge 4 4943 5021 root658Tail1 winner.bound = root658Candidates0 := by
  unfold candidates
  have row : edge 5021 = (SortedNatTree.contains FastRows.row5021) := by
    change Data.row5021.contains = (SortedNatTree.contains FastRows.row5021)
    exact FastRows.row5021_checked.symm
  rw [row]
  decide
theorem root658Step0 : scan edge (search edge 3) 4 [4943] 4943 root658Tail0 winner =
    scan edge (search edge 3) 4 [4943] 4943 root658Tail1 winner := by
  rw [root658Tail0, scan, if_neg (by decide), root658Candidates0_checked]
  change scan edge (search edge 3) 4 [4943] 4943 root658Tail1
    (if 3 ≤ root658Candidates0.length then search edge 3 [4943, 5021] root658Candidates0 9964 winner else winner) = _
  rw [if_neg (by decide)]
def root658Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root658Candidates1_checked : candidates edge 4 4943 5039 root658Tail2 winner.bound = root658Candidates1 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
theorem root658Step1 : scan edge (search edge 3) 4 [4943] 4943 root658Tail1 winner =
    scan edge (search edge 3) 4 [4943] 4943 root658Tail2 winner := by
  rw [root658Tail1, scan, if_neg (by decide), root658Candidates1_checked]
  change scan edge (search edge 3) 4 [4943] 4943 root658Tail2
    (if 3 ≤ root658Candidates1.length then search edge 3 [4943, 5039] root658Candidates1 9982 winner else winner) = _
  rw [if_neg (by decide)]
def root658Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root658Candidates2_checked : candidates edge 4 4943 5189 root658Tail3 winner.bound = root658Candidates2 := by
  unfold candidates
  have row : edge 5189 = (SortedNatTree.contains FastRows.row5189) := by
    change Data.row5189.contains = (SortedNatTree.contains FastRows.row5189)
    exact FastRows.row5189_checked.symm
  rw [row]
  decide
theorem root658Step2 : scan edge (search edge 3) 4 [4943] 4943 root658Tail2 winner =
    scan edge (search edge 3) 4 [4943] 4943 root658Tail3 winner := by
  rw [root658Tail2, scan, if_neg (by decide), root658Candidates2_checked]
  change scan edge (search edge 3) 4 [4943] 4943 root658Tail3
    (if 3 ≤ root658Candidates2.length then search edge 3 [4943, 5189] root658Candidates2 10132 winner else winner) = _
  rw [if_neg (by decide)]
def root658Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root658Candidates3_checked : candidates edge 4 4943 5261 root658Tail4 winner.bound = root658Candidates3 := by
  unfold candidates
  have row : edge 5261 = (SortedNatTree.contains FastRows.row5261) := by
    change Data.row5261.contains = (SortedNatTree.contains FastRows.row5261)
    exact FastRows.row5261_checked.symm
  rw [row]
  decide
theorem root658Step3 : scan edge (search edge 3) 4 [4943] 4943 root658Tail3 winner =
    scan edge (search edge 3) 4 [4943] 4943 root658Tail4 winner := by
  rw [root658Tail3, scan, if_neg (by decide), root658Candidates3_checked]
  change scan edge (search edge 3) 4 [4943] 4943 root658Tail4
    (if 3 ≤ root658Candidates3.length then search edge 3 [4943, 5261] root658Candidates3 10204 winner else winner) = _
  rw [if_neg (by decide)]
theorem root658Scan4 : scan edge (search edge 3) 4 [4943] 4943 root658Tail4 winner = winner := by rw [root658Tail4, scan, if_pos (by decide)]
theorem root658Scan3 : scan edge (search edge 3) 4 [4943] 4943 root658Tail3 winner = winner :=
  (root658Step3).trans root658Scan4
theorem root658Scan2 : scan edge (search edge 3) 4 [4943] 4943 root658Tail2 winner = winner :=
  (root658Step2).trans root658Scan3
theorem root658Scan1 : scan edge (search edge 3) 4 [4943] 4943 root658Tail1 winner = winner :=
  (root658Step1).trans root658Scan2
theorem root658Scan0 : scan edge (search edge 3) 4 [4943] 4943 root658Tail0 winner = winner :=
  (root658Step0).trans root658Scan1
set_option maxRecDepth 4096 in
theorem root658Branch : search edge 4 [4943] root658Candidates 4943 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4943] 4943 root658Tail0 winner = winner
  exact root658Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
