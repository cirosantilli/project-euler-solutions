import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root589Candidates : List Nat := Data.row4337
set_option maxRecDepth 8192 in
theorem root589Candidates_checked : candidates edge 5 0 4337 tail590 winner.bound = root589Candidates := by
  apply candidates_eq_row edge 5 0 4337 tail590 root589Candidates winner.bound tail590_ordered
  · decide
  · decide
  · funext q
    rfl
def root589Tail24 : List Nat := []
def root589Tail23 : List Nat := 21401 :: root589Tail24
def root589Tail22 : List Nat := 21107 :: root589Tail23
def root589Tail21 : List Nat := 20921 :: root589Tail22
def root589Tail20 : List Nat := 19037 :: root589Tail21
def root589Tail19 : List Nat := 18371 :: root589Tail20
def root589Tail18 : List Nat := 18251 :: root589Tail19
def root589Tail17 : List Nat := 18119 :: root589Tail18
def root589Tail16 : List Nat := 17981 :: root589Tail17
def root589Tail15 : List Nat := 17183 :: root589Tail16
def root589Tail14 : List Nat := 15473 :: root589Tail15
def root589Tail13 : List Nat := 13613 :: root589Tail14
def root589Tail12 : List Nat := 12269 :: root589Tail13
def root589Tail11 : List Nat := 11903 :: root589Tail12
def root589Tail10 : List Nat := 11321 :: root589Tail11
def root589Tail9 : List Nat := 9551 :: root589Tail10
def root589Tail8 : List Nat := 9293 :: root589Tail9
def root589Tail7 : List Nat := 7547 :: root589Tail8
def root589Tail6 : List Nat := 7043 :: root589Tail7
def root589Tail5 : List Nat := 5471 :: root589Tail6
def root589Tail4 : List Nat := 5237 :: root589Tail5
def root589Tail3 : List Nat := 5099 :: root589Tail4
def root589Tail2 : List Nat := 4889 :: root589Tail3
def root589Tail1 : List Nat := 4523 :: root589Tail2
def root589Tail0 : List Nat := 4421 :: root589Tail1
def root589Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root589Candidates0_checked : candidates edge 4 4337 4421 root589Tail1 winner.bound = root589Candidates0 := by
  unfold candidates
  have row : edge 4421 = (SortedNatTree.contains FastRows.row4421) := by
    change Data.row4421.contains = (SortedNatTree.contains FastRows.row4421)
    exact FastRows.row4421_checked.symm
  rw [row]
  decide
theorem root589Step0 : scan edge (search edge 3) 4 [4337] 4337 root589Tail0 winner =
    scan edge (search edge 3) 4 [4337] 4337 root589Tail1 winner := by
  rw [root589Tail0, scan, if_neg (by decide), root589Candidates0_checked]
  change scan edge (search edge 3) 4 [4337] 4337 root589Tail1
    (if 3 ≤ root589Candidates0.length then search edge 3 [4337, 4421] root589Candidates0 8758 winner else winner) = _
  rw [if_neg (by decide)]
def root589Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root589Candidates1_checked : candidates edge 4 4337 4523 root589Tail2 winner.bound = root589Candidates1 := by
  unfold candidates
  have row : edge 4523 = (SortedNatTree.contains FastRows.row4523) := by
    change Data.row4523.contains = (SortedNatTree.contains FastRows.row4523)
    exact FastRows.row4523_checked.symm
  rw [row]
  decide
theorem root589Step1 : scan edge (search edge 3) 4 [4337] 4337 root589Tail1 winner =
    scan edge (search edge 3) 4 [4337] 4337 root589Tail2 winner := by
  rw [root589Tail1, scan, if_neg (by decide), root589Candidates1_checked]
  change scan edge (search edge 3) 4 [4337] 4337 root589Tail2
    (if 3 ≤ root589Candidates1.length then search edge 3 [4337, 4523] root589Candidates1 8860 winner else winner) = _
  rw [if_neg (by decide)]
def root589Candidates2 : List Nat := [11903]
set_option maxRecDepth 8192 in
theorem root589Candidates2_checked : candidates edge 4 4337 4889 root589Tail3 winner.bound = root589Candidates2 := by
  unfold candidates
  have row : edge 4889 = (SortedNatTree.contains FastRows.row4889) := by
    change Data.row4889.contains = (SortedNatTree.contains FastRows.row4889)
    exact FastRows.row4889_checked.symm
  rw [row]
  decide
theorem root589Step2 : scan edge (search edge 3) 4 [4337] 4337 root589Tail2 winner =
    scan edge (search edge 3) 4 [4337] 4337 root589Tail3 winner := by
  rw [root589Tail2, scan, if_neg (by decide), root589Candidates2_checked]
  change scan edge (search edge 3) 4 [4337] 4337 root589Tail3
    (if 3 ≤ root589Candidates2.length then search edge 3 [4337, 4889] root589Candidates2 9226 winner else winner) = _
  rw [if_neg (by decide)]
def root589Candidates3 : List Nat := [5471]
set_option maxRecDepth 8192 in
theorem root589Candidates3_checked : candidates edge 4 4337 5099 root589Tail4 winner.bound = root589Candidates3 := by
  unfold candidates
  have row : edge 5099 = (SortedNatTree.contains FastRows.row5099) := by
    change Data.row5099.contains = (SortedNatTree.contains FastRows.row5099)
    exact FastRows.row5099_checked.symm
  rw [row]
  decide
theorem root589Step3 : scan edge (search edge 3) 4 [4337] 4337 root589Tail3 winner =
    scan edge (search edge 3) 4 [4337] 4337 root589Tail4 winner := by
  rw [root589Tail3, scan, if_neg (by decide), root589Candidates3_checked]
  change scan edge (search edge 3) 4 [4337] 4337 root589Tail4
    (if 3 ≤ root589Candidates3.length then search edge 3 [4337, 5099] root589Candidates3 9436 winner else winner) = _
  rw [if_neg (by decide)]
def root589Candidates4 : List Nat := [5471, 7043]
set_option maxRecDepth 8192 in
theorem root589Candidates4_checked : candidates edge 4 4337 5237 root589Tail5 winner.bound = root589Candidates4 := by
  unfold candidates
  have row : edge 5237 = (SortedNatTree.contains FastRows.row5237) := by
    change Data.row5237.contains = (SortedNatTree.contains FastRows.row5237)
    exact FastRows.row5237_checked.symm
  rw [row]
  decide
theorem root589Step4 : scan edge (search edge 3) 4 [4337] 4337 root589Tail4 winner =
    scan edge (search edge 3) 4 [4337] 4337 root589Tail5 winner := by
  rw [root589Tail4, scan, if_neg (by decide), root589Candidates4_checked]
  change scan edge (search edge 3) 4 [4337] 4337 root589Tail5
    (if 3 ≤ root589Candidates4.length then search edge 3 [4337, 5237] root589Candidates4 9574 winner else winner) = _
  rw [if_neg (by decide)]
theorem root589Scan5 : scan edge (search edge 3) 4 [4337] 4337 root589Tail5 winner = winner := by rw [root589Tail5, scan, if_pos (by decide)]
theorem root589Scan4 : scan edge (search edge 3) 4 [4337] 4337 root589Tail4 winner = winner :=
  (root589Step4).trans root589Scan5
theorem root589Scan3 : scan edge (search edge 3) 4 [4337] 4337 root589Tail3 winner = winner :=
  (root589Step3).trans root589Scan4
theorem root589Scan2 : scan edge (search edge 3) 4 [4337] 4337 root589Tail2 winner = winner :=
  (root589Step2).trans root589Scan3
theorem root589Scan1 : scan edge (search edge 3) 4 [4337] 4337 root589Tail1 winner = winner :=
  (root589Step1).trans root589Scan2
theorem root589Scan0 : scan edge (search edge 3) 4 [4337] 4337 root589Tail0 winner = winner :=
  (root589Step0).trans root589Scan1
set_option maxRecDepth 4096 in
theorem root589Branch : search edge 4 [4337] root589Candidates 4337 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4337] 4337 root589Tail0 winner = winner
  exact root589Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
