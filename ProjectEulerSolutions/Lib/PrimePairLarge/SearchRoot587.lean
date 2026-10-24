import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root587Candidates : List Nat := Data.row4297
set_option maxRecDepth 8192 in
theorem root587Candidates_checked : candidates edge 5 0 4297 tail588 winner.bound = root587Candidates := by
  apply candidates_eq_row edge 5 0 4297 tail588 root587Candidates winner.bound tail588_ordered
  · decide
  · decide
  · funext q
    rfl
def root587Tail29 : List Nat := []
def root587Tail28 : List Nat := 19681 :: root587Tail29
def root587Tail27 : List Nat := 18379 :: root587Tail28
def root587Tail26 : List Nat := 16747 :: root587Tail27
def root587Tail25 : List Nat := 15259 :: root587Tail26
def root587Tail24 : List Nat := 14869 :: root587Tail25
def root587Tail23 : List Nat := 14011 :: root587Tail24
def root587Tail22 : List Nat := 13693 :: root587Tail23
def root587Tail21 : List Nat := 13591 :: root587Tail22
def root587Tail20 : List Nat := 12697 :: root587Tail21
def root587Tail19 : List Nat := 12541 :: root587Tail20
def root587Tail18 : List Nat := 11593 :: root587Tail19
def root587Tail17 : List Nat := 11497 :: root587Tail18
def root587Tail16 : List Nat := 10111 :: root587Tail17
def root587Tail15 : List Nat := 9967 :: root587Tail16
def root587Tail14 : List Nat := 9619 :: root587Tail15
def root587Tail13 : List Nat := 9241 :: root587Tail14
def root587Tail12 : List Nat := 9133 :: root587Tail13
def root587Tail11 : List Nat := 8581 :: root587Tail12
def root587Tail10 : List Nat := 8317 :: root587Tail11
def root587Tail9 : List Nat := 8233 :: root587Tail10
def root587Tail8 : List Nat := 7213 :: root587Tail9
def root587Tail7 : List Nat := 6781 :: root587Tail8
def root587Tail6 : List Nat := 6607 :: root587Tail7
def root587Tail5 : List Nat := 6343 :: root587Tail6
def root587Tail4 : List Nat := 5689 :: root587Tail5
def root587Tail3 : List Nat := 5521 :: root587Tail4
def root587Tail2 : List Nat := 4507 :: root587Tail3
def root587Tail1 : List Nat := 4483 :: root587Tail2
def root587Tail0 : List Nat := 4441 :: root587Tail1
def root587Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root587Candidates0_checked : candidates edge 4 4297 4441 root587Tail1 winner.bound = root587Candidates0 := by
  unfold candidates
  have row : edge 4441 = (SortedNatTree.contains FastRows.row4441) := by
    change Data.row4441.contains = (SortedNatTree.contains FastRows.row4441)
    exact FastRows.row4441_checked.symm
  rw [row]
  decide
theorem root587Step0 : scan edge (search edge 3) 4 [4297] 4297 root587Tail0 winner =
    scan edge (search edge 3) 4 [4297] 4297 root587Tail1 winner := by
  rw [root587Tail0, scan, if_neg (by decide), root587Candidates0_checked]
  change scan edge (search edge 3) 4 [4297] 4297 root587Tail1
    (if 3 ≤ root587Candidates0.length then search edge 3 [4297, 4441] root587Candidates0 8738 winner else winner) = _
  rw [if_neg (by decide)]
def root587Candidates1 : List Nat := [6607]
set_option maxRecDepth 8192 in
theorem root587Candidates1_checked : candidates edge 4 4297 4483 root587Tail2 winner.bound = root587Candidates1 := by
  unfold candidates
  have row : edge 4483 = (SortedNatTree.contains FastRows.row4483) := by
    change Data.row4483.contains = (SortedNatTree.contains FastRows.row4483)
    exact FastRows.row4483_checked.symm
  rw [row]
  decide
theorem root587Step1 : scan edge (search edge 3) 4 [4297] 4297 root587Tail1 winner =
    scan edge (search edge 3) 4 [4297] 4297 root587Tail2 winner := by
  rw [root587Tail1, scan, if_neg (by decide), root587Candidates1_checked]
  change scan edge (search edge 3) 4 [4297] 4297 root587Tail2
    (if 3 ≤ root587Candidates1.length then search edge 3 [4297, 4483] root587Candidates1 8780 winner else winner) = _
  rw [if_neg (by decide)]
def root587Candidates2 : List Nat := [9967]
set_option maxRecDepth 8192 in
theorem root587Candidates2_checked : candidates edge 4 4297 4507 root587Tail3 winner.bound = root587Candidates2 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
theorem root587Step2 : scan edge (search edge 3) 4 [4297] 4297 root587Tail2 winner =
    scan edge (search edge 3) 4 [4297] 4297 root587Tail3 winner := by
  rw [root587Tail2, scan, if_neg (by decide), root587Candidates2_checked]
  change scan edge (search edge 3) 4 [4297] 4297 root587Tail3
    (if 3 ≤ root587Candidates2.length then search edge 3 [4297, 4507] root587Candidates2 8804 winner else winner) = _
  rw [if_neg (by decide)]
theorem root587Scan3 : scan edge (search edge 3) 4 [4297] 4297 root587Tail3 winner = winner := by rw [root587Tail3, scan, if_pos (by decide)]
theorem root587Scan2 : scan edge (search edge 3) 4 [4297] 4297 root587Tail2 winner = winner :=
  (root587Step2).trans root587Scan3
theorem root587Scan1 : scan edge (search edge 3) 4 [4297] 4297 root587Tail1 winner = winner :=
  (root587Step1).trans root587Scan2
theorem root587Scan0 : scan edge (search edge 3) 4 [4297] 4297 root587Tail0 winner = winner :=
  (root587Step0).trans root587Scan1
set_option maxRecDepth 4096 in
theorem root587Branch : search edge 4 [4297] root587Candidates 4297 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4297] 4297 root587Tail0 winner = winner
  exact root587Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
