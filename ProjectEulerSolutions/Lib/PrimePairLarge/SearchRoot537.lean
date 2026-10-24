import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root537Candidates : List Nat := Data.row3907
set_option maxRecDepth 8192 in
theorem root537Candidates_checked : candidates edge 5 0 3907 tail538 winner.bound = root537Candidates := by
  apply candidates_eq_row edge 5 0 3907 tail538 root537Candidates winner.bound tail538_ordered
  · decide
  · decide
  · funext q
    rfl
def root537Tail28 : List Nat := []
def root537Tail27 : List Nat := 21589 :: root537Tail28
def root537Tail26 : List Nat := 21157 :: root537Tail27
def root537Tail25 : List Nat := 20593 :: root537Tail26
def root537Tail24 : List Nat := 20287 :: root537Tail25
def root537Tail23 : List Nat := 20149 :: root537Tail24
def root537Tail22 : List Nat := 19867 :: root537Tail23
def root537Tail21 : List Nat := 19051 :: root537Tail22
def root537Tail20 : List Nat := 18061 :: root537Tail21
def root537Tail19 : List Nat := 16747 :: root537Tail20
def root537Tail18 : List Nat := 16561 :: root537Tail19
def root537Tail17 : List Nat := 15991 :: root537Tail18
def root537Tail16 : List Nat := 15427 :: root537Tail17
def root537Tail15 : List Nat := 14503 :: root537Tail16
def root537Tail14 : List Nat := 13003 :: root537Tail15
def root537Tail13 : List Nat := 12919 :: root537Tail14
def root537Tail12 : List Nat := 12601 :: root537Tail13
def root537Tail11 : List Nat := 9619 :: root537Tail12
def root537Tail10 : List Nat := 8887 :: root537Tail11
def root537Tail9 : List Nat := 7573 :: root537Tail10
def root537Tail8 : List Nat := 7333 :: root537Tail9
def root537Tail7 : List Nat := 6997 :: root537Tail8
def root537Tail6 : List Nat := 6871 :: root537Tail7
def root537Tail5 : List Nat := 6469 :: root537Tail6
def root537Tail4 : List Nat := 6043 :: root537Tail5
def root537Tail3 : List Nat := 5431 :: root537Tail4
def root537Tail2 : List Nat := 4603 :: root537Tail3
def root537Tail1 : List Nat := 4327 :: root537Tail2
def root537Tail0 : List Nat := 4027 :: root537Tail1
def root537Candidates0 : List Nat := [6871, 15427]
set_option maxRecDepth 8192 in
theorem root537Candidates0_checked : candidates edge 4 3907 4027 root537Tail1 winner.bound = root537Candidates0 := by
  unfold candidates
  have row : edge 4027 = (SortedNatTree.contains FastRows.row4027) := by
    change Data.row4027.contains = (SortedNatTree.contains FastRows.row4027)
    exact FastRows.row4027_checked.symm
  rw [row]
  decide
theorem root537Step0 : scan edge (search edge 3) 4 [3907] 3907 root537Tail0 winner =
    scan edge (search edge 3) 4 [3907] 3907 root537Tail1 winner := by
  rw [root537Tail0, scan, if_neg (by decide), root537Candidates0_checked]
  change scan edge (search edge 3) 4 [3907] 3907 root537Tail1
    (if 3 ≤ root537Candidates0.length then search edge 3 [3907, 4027] root537Candidates0 7934 winner else winner) = _
  rw [if_neg (by decide)]
def root537Candidates1 : List Nat := [9619]
set_option maxRecDepth 8192 in
theorem root537Candidates1_checked : candidates edge 4 3907 4327 root537Tail2 winner.bound = root537Candidates1 := by
  unfold candidates
  have row : edge 4327 = (SortedNatTree.contains FastRows.row4327) := by
    change Data.row4327.contains = (SortedNatTree.contains FastRows.row4327)
    exact FastRows.row4327_checked.symm
  rw [row]
  decide
theorem root537Step1 : scan edge (search edge 3) 4 [3907] 3907 root537Tail1 winner =
    scan edge (search edge 3) 4 [3907] 3907 root537Tail2 winner := by
  rw [root537Tail1, scan, if_neg (by decide), root537Candidates1_checked]
  change scan edge (search edge 3) 4 [3907] 3907 root537Tail2
    (if 3 ≤ root537Candidates1.length then search edge 3 [3907, 4327] root537Candidates1 8234 winner else winner) = _
  rw [if_neg (by decide)]
def root537Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root537Candidates2_checked : candidates edge 4 3907 4603 root537Tail3 winner.bound = root537Candidates2 := by
  unfold candidates
  have row : edge 4603 = (SortedNatTree.contains FastRows.row4603) := by
    change Data.row4603.contains = (SortedNatTree.contains FastRows.row4603)
    exact FastRows.row4603_checked.symm
  rw [row]
  decide
theorem root537Step2 : scan edge (search edge 3) 4 [3907] 3907 root537Tail2 winner =
    scan edge (search edge 3) 4 [3907] 3907 root537Tail3 winner := by
  rw [root537Tail2, scan, if_neg (by decide), root537Candidates2_checked]
  change scan edge (search edge 3) 4 [3907] 3907 root537Tail3
    (if 3 ≤ root537Candidates2.length then search edge 3 [3907, 4603] root537Candidates2 8510 winner else winner) = _
  rw [if_neg (by decide)]
def root537Candidates3 : List Nat := [7573]
set_option maxRecDepth 8192 in
theorem root537Candidates3_checked : candidates edge 4 3907 5431 root537Tail4 winner.bound = root537Candidates3 := by
  unfold candidates
  have row : edge 5431 = (SortedNatTree.contains FastRows.row5431) := by
    change Data.row5431.contains = (SortedNatTree.contains FastRows.row5431)
    exact FastRows.row5431_checked.symm
  rw [row]
  decide
theorem root537Step3 : scan edge (search edge 3) 4 [3907] 3907 root537Tail3 winner =
    scan edge (search edge 3) 4 [3907] 3907 root537Tail4 winner := by
  rw [root537Tail3, scan, if_neg (by decide), root537Candidates3_checked]
  change scan edge (search edge 3) 4 [3907] 3907 root537Tail4
    (if 3 ≤ root537Candidates3.length then search edge 3 [3907, 5431] root537Candidates3 9338 winner else winner) = _
  rw [if_neg (by decide)]
theorem root537Scan4 : scan edge (search edge 3) 4 [3907] 3907 root537Tail4 winner = winner := by rw [root537Tail4, scan, if_pos (by decide)]
theorem root537Scan3 : scan edge (search edge 3) 4 [3907] 3907 root537Tail3 winner = winner :=
  (root537Step3).trans root537Scan4
theorem root537Scan2 : scan edge (search edge 3) 4 [3907] 3907 root537Tail2 winner = winner :=
  (root537Step2).trans root537Scan3
theorem root537Scan1 : scan edge (search edge 3) 4 [3907] 3907 root537Tail1 winner = winner :=
  (root537Step1).trans root537Scan2
theorem root537Scan0 : scan edge (search edge 3) 4 [3907] 3907 root537Tail0 winner = winner :=
  (root537Step0).trans root537Scan1
set_option maxRecDepth 4096 in
theorem root537Branch : search edge 4 [3907] root537Candidates 3907 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3907] 3907 root537Tail0 winner = winner
  exact root537Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
