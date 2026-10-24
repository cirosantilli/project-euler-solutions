import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root502Candidates : List Nat := Data.row3613
set_option maxRecDepth 8192 in
theorem root502Candidates_checked : candidates edge 5 0 3613 tail503 winner.bound = root502Candidates := by
  apply candidates_eq_row edge 5 0 3613 tail503 root502Candidates winner.bound tail503_ordered
  · decide
  · decide
  · funext q
    rfl
def root502Tail29 : List Nat := []
def root502Tail28 : List Nat := 21739 :: root502Tail29
def root502Tail27 : List Nat := 20269 :: root502Tail28
def root502Tail26 : List Nat := 18973 :: root502Tail27
def root502Tail25 : List Nat := 18229 :: root502Tail26
def root502Tail24 : List Nat := 17791 :: root502Tail25
def root502Tail23 : List Nat := 17041 :: root502Tail24
def root502Tail22 : List Nat := 15919 :: root502Tail23
def root502Tail21 : List Nat := 15313 :: root502Tail22
def root502Tail20 : List Nat := 15289 :: root502Tail21
def root502Tail19 : List Nat := 15061 :: root502Tail20
def root502Tail18 : List Nat := 14983 :: root502Tail19
def root502Tail17 : List Nat := 14851 :: root502Tail18
def root502Tail16 : List Nat := 14293 :: root502Tail17
def root502Tail15 : List Nat := 14071 :: root502Tail16
def root502Tail14 : List Nat := 14029 :: root502Tail15
def root502Tail13 : List Nat := 13159 :: root502Tail14
def root502Tail12 : List Nat := 10987 :: root502Tail13
def root502Tail11 : List Nat := 9613 :: root502Tail12
def root502Tail10 : List Nat := 9151 :: root502Tail11
def root502Tail9 : List Nat := 8887 :: root502Tail10
def root502Tail8 : List Nat := 8803 :: root502Tail9
def root502Tail7 : List Nat := 8353 :: root502Tail8
def root502Tail6 : List Nat := 5821 :: root502Tail7
def root502Tail5 : List Nat := 5683 :: root502Tail6
def root502Tail4 : List Nat := 5623 :: root502Tail5
def root502Tail3 : List Nat := 5233 :: root502Tail4
def root502Tail2 : List Nat := 5023 :: root502Tail3
def root502Tail1 : List Nat := 4831 :: root502Tail2
def root502Tail0 : List Nat := 3697 :: root502Tail1
def root502Candidates0 : List Nat := [8803]
set_option maxRecDepth 8192 in
theorem root502Candidates0_checked : candidates edge 4 3613 3697 root502Tail1 winner.bound = root502Candidates0 := by
  unfold candidates
  have row : edge 3697 = (SortedNatTree.contains FastRows.row3697) := by
    change Data.row3697.contains = (SortedNatTree.contains FastRows.row3697)
    exact FastRows.row3697_checked.symm
  rw [row]
  decide
theorem root502Step0 : scan edge (search edge 3) 4 [3613] 3613 root502Tail0 winner =
    scan edge (search edge 3) 4 [3613] 3613 root502Tail1 winner := by
  rw [root502Tail0, scan, if_neg (by decide), root502Candidates0_checked]
  change scan edge (search edge 3) 4 [3613] 3613 root502Tail1
    (if 3 ≤ root502Candidates0.length then search edge 3 [3613, 3697] root502Candidates0 7310 winner else winner) = _
  rw [if_neg (by decide)]
def root502Candidates1 : List Nat := [5623]
set_option maxRecDepth 8192 in
theorem root502Candidates1_checked : candidates edge 4 3613 4831 root502Tail2 winner.bound = root502Candidates1 := by
  unfold candidates
  have row : edge 4831 = (SortedNatTree.contains FastRows.row4831) := by
    change Data.row4831.contains = (SortedNatTree.contains FastRows.row4831)
    exact FastRows.row4831_checked.symm
  rw [row]
  decide
theorem root502Step1 : scan edge (search edge 3) 4 [3613] 3613 root502Tail1 winner =
    scan edge (search edge 3) 4 [3613] 3613 root502Tail2 winner := by
  rw [root502Tail1, scan, if_neg (by decide), root502Candidates1_checked]
  change scan edge (search edge 3) 4 [3613] 3613 root502Tail2
    (if 3 ≤ root502Candidates1.length then search edge 3 [3613, 4831] root502Candidates1 8444 winner else winner) = _
  rw [if_neg (by decide)]
def root502Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root502Candidates2_checked : candidates edge 4 3613 5023 root502Tail3 winner.bound = root502Candidates2 := by
  unfold candidates
  have row : edge 5023 = (SortedNatTree.contains FastRows.row5023) := by
    change Data.row5023.contains = (SortedNatTree.contains FastRows.row5023)
    exact FastRows.row5023_checked.symm
  rw [row]
  decide
theorem root502Step2 : scan edge (search edge 3) 4 [3613] 3613 root502Tail2 winner =
    scan edge (search edge 3) 4 [3613] 3613 root502Tail3 winner := by
  rw [root502Tail2, scan, if_neg (by decide), root502Candidates2_checked]
  change scan edge (search edge 3) 4 [3613] 3613 root502Tail3
    (if 3 ≤ root502Candidates2.length then search edge 3 [3613, 5023] root502Candidates2 8636 winner else winner) = _
  rw [if_neg (by decide)]
def root502Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root502Candidates3_checked : candidates edge 4 3613 5233 root502Tail4 winner.bound = root502Candidates3 := by
  unfold candidates
  have row : edge 5233 = (SortedNatTree.contains FastRows.row5233) := by
    change Data.row5233.contains = (SortedNatTree.contains FastRows.row5233)
    exact FastRows.row5233_checked.symm
  rw [row]
  decide
theorem root502Step3 : scan edge (search edge 3) 4 [3613] 3613 root502Tail3 winner =
    scan edge (search edge 3) 4 [3613] 3613 root502Tail4 winner := by
  rw [root502Tail3, scan, if_neg (by decide), root502Candidates3_checked]
  change scan edge (search edge 3) 4 [3613] 3613 root502Tail4
    (if 3 ≤ root502Candidates3.length then search edge 3 [3613, 5233] root502Candidates3 8846 winner else winner) = _
  rw [if_neg (by decide)]
theorem root502Scan4 : scan edge (search edge 3) 4 [3613] 3613 root502Tail4 winner = winner := by rw [root502Tail4, scan, if_pos (by decide)]
theorem root502Scan3 : scan edge (search edge 3) 4 [3613] 3613 root502Tail3 winner = winner :=
  (root502Step3).trans root502Scan4
theorem root502Scan2 : scan edge (search edge 3) 4 [3613] 3613 root502Tail2 winner = winner :=
  (root502Step2).trans root502Scan3
theorem root502Scan1 : scan edge (search edge 3) 4 [3613] 3613 root502Tail1 winner = winner :=
  (root502Step1).trans root502Scan2
theorem root502Scan0 : scan edge (search edge 3) 4 [3613] 3613 root502Tail0 winner = winner :=
  (root502Step0).trans root502Scan1
set_option maxRecDepth 4096 in
theorem root502Branch : search edge 4 [3613] root502Candidates 3613 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3613] 3613 root502Tail0 winner = winner
  exact root502Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
