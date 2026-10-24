import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root519Candidates : List Nat := Data.row3739
set_option maxRecDepth 8192 in
theorem root519Candidates_checked : candidates edge 5 0 3739 tail520 winner.bound = root519Candidates := by
  apply candidates_eq_row edge 5 0 3739 tail520 root519Candidates winner.bound tail520_ordered
  · decide
  · decide
  · funext q
    rfl
def root519Tail34 : List Nat := []
def root519Tail33 : List Nat := 22291 :: root519Tail34
def root519Tail32 : List Nat := 21283 :: root519Tail33
def root519Tail31 : List Nat := 21139 :: root519Tail32
def root519Tail30 : List Nat := 20749 :: root519Tail31
def root519Tail29 : List Nat := 20731 :: root519Tail30
def root519Tail28 : List Nat := 20563 :: root519Tail29
def root519Tail27 : List Nat := 20023 :: root519Tail28
def root519Tail26 : List Nat := 19333 :: root519Tail27
def root519Tail25 : List Nat := 19237 :: root519Tail26
def root519Tail24 : List Nat := 17749 :: root519Tail25
def root519Tail23 : List Nat := 17683 :: root519Tail24
def root519Tail22 : List Nat := 17377 :: root519Tail23
def root519Tail21 : List Nat := 16729 :: root519Tail22
def root519Tail20 : List Nat := 16363 :: root519Tail21
def root519Tail19 : List Nat := 14869 :: root519Tail20
def root519Tail18 : List Nat := 14071 :: root519Tail19
def root519Tail17 : List Nat := 14011 :: root519Tail18
def root519Tail16 : List Nat := 13807 :: root519Tail17
def root519Tail15 : List Nat := 12379 :: root519Tail16
def root519Tail14 : List Nat := 10993 :: root519Tail15
def root519Tail13 : List Nat := 9829 :: root519Tail14
def root519Tail12 : List Nat := 9781 :: root519Tail13
def root519Tail11 : List Nat := 9151 :: root519Tail12
def root519Tail10 : List Nat := 8539 :: root519Tail11
def root519Tail9 : List Nat := 8353 :: root519Tail10
def root519Tail8 : List Nat := 8233 :: root519Tail9
def root519Tail7 : List Nat := 6871 :: root519Tail8
def root519Tail6 : List Nat := 6763 :: root519Tail7
def root519Tail5 : List Nat := 6679 :: root519Tail6
def root519Tail4 : List Nat := 6637 :: root519Tail5
def root519Tail3 : List Nat := 5851 :: root519Tail4
def root519Tail2 : List Nat := 5233 :: root519Tail3
def root519Tail1 : List Nat := 5107 :: root519Tail2
def root519Tail0 : List Nat := 4099 :: root519Tail1
def root519Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root519Candidates0_checked : candidates edge 4 3739 4099 root519Tail1 winner.bound = root519Candidates0 := by
  unfold candidates
  have row : edge 4099 = (SortedNatTree.contains FastRows.row4099) := by
    change Data.row4099.contains = (SortedNatTree.contains FastRows.row4099)
    exact FastRows.row4099_checked.symm
  rw [row]
  decide
theorem root519Step0 : scan edge (search edge 3) 4 [3739] 3739 root519Tail0 winner =
    scan edge (search edge 3) 4 [3739] 3739 root519Tail1 winner := by
  rw [root519Tail0, scan, if_neg (by decide), root519Candidates0_checked]
  change scan edge (search edge 3) 4 [3739] 3739 root519Tail1
    (if 3 ≤ root519Candidates0.length then search edge 3 [3739, 4099] root519Candidates0 7838 winner else winner) = _
  rw [if_neg (by decide)]
def root519Candidates1 : List Nat := [6763]
set_option maxRecDepth 8192 in
theorem root519Candidates1_checked : candidates edge 4 3739 5107 root519Tail2 winner.bound = root519Candidates1 := by
  unfold candidates
  have row : edge 5107 = (SortedNatTree.contains FastRows.row5107) := by
    change Data.row5107.contains = (SortedNatTree.contains FastRows.row5107)
    exact FastRows.row5107_checked.symm
  rw [row]
  decide
theorem root519Step1 : scan edge (search edge 3) 4 [3739] 3739 root519Tail1 winner =
    scan edge (search edge 3) 4 [3739] 3739 root519Tail2 winner := by
  rw [root519Tail1, scan, if_neg (by decide), root519Candidates1_checked]
  change scan edge (search edge 3) 4 [3739] 3739 root519Tail2
    (if 3 ≤ root519Candidates1.length then search edge 3 [3739, 5107] root519Candidates1 8846 winner else winner) = _
  rw [if_neg (by decide)]
def root519Candidates2 : List Nat := [9829]
set_option maxRecDepth 8192 in
theorem root519Candidates2_checked : candidates edge 4 3739 5233 root519Tail3 winner.bound = root519Candidates2 := by
  unfold candidates
  have row : edge 5233 = (SortedNatTree.contains FastRows.row5233) := by
    change Data.row5233.contains = (SortedNatTree.contains FastRows.row5233)
    exact FastRows.row5233_checked.symm
  rw [row]
  decide
theorem root519Step2 : scan edge (search edge 3) 4 [3739] 3739 root519Tail2 winner =
    scan edge (search edge 3) 4 [3739] 3739 root519Tail3 winner := by
  rw [root519Tail2, scan, if_neg (by decide), root519Candidates2_checked]
  change scan edge (search edge 3) 4 [3739] 3739 root519Tail3
    (if 3 ≤ root519Candidates2.length then search edge 3 [3739, 5233] root519Candidates2 8972 winner else winner) = _
  rw [if_neg (by decide)]
theorem root519Scan3 : scan edge (search edge 3) 4 [3739] 3739 root519Tail3 winner = winner := by rw [root519Tail3, scan, if_pos (by decide)]
theorem root519Scan2 : scan edge (search edge 3) 4 [3739] 3739 root519Tail2 winner = winner :=
  (root519Step2).trans root519Scan3
theorem root519Scan1 : scan edge (search edge 3) 4 [3739] 3739 root519Tail1 winner = winner :=
  (root519Step1).trans root519Scan2
theorem root519Scan0 : scan edge (search edge 3) 4 [3739] 3739 root519Tail0 winner = winner :=
  (root519Step0).trans root519Scan1
set_option maxRecDepth 4096 in
theorem root519Branch : search edge 4 [3739] root519Candidates 3739 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3739] 3739 root519Tail0 winner = winner
  exact root519Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
