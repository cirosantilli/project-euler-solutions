import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root570Candidates : List Nat := Data.row4159
set_option maxRecDepth 8192 in
theorem root570Candidates_checked : candidates edge 5 0 4159 tail571 winner.bound = root570Candidates := by
  apply candidates_eq_row edge 5 0 4159 tail571 root570Candidates winner.bound tail571_ordered
  · decide
  · decide
  · funext q
    rfl
def root570Tail30 : List Nat := []
def root570Tail29 : List Nat := 21067 :: root570Tail30
def root570Tail28 : List Nat := 19681 :: root570Tail29
def root570Tail27 : List Nat := 19429 :: root570Tail28
def root570Tail26 : List Nat := 19417 :: root570Tail27
def root570Tail25 : List Nat := 18757 :: root570Tail26
def root570Tail24 : List Nat := 18397 :: root570Tail25
def root570Tail23 : List Nat := 18049 :: root570Tail24
def root570Tail22 : List Nat := 17839 :: root570Tail23
def root570Tail21 : List Nat := 17209 :: root570Tail22
def root570Tail20 : List Nat := 16759 :: root570Tail21
def root570Tail19 : List Nat := 15541 :: root570Tail20
def root570Tail18 : List Nat := 15121 :: root570Tail19
def root570Tail17 : List Nat := 14767 :: root570Tail18
def root570Tail16 : List Nat := 13513 :: root570Tail17
def root570Tail15 : List Nat := 13033 :: root570Tail16
def root570Tail14 : List Nat := 12163 :: root570Tail15
def root570Tail13 : List Nat := 12037 :: root570Tail14
def root570Tail12 : List Nat := 11677 :: root570Tail13
def root570Tail11 : List Nat := 11161 :: root570Tail12
def root570Tail10 : List Nat := 10069 :: root570Tail11
def root570Tail9 : List Nat := 9769 :: root570Tail10
def root570Tail8 : List Nat := 9643 :: root570Tail9
def root570Tail7 : List Nat := 8233 :: root570Tail8
def root570Tail6 : List Nat := 7681 :: root570Tail7
def root570Tail5 : List Nat := 7573 :: root570Tail6
def root570Tail4 : List Nat := 6661 :: root570Tail5
def root570Tail3 : List Nat := 5779 :: root570Tail4
def root570Tail2 : List Nat := 5011 :: root570Tail3
def root570Tail1 : List Nat := 4789 :: root570Tail2
def root570Tail0 : List Nat := 4621 :: root570Tail1
def root570Candidates0 : List Nat := [5779, 7573, 11161]
set_option maxRecDepth 8192 in
theorem root570Candidates0_checked : candidates edge 4 4159 4621 root570Tail1 winner.bound = root570Candidates0 := by
  unfold candidates
  have row : edge 4621 = (SortedNatTree.contains FastRows.row4621) := by
    change Data.row4621.contains = (SortedNatTree.contains FastRows.row4621)
    exact FastRows.row4621_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root570Child0 : search edge 3 [4159, 4621] root570Candidates0 8780 winner = winner := by decide
theorem root570Step0 : scan edge (search edge 3) 4 [4159] 4159 root570Tail0 winner =
    scan edge (search edge 3) 4 [4159] 4159 root570Tail1 winner := by
  rw [root570Tail0, scan, if_neg (by decide), root570Candidates0_checked]
  change scan edge (search edge 3) 4 [4159] 4159 root570Tail1
    (if 3 ≤ root570Candidates0.length then search edge 3 [4159, 4621] root570Candidates0 8780 winner else winner) = _
  rw [if_pos (by decide), root570Child0]
def root570Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root570Candidates1_checked : candidates edge 4 4159 4789 root570Tail2 winner.bound = root570Candidates1 := by
  unfold candidates
  have row : edge 4789 = (SortedNatTree.contains FastRows.row4789) := by
    change Data.row4789.contains = (SortedNatTree.contains FastRows.row4789)
    exact FastRows.row4789_checked.symm
  rw [row]
  decide
theorem root570Step1 : scan edge (search edge 3) 4 [4159] 4159 root570Tail1 winner =
    scan edge (search edge 3) 4 [4159] 4159 root570Tail2 winner := by
  rw [root570Tail1, scan, if_neg (by decide), root570Candidates1_checked]
  change scan edge (search edge 3) 4 [4159] 4159 root570Tail2
    (if 3 ≤ root570Candidates1.length then search edge 3 [4159, 4789] root570Candidates1 8948 winner else winner) = _
  rw [if_neg (by decide)]
def root570Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root570Candidates2_checked : candidates edge 4 4159 5011 root570Tail3 winner.bound = root570Candidates2 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
theorem root570Step2 : scan edge (search edge 3) 4 [4159] 4159 root570Tail2 winner =
    scan edge (search edge 3) 4 [4159] 4159 root570Tail3 winner := by
  rw [root570Tail2, scan, if_neg (by decide), root570Candidates2_checked]
  change scan edge (search edge 3) 4 [4159] 4159 root570Tail3
    (if 3 ≤ root570Candidates2.length then search edge 3 [4159, 5011] root570Candidates2 9170 winner else winner) = _
  rw [if_neg (by decide)]
theorem root570Scan3 : scan edge (search edge 3) 4 [4159] 4159 root570Tail3 winner = winner := by rw [root570Tail3, scan, if_pos (by decide)]
theorem root570Scan2 : scan edge (search edge 3) 4 [4159] 4159 root570Tail2 winner = winner :=
  (root570Step2).trans root570Scan3
theorem root570Scan1 : scan edge (search edge 3) 4 [4159] 4159 root570Tail1 winner = winner :=
  (root570Step1).trans root570Scan2
theorem root570Scan0 : scan edge (search edge 3) 4 [4159] 4159 root570Tail0 winner = winner :=
  (root570Step0).trans root570Scan1
set_option maxRecDepth 4096 in
theorem root570Branch : search edge 4 [4159] root570Candidates 4159 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4159] 4159 root570Tail0 winner = winner
  exact root570Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
